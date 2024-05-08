import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from scipy.signal import resample
from torch.distributed import init_process_group, destroy_process_group

plt.switch_backend('agg')

class EarlyStopping:
    def __init__(self, patience=3, verbose=False, delta=0, if_max=False, device=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.if_max = if_max
        self.device = device

    def __call__(self, val_loss):
        if self.device is not None and self.device != 0:
            return
        if self.patience == 0:
            return
        if self.if_max:
            score = val_loss
        else:
            score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


class StandardScaler:
    """
    Standardize the input
    """

    def __init__(self, mean, std, device=None):
        self.mean = mean  # (1,num_nodes,1)
        self.std = std  # (1,num_nodes,1)
        self._device = device

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data, is_tensor=False):
        """
        Masked inverse transform
        Args:
            data: data for inverse scaling
            is_tensor: whether data is a tensor
        """
        mean = self.mean.copy()
        std = self.std.copy()
        if len(mean.shape) == 0:
            mean = [mean]
            std = [std]
        if is_tensor:
            mean = torch.FloatTensor(mean).to(self._device)
            std = torch.FloatTensor(std).to(self._device)
        return (data * std + mean)
    

def compute_sampling_threshold(cl_decay_steps, global_step):
    """
    Compute scheduled sampling threshold
    """
    return cl_decay_steps / \
        (cl_decay_steps + np.exp(global_step / cl_decay_steps))

def getOrderedChannels(file_name, verbose, labels_object, channel_names):
    labels = list(labels_object)
    for i in range(len(labels)):
        labels[i] = labels[i].split("-")[0]

    ordered_channels = []
    for ch in channel_names:
        try:
            ordered_channels.append(labels.index(ch))
        except:
            if verbose:
                print(file_name + " failed to get channel " + ch)
            raise Exception("channel not match")
    return ordered_channels


def getEDFsignals(edf):
    """
    Get EEG signal in edf file
    Args:
        edf: edf object
    Returns:
        signals: shape (num_channels, num_data_points)
    """
    n = edf.signals_in_file
    samples = edf.getNSamples()[0]
    signals = np.zeros((n, samples))
    for i in range(n):
        try:
            signals[i, :] = edf.readSignal(i)
        except:
            pass
    return signals


def resampleData(signals, to_freq=200, window_size=4):
    """
    Resample signals from its original sampling freq to another freq
    Args:
        signals: EEG signal slice, (num_channels, num_data_points)
        to_freq: Re-sampled frequency in Hz
        window_size: time window in seconds
    Returns:
        resampled: (num_channels, resampled_data_points)
    """
    num = int(to_freq * window_size)
    resampled = resample(signals, num=num, axis=1)
    return resampled


def seed_torch(seed=1029):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) 
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def ddp_cleanup():
    destroy_process_group()

class WriterFilter:
    def __init__(self, working_class):
        self.working_class = working_class
        self.filter_methods = lambda : int(os.environ["LOCAL_RANK"]) == 0

    def __getattr__(self, name):
        # rewrite __getattr__ methods
        if self.filter_methods():
            return getattr(self.working_class, name)
        else:
            return self._filter_method
    def _filter_method(self, *args, **kwargs):
        return
