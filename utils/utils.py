import torch
import os
import shutil
import queue
from utils.constants import INCLUDED_CHANNELS
import numpy as np
from typing import Optional, Tuple

class CheckpointSaver:
    """Class to save and load model checkpoints.
    Save the best checkpoints as measured by a metric value passed into the
    `save` method. Overwrite checkpoints with better checkpoints once
    `max_checkpoints` have been saved.
    Args:
        save_dir (str): Directory to save checkpoints.
        metric_name (str): Name of metric used to determine best model.
        maximize_metric (bool): If true, best checkpoint is that which maximizes
            the metric value passed in via `save`. Otherwise, best checkpoint
            minimizes the metric.
        log (logging.Logger): Optional logger for printing information.
    """

    def __init__(self, save_dir, metric_name, maximize_metric=False, fn=lambda : int(os.environ["LOCAL_RANK"])==0):
        super(CheckpointSaver, self).__init__()

        self.save_dir = save_dir
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_val = None
        self.ckpt_paths = queue.PriorityQueue()
        self.fn = fn

    def is_best(self, metric_val):
        """Check whether `metric_val` is the best seen so far.
        Args:
            metric_val (float): Metric value to compare to prior checkpoints.
        """
        if metric_val is None:
            # No metric reported
            return False

        if self.best_val is None:
            # No checkpoint saved yet
            return True

        return ((self.maximize_metric and self.best_val <= metric_val)
                or (not self.maximize_metric and self.best_val >= metric_val))

    def save(self, epoch, model, optimizer, metric_val):
        """Save model parameters to disk.
        Args:
            epoch (int): Current epoch.
            model (torch.nn.DataParallel): Model to save.
            optimizer: optimizer
            metric_val (float): Determines whether checkpoint is best so far.
        """
        if not self.fn():
            return
        ckpt_dict = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }

        checkpoint_path = os.path.join(self.save_dir, 'last.pth.tar')
        torch.save(ckpt_dict, checkpoint_path)

        best_path = ''
        if self.is_best(metric_val):
            # Save the best model
            self.best_val = metric_val
            best_path = os.path.join(self.save_dir, 'best.pth.tar')
            shutil.copy(checkpoint_path, best_path)


def load_model_checkpoint(checkpoint_file, model, optimizer=None, map_location=None, if_strict=False):
    if map_location is not None:
        loc = f'cuda:{map_location}'
        checkpoint = torch.load(checkpoint_file, map_location=loc)
    else:
        checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint['model_state'], strict=if_strict)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        return model, optimizer
    return model

def last_relevant_pytorch(output, lengths, batch_first=True):
    # masks of the true seq lengths
    masks = (lengths - 1).view(-1, 1).expand(len(lengths), output.size(2)).cuda()
    time_dimension = 1 if batch_first else 0
    masks = masks.unsqueeze(time_dimension)
    last_output = output.gather(time_dimension, masks).squeeze(time_dimension).cuda()

    return last_output

def get_swap_pairs(channels):
    """
    Swap select adjacenet channels
    Args:
        channels: list of channel names
    Returns:
        list of tuples, each a pair of channel indices being swapped
    """
    swap_pairs = []
    if ("EEG FP1" in channels) and ("EEG FP2" in channels):
        swap_pairs.append([channels.index("EEG FP1"), channels.index("EEG FP2")])
    if ("EEG Fp1" in channels) and ("EEG Fp2" in channels):
        swap_pairs.append([channels.index("EEG Fp1"), channels.index("EEG Fp2")])
    if ("EEG F3" in channels) and ("EEG F4" in channels):
        swap_pairs.append([channels.index("EEG F3"), channels.index("EEG F4")])
    if ("EEG F7" in channels) and ("EEG F8" in channels):
        swap_pairs.append([channels.index("EEG F7"), channels.index("EEG F8")])
    if ("EEG C3" in channels) and ("EEG C4" in channels):
        swap_pairs.append([channels.index("EEG C3"), channels.index("EEG C4")])
    if ("EEG T3" in channels) and ("EEG T4" in channels):
        swap_pairs.append([channels.index("EEG T3"), channels.index("EEG T4")])
    if ("EEG T5" in channels) and ("EEG T6" in channels):
        swap_pairs.append([channels.index("EEG T5"), channels.index("EEG T6")])
    if ("EEG O1" in channels) and ("EEG O2" in channels):
        swap_pairs.append([channels.index("EEG O1"), channels.index("EEG O2")])

    return swap_pairs

def getOriginalData(x:torch.Tensor, isAug:torch.Tensor):
    x_new = x.clone().cuda()
    change_channels = torch.Tensor(get_swap_pairs(INCLUDED_CHANNELS)).int()
    channel_1 = change_channels[:, 0]
    channel_2 = change_channels[:, 1]
    x_new[:, channel_1, :] = x[:, channel_2, :]
    x_new[:, channel_2, :] = x[:, channel_1, :]
    isAug = isAug.clone().reshape(-1, 1, 1).cuda()
    return x_new * isAug + x * (1 - isAug)


def compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[torch.Tensor],
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
    require_same_masks: bool = True,
    mask_dropout: float = 0.0,
    add_masks: bool = False,
    seed: Optional[int] = None,
    epoch: Optional[int] = None,
    indices: Optional[torch.Tensor] = None,
) -> np.ndarray:
    """
    Computes random mask spans for a given shape

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
        require_same_masks: if true, will randomly drop out masks until same amount of masks remains in each sample
        mask_dropout: randomly dropout this percentage of masks in each example
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    mask_idcs = []
    for i in range(bsz):
        if seed is not None and epoch is not None and indices is not None:
            seed_i = int(hash((seed, epoch, indices[i].item())) % 1e6)
        else:
            seed_i = None

        rng = np.random.default_rng(seed_i)

        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            assert sz >= 0, sz
        else:
            sz = all_sz

        num_mask = int(
            # add a random number for probabilistic rounding
            mask_prob * sz / float(mask_length)
            + rng.random()
        )
        num_mask = max(min_masks, num_mask)

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = rng.integers(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = rng.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = rng.poisson(mask_length, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            if mask_type == "static":
                raise ValueError(f"this should never happens")
            else:
                lengths = [min(mask_length, sz - 1)]

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = rng.integers(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = rng.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            mask_idc = rng.choice(sz, num_mask, replace=False)
            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idc = np.unique(mask_idc[mask_idc < sz])
        if len(mask_idc) >= sz:
            raise ValueError(
                (
                    f"the entire sequence is masked. "
                    f"sz={sz}; mask_idc[mask_idc]; "
                    f"index={indices[i] if indices is not None else None}"
                )
            )
        mask_idcs.append(mask_idc)

    target_len = None
    if require_same_masks:
        if add_masks:
            target_len = max([len(m) for m in mask_idcs])
        else:
            target_len = min([len(m) for m in mask_idcs])

    for i, mask_idc in enumerate(mask_idcs):
        if target_len is not None and len(mask_idc) > target_len:
            mask_idc = rng.choice(mask_idc, target_len, replace=False)

        mask[i, mask_idc] = True

        if target_len is not None and len(mask_idc) < target_len:
            unmasked = np.flatnonzero(~mask[i])
            to_mask = rng.choice(unmasked, target_len - len(mask_idc), replace=False)
            mask[i, to_mask] = True

        if mask_dropout > 0:
            masked = np.flatnonzero(mask[i])
            num_holes = np.rint(len(masked) * mask_dropout).astype(int)
            to_drop = rng.choice(masked, num_holes, replace=False)
            mask[i, to_drop] = False

    return mask
