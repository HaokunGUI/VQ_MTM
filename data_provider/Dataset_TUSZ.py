import argparse
from torch.utils.data import Dataset
import warnings
import os
from typing import Any, Tuple
import h5py
import torch
import numpy as np
from utils.constants import INCLUDED_CHANNELS
from utils.utils import get_swap_pairs

warnings.filterwarnings('ignore')


class Dataset_TUSZ(Dataset):
    def __init__(self, args:argparse.Namespace, scalar):
        super(Dataset_TUSZ, self).__init__()
        self.task_name = args.task_name
        self.root_path = args.root_path
        self.marker_dir = args.marker_dir
        self.input_len = args.input_len
        self.output_len = args.output_len
        self.split = args.split
        self.data_augment = args.data_augment
        self.use_graph = args.use_graph
        self.args = args
        self.scalar = scalar

        marker_dir = f'file_markers_{self.task_name}'
        self.marker_dir = os.path.join(self.marker_dir, marker_dir)
        if self.task_name == 'ssl':
            file_name = f'{self.split}Set_seq2seq_{self.input_len}s.txt'
            file_path = os.path.join(self.marker_dir, file_name)
            with open(file_path, 'r') as f:
                f_str = f.readlines()
                self.file_tuples = [f_str[i].strip().split(',') for i in range(len(f_str))]
                self.size = len(self.file_tuples)

        elif self.task_name == 'anomaly_detection':
            nosz_file_name = f'{self.split}Set_seq2seq_{self.input_len}s_nosz.txt'
            sz_file_name = f'{self.split}Set_seq2seq_{self.input_len}s_sz.txt'
            with open(os.path.join(self.marker_dir, nosz_file_name), 'r') as f_nosz:
                with open(os.path.join(self.marker_dir, sz_file_name), 'r') as f_sz:
                    f_nosz_str = f_nosz.readlines()
                    f_sz_str = f_sz.readlines()
            if self.split == 'train' and self.args.balanced:
                num_points = int(self.args.scale_ratio * len(f_sz_str))
                np.random.shuffle(f_nosz_str)
                f_nosz_str = f_nosz_str[:num_points]
                np.random.shuffle(f_sz_str)
                f_sz_str = f_sz_str[:num_points]
            f_combine_str = f_nosz_str + f_sz_str
            np.random.shuffle(f_combine_str)
            self.file_tuples = []
            for i in range(len(f_combine_str)):
                tup = f_combine_str[i].strip('\n').split(',')
                tup[1] = int(tup[1])
                self.file_tuples.append(tup)
            self.size = len(self.file_tuples)

        elif self.task_name == 'classification':
            file_name = f'{self.split}_{self.input_len}s.h5'
            file_path = os.path.join(self.args.classification_dir, file_name)
            with h5py.File(file_path, 'r') as f:
                self.clips = f['clips'][()]
                self.labels = f['labels'][()]
                self.paddings = f['paddings'][()]
            
            # for i in range(4):
            #     print(f'num of class {i}: {np.sum(self.labels == i)}')
            self.size = len(self.labels)

        else:
            raise NotImplementedError


    def __getitem__(self, index: int) -> Any:
        if self.task_name == 'ssl':
            file_name_tuple = self.file_tuples[index]

            x, y = self._getIdx2Slice(file_name_tuple)

            if self.data_augment:
                # reflect or not reflect for both x and y
                reflect = np.random.choice([True, False])
                x = self._random_reflect(x, reflect=reflect)
                y = self._random_reflect(y, reflect=reflect)
            
                # scale by the same factor for both x and y            
                scale_factor = np.random.uniform(0.8, 1.2)
                x = self._random_scale(x, scale_factor=scale_factor)
                y = self._random_scale(y, scale_factor=scale_factor)

            if self.scalar is not None:
                x = self.scalar.transform(x)
                y = self.scalar.transform(y)
            # convert to Tensor
            x = torch.Tensor(x)
            y = torch.Tensor(y)

            return x, y, int(self.data_augment and reflect)
        
        elif self.task_name == 'anomaly_detection':
            file_name, label = self.file_tuples[index]
            x = self._getSlice(file_name)
            if self.data_augment:
                # reflect or not reflect for both x and y
                reflect = np.random.choice([True, False])
                x = self._random_reflect(x, reflect=reflect)
            
                # scale by the same factor for both x and y            
                scale_factor = np.random.uniform(0.8, 1.2)
                x = self._random_scale(x, scale_factor=scale_factor)

            if self.scalar is not None:
                x = self.scalar.transform(x)
            # convert to Tensor
            x = torch.Tensor(x)
            y = torch.Tensor([label])
            return x, y, int(self.data_augment and reflect)

        elif self.task_name == 'classification':
            x, y, padding = self.clips[index], self.labels[index], self.paddings[index]
            if self.scalar is not None:
                x = self.scalar.transform(x)
            
            x = torch.Tensor(x)
            y = torch.Tensor([y])
            # set the paddings to 0
            x[:, padding == 1] = 0
            return x, y, int(False)

        else:
            raise NotImplementedError
        
    

    def __len__(self) -> int: 
        return self.size

    def _getIdx2Slice(self, file_name_tuple: Tuple[str]):
        file_name_i, file_name_o = file_name_tuple

        slice_num_i = int(file_name_i.split('_')[-1].split('.h5')[0])
        slice_num_o = int(file_name_o.split('_')[-1].split('.h5')[0])
        assert slice_num_o == slice_num_i + 1, 'slice_num_o should be equal to slice_num_i + 1'

        file_name_i = file_name_i.split('.edf')[0] + '.h5'
        file_name_o = file_name_o.split('.edf')[0] + '.h5'
        assert file_name_i == file_name_o, 'file_name_i should be equal to file_name_o'

        file_path = os.path.join(self.root_path, file_name_i)
        with h5py.File(file_path, 'r') as f:
            signals = f["resample_signal"][()]
            freq = f["resample_freq"][()]
        input_node_num = int(freq * self.input_len)
        output_node_num = int(freq * self.output_len)

        start_window = input_node_num * slice_num_i
        end_window = start_window + input_node_num + output_node_num
        # (num_channel, input_len)
        input_slice = signals[:, start_window:end_window]
        return input_slice[:, :input_node_num], input_slice[:, input_node_num:]
    
    def _getSlice(self, file_name:str):
        slice_num = int(file_name.split('_')[-1].split('.h5')[0])
        file_name = file_name.split('.edf')[0] + '.h5'
        file_path = os.path.join(self.root_path, file_name)
        with h5py.File(file_path, 'r') as f:
            signals = f["resample_signal"][()]
            freq = f["resample_freq"][()]
        input_node_num = int(freq * self.input_len)

        start_window = input_node_num * slice_num
        end_window = start_window + input_node_num
        return signals[:, start_window:end_window]

    def _random_reflect(self, EEG_seq, reflect=False):
        """
        Randomly reflect EEG channels along the midline
        """
        swap_pairs = get_swap_pairs(INCLUDED_CHANNELS)
        EEG_seq_reflect = EEG_seq.copy()
        if reflect:            
            for pair in swap_pairs:
                EEG_seq_reflect[[pair[0],pair[1]],:] = EEG_seq[[pair[1], pair[0]],:]
        return EEG_seq_reflect

    def _random_scale(self, EEG_seq, scale_factor=None):
        """
        Scale EEG signals by a random number between 0.8 and 1.2
        """
        if scale_factor is None:
            scale_factor = np.random.uniform(0.8, 1.2)
        EEG_seq *= scale_factor
        return EEG_seq