import sys
sys.path.append("/home/guihaokun/Time-Series-Pretrain")

from utils.constants import INCLUDED_CHANNELS
from utils.tools import resampleData, getEDFsignals, getOrderedChannels
from tqdm import tqdm
import argparse
import numpy as np
import os
import pyedflib
import pickle

def resample_all(raw_edf_dir, save_dir):
    edf_files = []
    for path, subdirs, files in os.walk(raw_edf_dir):
        for name in files:
            if ".edf" in name:
                edf_files.append(os.path.join(path, name))

    means = []
    stds = []
    for idx in tqdm(range(len(edf_files))):
        edf_fn = edf_files[idx]
        try:
            f = pyedflib.EdfReader(edf_fn)

            orderedChannels = getOrderedChannels(
                edf_fn, False, f.getSignalLabels(), INCLUDED_CHANNELS
            )
            signals = getEDFsignals(f)
            signal_array = np.array(signals[orderedChannels, :])

            mean = np.mean(signal_array, axis=1)
            std = np.std(signal_array, axis=1)
            means.append(mean)
            stds.append(std)
            f.close()

        except Exception as e:
            print("Error occur:", str(e))

    means = np.array(means)
    stds = np.array(stds)
    total_mean = np.mean(means, axis=0)
    total_std = np.mean(stds, axis=0)
    with open(os.path.join(save_dir, "means_seq2seq_nofft.pkl"), "wb") as f:
        pickle.dump(total_mean, f)
    with open(os.path.join(save_dir, "stds_seq2seq_nofft.pkl"), "wb") as f:
        pickle.dump(total_std, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("cal_mean_std")
    parser.add_argument(
        "--raw_edf_dir",
        type=str,
        default="<raw_edf_dir>",
        help="Full path to raw edf files.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="<save_dir>",
        help="Full path to save mean and std.",
    )
    args = parser.parse_args()

    resample_all(args.raw_edf_dir, args.save_dir)