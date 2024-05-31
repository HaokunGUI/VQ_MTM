import sys
sys.path.append("/home/guihaokun/Time-Series-Pretrain")

from utils.constants import INCLUDED_CHANNELS
from utils.tools import resampleData, getEDFsignals, getOrderedChannels
from tqdm import tqdm
import argparse
import numpy as np
import os
import pyedflib
import h5py

def resample_all(raw_edf_dir, save_dir, freq:int=None):
    os.makedirs(save_dir, exist_ok=True)
    edf_files = []
    for path, subdirs, files in os.walk(raw_edf_dir):
        for name in files:
            if ".edf" in name:
                edf_files.append(os.path.join(path, name))

    failed_files = []
    for idx in tqdm(range(len(edf_files))):
        edf_fn = edf_files[idx]

        save_fn = os.path.join(save_dir, edf_fn.split("/")[-1].split(".edf")[0] + ".h5")
        if os.path.exists(save_fn):
            continue
        try:
            f = pyedflib.EdfReader(edf_fn)

            sample_freq = f.getSampleFrequency(0)
            freq = sample_freq if freq is None else freq
            resample = True if sample_freq != freq else False

            orderedChannels = getOrderedChannels(
                edf_fn, False, f.getSignalLabels(), INCLUDED_CHANNELS
            )
            signals = getEDFsignals(f)
            signal_array = np.array(signals[orderedChannels, :]).copy()

            ordered_channel_freqs = set()
            for channel in orderedChannels:
                channel_freq = f.getSampleFrequency(channel)
                ordered_channel_freqs.add(channel_freq)
            f.close()

            if len(ordered_channel_freqs) != 1:
                resample = True
                print('resample in file:', edf_fn)
            else:
                resample = False
            if resample:
                signal_array = resampleData(
                    signal_array,
                    to_freq=freq,
                    window_size=int(signal_array.shape[1] / sample_freq),
                )

            with h5py.File(save_fn, "w") as hf:
                hf.create_dataset("resample_signal", data=signal_array.copy())
                hf.create_dataset("resample_freq", data=freq)

                del signal_array

        except Exception as e:
            print("Error occur:", str(e), 'fail file number:', len(failed_files))
            failed_files.append(edf_fn)

    print("DONE. {} files failed.".format(len(failed_files)))
    with open("failed_files.txt", "w") as f:
        for fn in failed_files:
            f.write(fn + "\n")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Resample.")
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
        help="Full path to dir to save resampled signals.",
    )
    args = parser.parse_args()

    resample_all(args.raw_edf_dir, args.save_dir)
