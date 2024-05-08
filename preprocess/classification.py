import h5py
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

map_dict = {
    'fnsz': 0, # FNSZ -> (Combined Focal)CF
    'spsz': 0, # SPSZ -> (Combined Focal)CF
    'cpsz': 0, # CPSZ -> (Combined Focal)CF

    'gnsz': 1, # GNSZ -> (Generalized Non-specific)GN

    'absz': 2, # ABSZ -> (Absence Seizure)AS

    'tnsz': 3, # TNSZ -> (Combined Tonic)CT
    'tcsz': 3, # TCSZ -> (Combined Tonic)CT
}

def get_seizure(file_path: str, processed_dir: str, clip_len: int) -> tuple:
    '''
    Args:
    file_path: path to the csv file
    '''
    # offset is sure
    offset = 2

    # get the edf info
    file_name = file_path.split("/")[-1]
    h5_file = file_name.split(".csv")[0] + ".h5"
    h5_path = os.path.join(processed_dir, h5_file)
    with h5py.File(h5_path, "r") as hf:
        signal_array = hf["resample_signal"][()]
        freq = hf["resample_freq"][()]
        physical_clip_len = int(freq * clip_len)

    def get_seizure_time(file_name: str):
        csv_file = file_name.split(".csv")[0] + ".csv_bi"
        seizure_times = []
        with open(csv_file) as f:
            for line in f.readlines():
                if "seiz" in line:  # if seizure
                    # seizure start and end time
                    seizure_times.append(
                        [
                            float(line.strip().split(",")[1]),
                            float(line.strip().split(",")[2]),
                        ]
                    )
        return seizure_times
    
    seizure_time = get_seizure_time(file_path)

    df = pd.read_csv(file_path, skiprows=5)

    # drop the channel 
    df = df.drop('channel', axis=1, errors='ignore')
    df = df.drop_duplicates(subset=['start_time', 'stop_time', 'label'])

    df = df[df['label'].isin(map_dict.keys())]
    df['label'] = df['label'].map(map_dict)

    clips = []
    paddings = []
    labels = []

    for i, (start_time, stop_time) in enumerate(seizure_time):
        select_row = df[(df['start_time'] >= start_time) & (df['stop_time'] <= stop_time)]
        if not select_row.empty:
            label = select_row['label'].values[0]

            if i > 0:
                pre_seizure_end = int(seizure_time[i - 1][1] * freq)
            else:
                pre_seizure_end = 0
            
            start_t = max(int(pre_seizure_end + 1), int((start_time - offset) * freq))
            stop_t = int(stop_time * freq)

            signal = signal_array[:, start_t:stop_t]
            padding_len = physical_clip_len - signal.shape[1] % physical_clip_len
            signal = np.concatenate([signal, np.zeros((19, padding_len))], axis=1)
            assert signal.shape[1] % physical_clip_len == 0, 'signal.shape[1] % physical_clip_len != 0'

            start_time_step = 0
            
            while start_time_step <= signal.shape[1] - physical_clip_len:
                # create paddings
                if start_time_step + physical_clip_len == signal.shape[1]:
                    padding = np.concatenate([[0]*(physical_clip_len - padding_len), [1]*padding_len], axis=0)
                else:
                    padding = np.zeros(physical_clip_len)

                end_time_step = start_time_step + physical_clip_len
                curr_time_step = signal[:, start_time_step:end_time_step]
                clips.append(curr_time_step)
                labels.append(label)
                paddings.append(padding)
                start_time_step = end_time_step
    return clips, labels, paddings

def preprocess(raw_dir: str, processed_data: str, output_dir: str, slice_len: int):
    '''
    raw_dir: path to the raw data
    processed_data: path to the resampled data
    slice_len: length of each clip in seconds(12s/60s)
    '''
    for mode in ["eval", "dev", "train"]:
        path_dir = os.path.join(raw_dir, mode)

        results = []
        labels = []
        paddings = []

        for dir, subdir, files in tqdm(os.walk(path_dir), desc=f"Processing {mode}", unit="file", unit_scale=True):
            for file in files:
                if not file.endswith('.csv'):
                    continue
                try:
                    result, label, padding = get_seizure(os.path.join(dir, file), processed_data, slice_len)
                except Exception as e:
                    print(f'Error: {e}')
                    continue
                if not result:
                    continue

                results.extend(result)
                labels.extend(label)
                paddings.extend(padding)

        results = np.array(results)
        labels = np.array(labels)
        paddings = np.array(paddings)

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        with h5py.File(os.path.join(output_dir, f'{mode}_{slice_len}s.h5'), 'w') as hf:
            hf.create_dataset('clips', data=results)
            hf.create_dataset('labels', data=labels)
            hf.create_dataset('paddings', data=paddings)
    return

if __name__ == '__main__':
    preprocess(
        raw_dir='/data/guihaokun/project/tuh_eeg_seizure/v2.0.0/edf',
        processed_data='/data/guihaokun/resample/tuh_eeg_seizure',
        output_dir='/data/guihaokun/classification',
        slice_len=12
    )