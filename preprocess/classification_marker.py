import h5py
import os
import numpy as np
import pandas as pd

map_dict = {
    'fnsz': 0, # FNSZ -> (Combined Focal)CF
    'spsz': 0, # SPSZ -> (Combined Focal)CF
    'cpsz': 0, # CPSZ -> (Combined Focal)CF

    'gnsz': 1, # GNSZ -> (Generalized Non-specific)GN

    'absz': 2, # ABSZ -> (Absence Seizure)AS

    'tnsz': 3, # TNSZ -> (Combined Tonic)CT
    'tcsz': 3, # TCSZ -> (Combined Tonic)CT
}

def get_seizure(file_path: str) -> pd.DataFrame:
    '''
    Args:
    file_path: path to the csv file
    '''
    df = pd.read_csv(file_path, skiprows=5)

    # Sort the DataFrame by label and start_time
    df = df.sort_values(by=['label', 'start_time'])

    # Function to merge overlapping intervals for each group
    def merge_overlapping(group):
        result = []
        current_start, current_end = group.iloc[0]['start_time'], group.iloc[0]['stop_time']

        for i in range(1, len(group)):
            if group.iloc[i]['start_time'] <= current_end + 1e-1:
                current_end = max(current_end, group.iloc[i]['stop_time'])
            else:
                result.append({'label': group.iloc[i - 1]['label'], 'start_time': current_start, 'stop_time': current_end})
                current_start, current_end = group.iloc[i]['start_time'], group.iloc[i]['stop_time']

        # Add the last interval
        result.append({'label': group.iloc[-1]['label'], 'start_time': current_start, 'stop_time': current_end})

        return pd.DataFrame(result)
    
    # Apply the function to each group and concatenate the results
    result = df.groupby('label').apply(merge_overlapping).reset_index(drop=True)

    # Filter the data not in the dict_map
    result = result[result['label'].isin(map_dict.keys())]

    # Reset index
    result['label'] = result['label'].map(map_dict)
    return result

def get_result_slice(file_name: str, raw_dir: str, processed_dir: str, clip_len: int):
    '''
    file_name: name of the csv file
    raw_dir: path to the raw data
    processed_dir: path to the resampled data
    clip_length: length of each clip in seconds(12s/60s)
    '''
    # Read the csv file
    file_path = os.path.join(raw_dir, file_name)
    df = get_seizure(file_path)
    preprocessed_file_path = os.path.join(processed_dir, file_name.split('.csv')[0] + '.h5')
    results = []
    paddings = []
    labels = []
    try:
        with h5py.File(preprocessed_file_path, 'r') as hf:
            sample_signal = hf['resample_signal'][()]
            freq = hf['resample_freq'][()]
    except:
        raise Exception('File not found: ' + preprocessed_file_path)
    chunk_size = int(freq * clip_len)
    for i in range(len(df)):
        start_time, stop_time, label = df.iloc[i]['start_time'], df.iloc[i]['stop_time'], df.iloc[i]['label']
        start_time = max(int((start_time - 2) * freq), 0)
        stop_time = min(int(stop_time * freq), sample_signal.shape[1])
        for j in range(start_time, stop_time, chunk_size):
            result = np.zeros((sample_signal.shape[0], chunk_size))
            padding = np.zeros((sample_signal.shape[0], chunk_size))
            if j + chunk_size <= stop_time:
                result[:, :chunk_size] = sample_signal[:, j:j + chunk_size]
            else:
                result[:, :stop_time - j] = sample_signal[:, j:stop_time]
                padding[:, stop_time - j:] = 1
            results.append(result)
            paddings.append(padding)
            labels.append(label)
    results = np.array(results)
    paddings = np.array(paddings)
    labels = np.array(labels)

    return results, paddings, labels

def preprocess(raw_dir: str, processed_data: str, output_dir: str, slice_len: int):
    '''
    raw_dir: path to the raw data
    processed_data: path to the resampled data
    slice_len: length of each clip in seconds(12s/60s)
    '''
    for mode in ["eval"]:
        path_dir = os.path.join(raw_dir, mode)

        results = np.empty((0, 19, slice_len * 250))
        paddings = np.empty((0, 19, slice_len * 250))
        labels = np.empty(0)

        for dir, subdir, files in os.walk(path_dir):
            for file in files:
                if not file.endswith('.csv'):
                    continue
                try:
                    result, padding, label = get_result_slice(file, dir, processed_data, slice_len)
                except:
                    print('File not found: ' + file)
                    continue
                if result.size == 0:
                    continue

                results = np.concatenate((results, result), axis=0)
                paddings = np.concatenate((paddings, padding), axis=0)
                labels = np.concatenate((labels, label), axis=0)
        with h5py.File(os.path.join(output_dir, f'{mode}_{slice_len}s.h5'), 'w') as hf:
            hf.create_dataset('results', data=results)
            hf.create_dataset('paddings', data=paddings)
            hf.create_dataset('labels', data=labels)
    return

if __name__ == '__main__':
    preprocess(
        raw_dir="<raw_dir>",
        processed_data="<processed_data>",
        output_dir="<output_dir>",
        slice_len=60
    )