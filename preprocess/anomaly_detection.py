import os
import h5py

def preprocess(input_dir:str, output_dir:str, raw_data_dir:str, clip_size:int):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    counter = 0

    for mode in ['eval', 'dev', 'train']:
        dir_path = os.path.join(raw_data_dir, mode)
        szf = open(os.path.join(output_dir, f'{mode}Set_seq2seq_{clip_size}s_sz.txt'), 'w')
        noszf = open(os.path.join(output_dir, f'{mode}Set_seq2seq_{clip_size}s_nosz.txt'), 'w')
        for dir, subdir, files in os.walk(dir_path):
            for file in files:
                if '.edf' not in file:
                    continue
                edf_file = os.path.join(dir, file)
                h5_path = input_dir + '/' + edf_file.split('/')[-1].split('.edf')[0] + '.h5'
                if not os.path.exists(h5_path):
                    counter += 1
                    continue
                sz_tuple, nosz_tuple = markSeizureSlice(
                    h5_dir=input_dir,
                    edf_path=edf_file,
                    clip_size=clip_size,
                )
                szf.write('\n'.join(sz_tuple))
                if len(sz_tuple) > 0:
                    szf.write('\n')
                noszf.write('\n'.join(nosz_tuple))
                if len(nosz_tuple) > 0:
                    noszf.write('\n')
        szf.close()
        noszf.close()
    print(f'failed files: {counter}')

def markSeizureSlice(
        h5_dir:str,
        edf_path:str,
        clip_size:int,
):
    file_path = os.path.join(h5_dir, edf_path.split('/')[-1].split('.edf')[0] + '.h5')
    with h5py.File(file_path, 'r') as f:
        signal_array = f["resampled_signal"][()]
        freq = f["resample_freq"][()]

    # get seizure times
    seizure_times = getSeizureTimes(edf_path)
    seq_len = signal_array.shape[1]

    # Iterating through signal
    physical_clip_len = int(freq * clip_size)

    sz_tuple = []
    nosz_tuple = []
    for clip_idx in range(0, seq_len // physical_clip_len):
        start_window = clip_idx * physical_clip_len
        end_window = start_window + physical_clip_len

        is_seizure = 0
        for t in seizure_times:
            start_t = int(t[0] * freq)
            end_t = int(t[1] * freq)
            if not ((end_window < start_t) or (start_window > end_t)):
                is_seizure = 1
                break
        path = edf_path.split('/')[-1]
        if is_seizure:
            sz_tuple.append(f'{path}_{clip_idx}.h5, {is_seizure}')
        else:
            nosz_tuple.append(f'{path}_{clip_idx}.h5, {is_seizure}')
    return sz_tuple, nosz_tuple

def getSeizureTimes(file_name):
    """
    Args:
        file_name: edf file name
    Returns:
        seizure_times: list of times of seizure onset in seconds
    """
    tse_file = file_name.split(".edf")[0] + ".csv_bi"

    seizure_times = []
    with open(tse_file) as f:
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

if __name__ == '__main__':
    preprocess(
        input_dir="<input_dir>",
        output_dir="<output_dir>",
        raw_data_dir="<raw_data_dir>",
        clip_size=12,
    )