import os
import h5py

norm_dict = {
    'abnormal': 1,
    'normal': 0,
}

inverse_dict = {
    1: 'sz',
    0: 'nosz',
}

def preprocess(input_dir:str, output_dir:str, raw_data_dir:str, clip_size:int):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    counter = 0

    for mode in ['eval', 'train']:
        for label in['abnormal', 'normal']:
            dir_path = os.path.join(raw_data_dir, mode, label)
            label = norm_dict[label]
            with open(os.path.join(output_dir, f'{mode}Set_seq2seq_{clip_size}s_{inverse_dict[label]}.txt'), 'w') as f:
                for dir, subdir, files in os.walk(dir_path):
                    for file in files:
                        if '.edf' not in file:
                            continue
                        h5_path = input_dir + '/' + file.split('.edf')[0] + '.h5'
                        if not os.path.exists(h5_path):
                            counter += 1
                            continue
                        tuples = markSeizureSlice(
                            h5_path=h5_path,
                            clip_size=clip_size,
                            label=label,
                        )
                        f.write('\n'.join(tuples))
                        if len(tuples) > 0:
                            f.write('\n')
                        
                        del tuples
    print(f'failed files: {counter}')


def markSeizureSlice(
        h5_path:str,
        clip_size:int,
        label:int,
):
    with h5py.File(h5_path, 'r') as f:
        signal_array = f["resample_signal"][()]
        freq = f["resample_freq"][()]

    # get seizure times
    seq_len = signal_array.shape[1]

    # Iterating through signal
    physical_clip_len = int(freq * clip_size)

    tuples = []
    for clip_idx in range(0, seq_len//physical_clip_len):
        path = h5_path.split('/')[-1].split('.h5')[0]
        tuples.append(f'{path}.edf_{clip_idx}.h5, {label}')
    return tuples

if __name__ == '__main__':
    preprocess(
        input_dir='/data/guihaokun/resample/tuh_eeg_abnormal',
        output_dir='/home/guihaokun/Time-Series-Pretrain/data/ab_detection',
        raw_data_dir='/data/guihaokun/project/tuh_eeg_abnormal/v3.0.0/edf/',
        clip_size=60,
    )