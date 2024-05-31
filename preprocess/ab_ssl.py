import h5py
import os
import argparse
from tqdm import tqdm

def generate_marker(dir_path:str, time_step:int, write_dir:str, h5_dir:str):
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)

    for mode in ['train', 'eval']:
        file_fail = 0
        file_name = mode + f'Set_seq2seq_{time_step}s.txt'
        write_path = os.path.join(write_dir, file_name)
        with open(write_path, 'w') as f:
            sub_dir_path = os.path.join(dir_path, mode)
            for dir, subdir, file in tqdm(os.walk(sub_dir_path), desc=f"Processing {mode}", unit="file", unit_scale=True):
                for name in file:
                    if not name.endswith('.edf'):
                        continue
                    name = name.split('.edf')[0] + '.h5'
                    path = os.path.join(h5_dir, name)
                    try:
                        with h5py.File(path, 'r') as hf:
                            sample_node = hf['resample_signal'][()].shape[1]
                            chunk_size = int(hf['resample_freq'][()] * time_step)
                            name = name.split('.h5')[0] + '.edf'
                            num = sample_node // chunk_size
                            for i in range(0, num - 1):
                                f.write(f'{name}_{i}.h5,{name}_{i+1}.h5\n')
                    except Exception as e:
                        print(f'Error occur: {str(e)}, total{file_fail} files failed.')
                        file_fail += 1
        print(f'{file_fail} files failed in mode {mode}.')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type=str, default="<dir_path>")
    parser.add_argument('--h5_dir', type=str, default="<h5_dir>")
    parser.add_argument('--time_step', type=int, default=60)
    parser.add_argument('--write_dir', type=str, default="<write_dir>")
    args = parser.parse_args()
    generate_marker(args.dir_path, args.time_step, args.write_dir, args.h5_dir)