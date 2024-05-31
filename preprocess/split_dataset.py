import os
import numpy as np

def get_split_patient(dir_path: str):
    file_name_dict = {
        'sz': set(),
        'nosz': set(),
    }
    for label in ['sz', 'nosz']:
        with open(os.path.join(dir_path, f'trainSet_seq2seq_60s_{label}.txt'), 'r') as f:
            f_str = f.readlines()
            for i in range(len(f_str)):
                file_name = f_str[i].strip('\n').split('.edf')[0]
                file_name_dict[label].add(file_name)
    szf = list(file_name_dict['sz'])
    noszf = list(file_name_dict['nosz'])

    f = szf + noszf
    np.random.shuffle(f)
    length = len(f)
    train_set = f[:int(length * 0.9)]
    dev_set = f[int(length * 0.9):]
    return  train_set, dev_set

def split_ssl_dataset(ssl_dir: str, output_dir: str, dev_set):
    output_dir = os.path.join(output_dir, 'ab_ssl')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    train_dataset = []
    dev_dataset = []
    with open(os.path.join(ssl_dir, 'trainSet_seq2seq_60s.txt'), 'r') as f:
        f_str = f.readlines()
        for i in range(len(f_str)):
            file_name = f_str[i].strip('\n').split('.edf')[0]
            if file_name in dev_set:
                dev_dataset.append(f_str[i])
            else:
                train_dataset.append(f_str[i])
    
    with open(os.path.join(output_dir, 'trainSet_seq2seq_60s.txt'), 'w') as f:
        for i in range(len(train_dataset)):
            f.write(train_dataset[i])
    with open(os.path.join(output_dir, 'devSet_seq2seq_60s.txt'), 'w') as f:
        for i in range(len(dev_dataset)):
            f.write(dev_dataset[i])

def split_detection_dataset(detection_dir: str, output_dir: str, dev_set):
    output_dir = os.path.join(output_dir, 'ab_detection')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_dataset_sz = []
    train_dataset_nosz = []
    dev_dataset_sz = []
    dev_dataset_nosz = []

    with open(os.path.join(detection_dir, 'trainSet_seq2seq_60s_sz.txt'), 'r') as f:
        f_str = f.readlines()
        for i in range(len(f_str)):
            file_name = f_str[i].strip('\n').split('.edf')[0]
            if file_name in dev_set:
                dev_dataset_sz.append(f_str[i])
            else:
                train_dataset_sz.append(f_str[i])

    with open(os.path.join(detection_dir, 'trainSet_seq2seq_60s_nosz.txt'), 'r') as f:
        f_str = f.readlines()
        for i in range(len(f_str)):
            file_name = f_str[i].strip('\n').split('.edf')[0]
            if file_name in dev_set:
                dev_dataset_nosz.append(f_str[i])
            else:
                train_dataset_nosz.append(f_str[i])
    
    with open(os.path.join(output_dir, 'trainSet_seq2seq_60s_sz.txt'), 'w') as f:
        for i in range(len(train_dataset_sz)):
            f.write(train_dataset_sz[i])
    with open(os.path.join(output_dir, 'trainSet_seq2seq_60s_nosz.txt'), 'w') as f:
        for i in range(len(train_dataset_nosz)):
            f.write(train_dataset_nosz[i])
    with open(os.path.join(output_dir, 'devSet_seq2seq_60s_sz.txt'), 'w') as f:
        for i in range(len(dev_dataset_sz)):
            f.write(dev_dataset_sz[i])
    with open(os.path.join(output_dir, 'devSet_seq2seq_60s_nosz.txt'), 'w') as f:
        for i in range(len(dev_dataset_nosz)):
            f.write(dev_dataset_nosz[i])

def process(detection_dir: str, ssl_dir: str, output_dir: str):
    train_set, dev_set = get_split_patient(detection_dir)
    split_detection_dataset(detection_dir, output_dir, dev_set)
    split_ssl_dataset(ssl_dir, output_dir, dev_set)

if __name__ == '__main__':
    detection_dir = "<detection_dir>"
    ssl_dir = "<ssl_dir>"
    output_dir = "<output_dir>"
    process(detection_dir, ssl_dir, output_dir)