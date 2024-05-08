from data_provider.Dataset_TUSZ import Dataset_TUSZ
from data_provider.Dataset_TUAB import Dataset_TUAB
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

import argparse

data_dict = {
    'TUSZ': Dataset_TUSZ,
    'TUAB': Dataset_TUAB,
}

def data_provider(args: argparse.Namespace, scalar=None):
    if args.split == 'train':
        shuffle = True
        batch_size = args.train_batch_size
    else:
        shuffle = False
        batch_size = args.test_batch_size
        args.data_augment = False

    dataset = data_dict[args.dataset](args, scalar=scalar)
    return dataset, \
        DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            sampler=DistributedSampler(dataset, shuffle=shuffle) \
                if args.use_gpu else \
                RandomSampler(dataset, shuffle=shuffle),
            persistent_workers=True,
        )
    
