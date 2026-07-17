
from datasets.dataset_bsds  import BsdsDataset
from datasets.dataset_biped import BipedDataset
from datasets.dataset_bsds_uncert import BSDS_UncertLoader

import torch.nn as nn
from torch.utils.data import DataLoader

def get_dataset(args, type = 'train'):
    dataset_dir = args.dataset_folder
    use_hsv = args.HSV
    dataset_name = (args.dataset_name).upper()

    DATASETS = {
        'BSDS': lambda: BsdsDataset(dataset_dir=dataset_dir, split = type, hsv = use_hsv),
        'BIPED': lambda: BipedDataset(dataset_dir=dataset_dir, split = type, hsv = use_hsv),
        'UNCERT_BSDS': lambda: BSDS_UncertLoader(root=dataset_dir, split = type)
    }
    
    if dataset_name not in DATASETS:
        raise ValueError(f"Can't recognize {dataset_name}.")
    
    dataset = DATASETS[dataset_name]()


    if type == 'train':
        loader = DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            num_workers=4,
            drop_last=True,
            shuffle=True,
        )
    else:
        loader = DataLoader(
            dataset,
            batch_size=args.test_batch_size,
            num_workers=4,
            drop_last=False,
            shuffle=False,
        )

    return loader