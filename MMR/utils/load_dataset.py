import logging
import torch
from enum import Enum
import numpy as np

LOGGER = logging.getLogger(__name__)

_DATASETS = {
    "traffic_train": ["datasets.traffic_train", "Traffic_Train_DataDataset"],
    "traffic_test": ["datasets.traffic_test", "Traffic_Test_DataDataset"],
}


class DatasetSplit(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def get_train_dataloaders(cfg, mode='train'):
    dataset_info = _DATASETS["traffic_train"]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    dataloaders = []

    # cfg.DATASET.subdatasets is a list which includes diverse objects
    for subdataset in cfg.DATASET.subdatasets:
        dataset = dataset_library.__dict__[dataset_info[1]](
            source=cfg.TRAIN.dataset_path,
            classname=subdataset,
            resize=cfg.DATASET.resize,
            imagesize=cfg.DATASET.imagesize,
            split=DatasetSplit.TRAIN,
            cfg=cfg,
            seed=cfg.RNG_SEED
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.TRAIN_SETUPS.batch_size,
            shuffle=True,
            num_workers=cfg.TRAIN_SETUPS.num_workers,
            pin_memory=True,
        )

        dataloader.name = cfg.DATASET.name
        if subdataset is not None:
            dataloader.name += "_" + subdataset

        dataloaders.append(dataloader)
    return dataloaders


def get_test_dataloaders(cfg, mode='test'):
    dataset_info = _DATASETS["traffic_test"]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    dataloaders = []

    # cfg.DATASET.subdatasets is a list which includes diverse objects
    for subdataset in cfg.DATASET.subdatasets:
        dataset = dataset_library.__dict__[dataset_info[1]](
            source=cfg.TEST.dataset_path,
            classname=subdataset,
            resize=cfg.DATASET.resize,
            imagesize=cfg.DATASET.imagesize,
            split=DatasetSplit.TEST,
            cfg=cfg,
            seed=cfg.RNG_SEED
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=cfg.TEST_SETUPS.batch_size,
            shuffle=False,
            num_workers=cfg.TRAIN_SETUPS.num_workers,
            pin_memory=True,
        )

        dataloader.name = cfg.DATASET.name
        if subdataset is not None:
            dataloader.name += "_" + subdataset

        dataloaders.append(dataloader)
    return dataloaders
