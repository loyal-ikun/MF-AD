import numpy as np
import torch
from torchvision import transforms
from utils.load_dataset import DatasetSplit

_CLASSNAMES = [
    "traffic"
]


class Traffic_Train_DataDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for traffic.
    """

    def __init__(
            self,
            source,
            classname,
            resize=64,
            imagesize=64,
            split=DatasetSplit.TRAIN,
            cfg=None,
            **kwargs
    ):
        super().__init__()
        self.source = source
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else _CLASSNAMES
        self.cfg = cfg
        self.array = np.load(cfg.TRAIN.dataset_path) 

        # todo: demo

        self.array = self.array.reshape(self.array.shape[0] * 8, 3, 64,
                                                             64)

        self.classname, self.is_anomaly = self.get_image_data()

        # for test
        self.transform_img = [
            transforms.Resize((resize, resize)),
            transforms.ToTensor()
        ]

        self.transform_img = transforms.Compose(self.transform_img)

        # for train
        self.transform_img_MMR = transforms.Compose([
            transforms.ToTensor(),
        ])

        transform_mask = [
            transforms.Resize((resize, resize)),
            transforms.ToTensor(),
        ]
        self.transform_mask = transforms.Compose(transform_mask)

        self.imagesize = (3, imagesize, imagesize)

    def __len__(self):
        return len(self.is_anomaly)

    def get_image_data(self):
        num = self.array.shape[0]
        classname = ["good" for _ in range(num)]
        is_anomaly = [0 for _ in range(num)]
        return classname, is_anomaly

    def __getitem__(self, idx):
        image = self.array[idx]
        image = torch.from_numpy(image)
        classname = self.classname[idx]
        is_anomaly = self.is_anomaly[idx]

        return {
            "image": image,
            "classname": classname,
            "is_anomaly": is_anomaly
        }
