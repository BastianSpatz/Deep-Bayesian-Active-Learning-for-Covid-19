import os
import random
import warnings
from typing import Any, Callable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from baal.utils.transforms import BaaLTransform
from torch.utils.data import Dataset


def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def default_image_load_fn(x):
    """
    Load volume tensor.
    Args:
        x (str): Input path.

    Returns:
        volume tensor (n_images, 512, 512).
    """
    volume = torch.tensor(np.load(x, allow_pickle=True))
    return volume.float()


class Data(Dataset):
    """
    Dataset object that loads the file paths.
    Args:
        data_path (str): The file path.
        target_path (str): The target path.
        transform (Optional[Callable]): torchvision.transform pipeline.
        image_load_fn (Optional[Callable]): Function that loads the image, by default uses numpy.load.
        seed (Optional[int]): Will set a seed before and between DA.
    """

    def __init__(
        self,
        data_path: str,
        target_path: str,
        image_load_fn: Optional[Callable] = None,
        seed=None,
    ):

        self.image_load_fn = image_load_fn or default_image_load_fn
        self.seed = seed
        self.files = []
        self.lbls = []
        for data_file_name in os.listdir(data_path):
            self.files.append(data_path + data_file_name)

            self.lbls.append(np.load(target_path + "label" + data_file_name[3:]))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x, y = self.files[idx], self.lbls[idx]

        np.random.seed(self.seed)
        batch_seed = np.random.randint(0, 100, 1).item()
        seed_all(batch_seed + idx)

        try:
            img = self.image_load_fn(x)
        except Exception as e:
            print(e)
            return self.__getitem__(idx - 1 if idx != 0 else idx + 1)

        y = torch.tensor(y)
        return img, y.type(torch.LongTensor)


class CustomSubset:
    """
    Subset object that stores indices and applies transformations.
    Args:
        dataset (Dataset): Dataset which we create a subset from.
        indices (List[int]): The labels of the subset.
        transform (Optional[Callable]): torchvision.transform pipeline.
    """

    def __init__(self, dataset, indices, transform=None) -> None:
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        if self.transform is not None:
            # unsqueeze channel in grayscale
            x = self.transform(x.unsqueeze(1)).squeeze(1)
            return x, y
        return x, y

    def __len__(self):
        return len(self.indices)
