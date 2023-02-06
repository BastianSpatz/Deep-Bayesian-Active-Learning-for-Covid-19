import os
import random
import warnings
from typing import Any, Callable, List, Optional, Sequence

import numpy as np
import torch
from baal.utils.transforms import BaaLTransform
from torch.utils.data import Dataset


def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

class CustomActiveLearningDataset(Dataset):
    def __init__(self, data_path, target_path, seed):

        self.data_paths = []
        self.targets = []
        self.CP_files = []
        self.NCP_files = []
        self.Normal_files = []
        self.seed = seed
        for idx, data_file_name in enumerate(os.listdir(data_path)):
            self.data_paths.append(data_path + data_file_name)
            # self.target_paths.append(path_to_npy_targets + target_file_name)
            self.targets.append(
                np.load(target_path + "label" + data_file_name[3:]))
            if "_CP_" in data_file_name:
                self.CP_files.append(idx)
            elif "_NCP_" in data_file_name:
                self.NCP_files.append(idx)
            elif "_Normal_" in data_file_name:
                self.Normal_files.append(idx)
            else:
                print("Weird path found: {}".format(data_file_name))

    def load_data(vol_path):
        volume = np.load(vol_path, allow_pickle=True)
        # volume = torch.tensor(volume).permute(0, 3, 1, 2).float()
        # # volume size: (3, n_images, *, *)
        # volume = volume.transpose(0, 1)
        return volume

    def balanced_train_test_split(self, split=[0.8, 0.2]):
        seed_all(self.seed)

        num_normal_files_train = int(len(self.Normal_files)*split[0])
        num_normal_files_test = int(len(self.Normal_files)*split[1])

        random.shuffle(self.CP_files)
        random.shuffle(self.NCP_files)
        random.shuffle(self.Normal_files)

        random_cp_files = set(self.CP_files)
        random_ncp_files = set(self.NCP_files)
        random_normal_files = set(self.Normal_files)

        self.cp_train_samples = set(random.sample(
            random_cp_files, num_normal_files_train))
        self.cp_test_samples = set(random.sample(
            random_cp_files - self.cp_train_samples, num_normal_files_test))

        self.ncp_train_samples = set(random.sample(
            random_ncp_files, num_normal_files_train))
        self.ncp_test_samples = set(random.sample(
            random_ncp_files - self.ncp_train_samples, num_normal_files_test))

        self.normal_train_samples = set(random.sample(
            random_normal_files, num_normal_files_train))
        self.normal_test_samples = set(random.sample(
            random_normal_files - self.normal_train_samples, num_normal_files_test))
        print("Number of samples per class: {}".format(num_normal_files_train))
    
    def get_active_learning_datasets(self, initial_pool=3, split=[0.8, 0.2]):
        self.balanced_train_test_split(split)
        num_class_samples = int(initial_pool/3)

        cp_labelled_samples = random.sample(self.cp_train_samples, num_class_samples)
        ncp_labelled_samples = random.sample(self.ncp_train_samples, num_class_samples)
        normal_labelled_samples = random.sample(self.normal_train_samples, num_class_samples)
        

        train_paths = self.cp_train_samples.union(self.ncp_train_samples, self.normal_train_samples)
        test_paths = self.cp_test_samples.union(self.ncp_test_samples, self.normal_test_samples)
        
        train_indices = [self.data_paths.index(path) for path in list(train_paths)]
        test_indices = [self.data_paths.index(path) for path in list(test_paths)]

        initial_pool_paths = cp_labelled_samples + ncp_labelled_samples + normal_labelled_samples
        initial_pool = [list(train_paths).index(path) for path in list(initial_pool_paths)]
        # initial_pool = [self.data_paths.index(path) for path in list(initial_pool_paths)]

        print("size train files: {}".format(len(train_indices)))
        print("size test files: {}".format(len(test_indices)))
        print("size initial pool files: {}".format(len(initial_pool)))
        
        return train_indices, test_indices, initial_pool

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        x, y = self.data_paths[idx], self.targets[idx]

        np.random.seed(self.seed)
        batch_seed = np.random.randint(0, 100, 1).item()
        seed_all(batch_seed + idx)

        vol = self.load_data(x)

        if self.transform:
            vol_t = self.transform(vol)
        else:
            vol_t = vol

        return vol_t, y

def default_image_load_fn(x):
    volume = np.load(x, allow_pickle=True)
    # volume size: (n_images, 3, 512, 512)
    volume = torch.tensor(volume).permute(0, 3, 1, 2).float()    
    return volume


class CustomFileDataset(Dataset):
    """
    Dataset object that load the files and apply a transformation.
    Args:
        files (List[str]): The files.
        lbls (List[Any]): The labels, -1 indicates that the label is unknown.
        transform (Optional[Callable]): torchvision.transform pipeline.
        target_transform (Optional[Callable]): Function that modifies the target.
        image_load_fn (Optional[Callable]): Function that loads the image, by default uses PIL.
        seed (Optional[int]): Will set a seed before and between DA.
    """

    def __init__(
        self,
        files: List[str],
        lbls: Optional[List[Any]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        image_load_fn: Optional[Callable] = None,
        seed=None,
    ):
        self.files = files

        if lbls is None:
            self.lbls = [-1] * len(self.files)
        else:
            self.lbls = lbls

        self.transform = transform
        self.target_transform = target_transform
        self.image_load_fn = image_load_fn or default_image_load_fn
        self.seed = seed

    def label(self, idx: int, lbl: Any):
        """
        Label the sample `idx` with `lbl`.
        Args:
            idx (int): The sample index.
            lbl (Any): The label to assign.
        """
        if self.lbls[idx] >= 0:
            warnings.warn(
                "We're modifying the class of the sample {} that we already know : {}.".format(
                    self.files[idx], self.lbls[idx]
                ),
                UserWarning,
            )

        self.lbls[idx] = lbl

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
        kwargs = self.get_kwargs(self.transform, image_shape=img.size, idx=idx)

        if self.transform:
            img_t = self.transform(img, **kwargs)
        else:
            img_t = img

        if self.target_transform:
            seed_all(batch_seed + idx)
            kwargs = self.get_kwargs(
                self.target_transform, image_shape=img.size, idx=idx)
            y = self.target_transform(y, **kwargs)
        y = torch.tensor(y)
        return img_t.transpose(0, 1), y.type(torch.LongTensor)

    @staticmethod
    def get_kwargs(transform, **kwargs):
        if isinstance(transform, BaaLTransform):
            t_kwargs = {k: kwargs[k] for k in transform.get_requires()}
        else:
            t_kwargs = {}
        return t_kwargs
