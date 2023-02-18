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
    # volume size: (n_images, 512, 512, 3)
    volume = torch.tensor(np.load(x, allow_pickle=True))
    # if len(volume.shape) < 4:
    #     volume = volume.unsqueeze(-1)
    # volume size: (n_images, 3, 512, 512)
    # print(volume.shape)
    # volume = volume.permute(1, 0, 2).float()    
    return volume.float()


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
            if len(img.shape)==3:
                # unsqueeze channel of gray scale image
                img = img.unsqueeze(0)
            img_t = self.transform(img, **kwargs)
        else:
            img_t = img

        if self.target_transform:
            seed_all(batch_seed + idx)
            kwargs = self.get_kwargs(
                self.target_transform, image_shape=img.size, idx=idx)
            y = self.target_transform(y, **kwargs)
        y = torch.tensor(y)
        return img_t.squeeze(0), y.type(torch.LongTensor)

    @staticmethod
    def get_kwargs(transform, **kwargs):
        if isinstance(transform, BaaLTransform):
            t_kwargs = {k: kwargs[k] for k in transform.get_requires()}
        else:
            t_kwargs = {}
        return t_kwargs

class CustomDataset(torch.utils.data.Dataset):    
    def __init__(self, data_path="/", target_path="/", is_transform=True):
        # CP 30 files
        # NCP 28 files
        # Normal 21 files
        self.is_transform = is_transform
        self.data_paths = []
        # self.target_paths = []
        self.targets = []
        for data_file_name in os.listdir(data_path):
            self.data_paths.append(data_path + data_file_name)
            # self.target_paths.append(tar + target_file_name)
            self.targets.append(np.load(target_path + "label" + data_file_name[3:]))


    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):

        try:
            # volume size: (n_images, 512, 512, 3)
            volume_path = self.data_paths[index]
            volume = torch.tensor(np.load(volume_path, allow_pickle=True))
            if len(volume.shape) < 4:
                volume = volume.unsqueeze(-1)
            # class size: (1)
            # class_path = self.target_paths[index]
            # class_id = np.load(class_path)
            class_id = self.targets[index]
            # if len(volume.shape) != 4:
            #     raise Exception("wrong volume shape: {}".format(volume.shape))
            # volume size: (n_images, 3, 512, 512)
            volume = volume.permute(0, 3, 1, 2).float()
            # volume = transforms.Resize((512//4, 512//4))(volume)
            # if self.is_transform is True:
            #     # if random.random() <= 0.5:
            #     #     volume = np.flip(np.fliplr(volume))[::-1].copy()
            # #     # volume size: (n_images, 3, 512, 512)  
            #     volume = torch.tensor(volume).permute(0, 3, 1, 2).float()
            #     volume = transforms.Resize((512//2, 512//2))(volume)

            # volume size: (3, n_images, *, *)    
            volume = volume.transpose(0, 1)

            class_id = torch.tensor(class_id)
            class_id = class_id.type(torch.LongTensor)
        except Exception as e:
            print(str(e) + self.data_paths[index])
            return self.__getitem__(index - 1 if index != 0 else index + 1)
        # class_id = torch.tensor(class_id)
        return volume, class_id

    def get_class_files(self, indices=None):
        CP_files = []
        NCP_files = []
        Normal_files = []
        if indices == None:
            paths = self.data_paths
        else:
            paths = [self.data_paths[idx] for idx in indices]
        for path in paths:
            if "_CP_" in path:
                CP_files.append(path)
            elif "_NCP_" in path:
                NCP_files.append(path)
            elif "_Normal_" in path:
                Normal_files.append(path)
            else:
                print("Weird path found: {}".format(path))
        print("CP sample len {}".format(len(CP_files)))
        print("NCP sample len {}".format(len(NCP_files)))
        print("Normal sample len {}".format(len(Normal_files)))
        return CP_files, NCP_files, Normal_files

    def balanced_train_test_split(self, split=[0.8, 0.2]):
        CP_files, NCP_files, Normal_files = self.get_class_files()
        num_normal_files_train = int(len(Normal_files)*split[0])
        num_normal_files_test= int(len(Normal_files)*split[1])

        num_class_samples_train = int(num_normal_files_train)
        num_class_samples_test = int(num_normal_files_test)

        random.shuffle(CP_files)
        random.shuffle(NCP_files)
        random.shuffle(Normal_files)

        random_cp_files = set(CP_files)
        random_ncp_files = set(NCP_files)
        random_normal_files = set(Normal_files)
        
        self.cp_train_samples = set(random.sample(random_cp_files, num_class_samples_train))
        self.cp_test_samples = set(random.sample(random_cp_files - self.cp_train_samples, num_class_samples_test))

        self.ncp_train_samples = set(random.sample(random_ncp_files, num_class_samples_train))
        self.ncp_test_samples = set(random.sample(random_ncp_files - self.ncp_train_samples, num_class_samples_test))

        self.normal_train_samples = set(random.sample(random_normal_files, num_class_samples_train))
        self.normal_test_samples = set(random.sample(random_normal_files - self.normal_train_samples, num_class_samples_test))
        print("Number of samples per class: {}".format(num_class_samples_train))

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


def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    volumes = []
    labels = []
    for (volume_tensor, class_id) in batch:
        if volume_tensor is None:
            print("Found volume with wrong dimensions {}".format(volume_tensor.shape))
            print("Found volume with wrong length {}".format(len(volume_tensor.shape)))
            continue
        volumes.append(volume_tensor.transpose(0, 1))
        labels.append(class_id)
    volumes = nn.utils.rnn.pad_sequence(volumes, batch_first=True).transpose(1, 2)
    return volumes, torch.Tensor(labels).type(torch.LongTensor)

