import os
import random
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np

from src.dataset.utils import MyRotationTransform

class Dataset(torch.utils.data.Dataset):    
    def __init__(self, path_to_npy_data="/", path_to_npy_targets="/", is_transform=True):
        # CP 30 files
        # NCP 28 files
        # Normal 21 files
        self.is_transform = is_transform
        self.data_paths = []
        self.target_paths = []
        for data_file_name in os.listdir(path_to_npy_data):
            self.data_paths.append(path_to_npy_data + data_file_name)
        for target_file_name in os.listdir(path_to_npy_targets):  
            self.target_paths.append(path_to_npy_targets + target_file_name)
        print(len(self.data_paths))


    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):

        try:
            # volume size: (n_images, 512, 512, 3)
            volume_path = self.data_paths[index]
            volume = np.load(volume_path, allow_pickle=True)
            # class size: (1)
            class_path = self.target_paths[index]
            class_id = np.load(class_path)
            # if len(volume.shape) != 4:
            #     raise Exception("wrong volume shape: {}".format(volume.shape))
            # volume size: (n_images, 3, 512, 512)
            volume = torch.tensor(volume).permute(0, 3, 1, 2).float()
            
            if self.is_transform is True:
                volume = transforms.Resize((512//4, 512//4))(volume)

            # volume size: (3, n_images, *, *)    
            volume = volume.transpose(0, 1)

            class_id = torch.tensor(class_id)
            class_id = class_id.type(torch.LongTensor)
        except Exception as e:
            print(str(e) + self.data_paths[index])
            return self.__getitem__(index - 1 if index != 0 else index + 1)
        # class_id = torch.tensor(class_id)
        return volume, class_id

    def transform(self, angles=[60, 30, 15, -15, -30, -60]):
        random_angle = random.choice(angles)
        RotationTransform = MyRotationTransform(random_angle)
        transform = transforms.Compose(
            [transforms.Resize((512//4, 512//4))])
        return transform



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

