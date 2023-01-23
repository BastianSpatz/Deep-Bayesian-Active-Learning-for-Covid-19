import torch
import torchvision.transforms.functional as TF

def train_test_validation_split(dataset, split=[.8, .1, .1]):
    dataset_len = len(dataset)
    train_size = int(0.8*dataset_len)
    test_size = int(0.1*dataset_len)
    valid_size = dataset_len - train_size - test_size
    train, test, validation = torch.utils.data.random_split(dataset, [train_size, test_size, valid_size])

    return train, test, validation

class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return TF.rotate(x, self.angle)