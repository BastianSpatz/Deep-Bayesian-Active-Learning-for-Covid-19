import random
import torch
import numpy as np
import torchvision.transforms.functional as TF

class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, x):
        return TF.rotate(x, self.angle)


def get_active_learning_datasets(dataset, initial_pool=3, split=[0.7, 0.1, 0.2]):
    if sum(split) != 1:
        return "split does not add up to 1: {}".format(split)
    
    (cp_train_samples, cp_val_samples, cp_test_samples), (ncp_train_samples, ncp_val_samples, ncp_test_samples), (normal_train_samples, normal_val_samples, normal_test_samples) = balanced_train_test_split(dataset, split)
    num_class_samples = int(initial_pool/3)

    cp_labelled_samples = random.sample(cp_train_samples, num_class_samples)
    ncp_labelled_samples = random.sample(ncp_train_samples, num_class_samples)
    normal_labelled_samples = random.sample(normal_train_samples, num_class_samples)
    

    train_paths = cp_train_samples.union(ncp_train_samples, normal_train_samples)
    val_paths = cp_val_samples.union(ncp_val_samples, normal_val_samples)
    test_paths = cp_test_samples.union(ncp_test_samples, normal_test_samples)

    train_indices = [dataset.data_paths.index(path) for path in list(train_paths)]
    val_indices = [dataset.data_paths.index(path) for path in list(val_paths)]
    test_indices = [dataset.data_paths.index(path) for path in list(test_paths)]



    initial_pool_paths = cp_labelled_samples + ncp_labelled_samples + normal_labelled_samples
    initial_pool = [list(train_paths).index(path) for path in list(initial_pool_paths)]
    # initial_pool = [dataset.data_paths.index(path) for path in list(initial_pool_paths)]

    print("size train files: {}".format(len(train_indices)))
    print("size test files: {}".format(len(test_indices)))
    print("size val files: {}".format(len(val_indices)))
    print("size initial pool files: {}".format(len(initial_pool)))
    return train_indices, val_indices, test_indices, initial_pool
    
def balanced_train_test_split(dataset, split):
    CP_files, NCP_files, Normal_files = get_class_files(dataset)
    num_normal_files_train = int(len(Normal_files)*split[0])
    num_normal_files_val= int(len(Normal_files)*split[1])
    num_normal_files_test= int(len(Normal_files)*split[2])

    # random.shuffle(CP_files)
    # random.shuffle(NCP_files)
    # random.shuffle(Normal_files)

    cp_files = set(CP_files)
    ncp_files = set(NCP_files)
    normal_files = set(Normal_files)
    
    cp_train_samples = set(random.sample(cp_files, num_normal_files_train))
    cp_val_samples = set(random.sample(cp_files - cp_train_samples, num_normal_files_val))
    cp_test_samples = set(random.sample(cp_files - cp_train_samples - cp_val_samples, num_normal_files_test))

    ncp_train_samples = set(random.sample(ncp_files, num_normal_files_train))
    ncp_val_samples = set(random.sample(ncp_files - ncp_train_samples, num_normal_files_val))
    ncp_test_samples = set(random.sample(ncp_files - ncp_train_samples - ncp_val_samples, num_normal_files_test))

    normal_train_samples = set(random.sample(normal_files, num_normal_files_train))
    normal_val_samples = set(random.sample(normal_files - normal_train_samples, num_normal_files_val))
    normal_test_samples = set(random.sample(normal_files - normal_train_samples - normal_val_samples, num_normal_files_test))
    
    return (cp_train_samples, cp_val_samples, cp_test_samples), (ncp_train_samples, ncp_val_samples, ncp_test_samples), (normal_train_samples, normal_val_samples, normal_test_samples)

def get_class_files(dataset, indices=None):
    CP_files = []
    NCP_files = []
    Normal_files = []
    if indices == None:
        paths = dataset.data_paths
    else:
        paths = [dataset.data_paths[idx] for idx in indices]
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


def train_test_validation_split(dataset, split=[.8, .1, .1]):
    dataset_len = len(dataset)
    train_size = int(0.8*dataset_len)
    test_size = int(0.1*dataset_len)
    valid_size = dataset_len - train_size - test_size
    train, test, validation = torch.utils.data.random_split(dataset, [train_size, test_size, valid_size])

    return train, test, validation
