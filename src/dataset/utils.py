import random
from typing import Generic, TypeVar

import numpy as np
from sklearn.model_selection import train_test_split

T = TypeVar("T")


def train_test_val_split(
    X,
    y,
    train_size=None,
    test_size=None,
    random_state=None,
    shuffle=True,
):
    (cp_files, ncp_files, normal_files) = get_class_files(X)

    cp_indices = random.sample(cp_files, 1000)
    ncp_indices = random.sample(ncp_files, 1000)
    normal_indices = random.sample(normal_files, 1000)

    indices = cp_indices + ncp_indices + normal_indices

    X = [X[idx] for idx in indices]
    y = [y[idx] for idx in indices]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle,
    )
    X_test, X_val, y_test, y_val = train_test_split(
        X_test,
        y_test,
        train_size=test_size / (1 - train_size),
        random_state=random_state,
        shuffle=shuffle,
    )

    return X_train, X_test, X_val, y_train, y_test, y_val


def get_balanced_initial_pool(y, initial_pool):
    num_samples = initial_pool // 3
    indices = []
    samples = [0, 0, 0]
    for idx, cls in enumerate(y):
        if samples[cls] < num_samples:
            indices.append(idx)
            samples[cls] += 1
        if np.sum(samples) == num_samples * 3:
            return indices
    return indices


def uncertainty_debug_split(
    dataset, samples_per_cls=[20, 20, 20], initial_pool=3, split=[0.7, 0.2, 0.1]
):
    (
        (cp_train_samples, cp_val_samples, cp_test_samples),
        (ncp_train_samples, ncp_val_samples, ncp_test_samples),
        (normal_train_samples, normal_val_samples, normal_test_samples),
    ) = balanced_train_test_val_split(dataset, split)
    train_samples = []
    # test_samples = []
    # initial_pool_paths = []
    test_samples = list(cp_test_samples.union(ncp_test_samples, normal_test_samples))
    num_class_samples = int(initial_pool / 3)

    cp_labelled_samples = random.sample(cp_train_samples, num_class_samples)
    ncp_labelled_samples = random.sample(ncp_train_samples, num_class_samples)
    normal_labelled_samples = random.sample(normal_train_samples, num_class_samples)

    train_samples = (
        list(random.sample(cp_train_samples, samples_per_cls[0]))
        + list(random.sample(ncp_train_samples, samples_per_cls[1]))
        + list(random.sample(normal_train_samples, samples_per_cls[2]))
    )

    # for cls in classes:
    #     if cls == "CP":
    #         train_samples += list(cp_train_samples)
    #         test_samples += list(cp_test_samples)
    #         initial_pool_paths += list(cp_labelled_samples)
    #     elif cls == "NCP":
    #         train_samples += list(ncp_train_samples)
    #         test_samples += list(ncp_test_samples)
    #         initial_pool_paths += list(ncp_labelled_samples)
    #     elif cls == "Normal":
    #         train_samples += list(normal_train_samples)
    #         test_samples += list(normal_test_samples)
    #         initial_pool_paths += list(normal_labelled_samples)

    full_set = cp_train_samples.union(ncp_train_samples, normal_train_samples)

    # initial_pool = [train_samples.index(idx) for idx in list(initial_pool_paths)]

    remaining_cls = list(full_set - set(train_samples))

    return train_samples, test_samples, remaining_cls


def get_class_files(paths: Generic[T], indices: list[int] = None) -> tuple:
    """
    Get indices per classes.
    Args:
        dataset (Data): The dataset.
        indices (List[int]): indices to split.

    Returns:
        tuple of cls indices
    """
    cp_files = []
    ncp_files = []
    normal_files = []
    for idx, path in enumerate(paths):
        if "/CP" in path:
            cp_files.append(idx)
        elif "/NCP" in path:
            ncp_files.append(idx)
        elif "/Normal" in path:
            normal_files.append(idx)
        else:
            print("Weird path found: {}".format(path))

    return (cp_files, ncp_files, normal_files)


def balanced_train_test_val_split(dataset: Generic[T], split: list[float]):
    (cp_indices, ncp_indices, normal_indices) = get_class_files(dataset)

    shortest_list = min([cp_indices, ncp_indices, normal_indices], key=len)

    len_train = int(len(shortest_list) * split[0])
    len_val = int(len(shortest_list) * split[1])
    len_test = int(len(shortest_list) * split[2])

    cp_indices = set(cp_indices)
    ncp_indices = set(ncp_indices)
    normal_indices = set(normal_indices)

    cp_train_indices = set(random.sample(cp_indices, len_train))
    cp_val_indices = set(random.sample(cp_indices - cp_train_indices, len_val))
    cp_test_indices = set(
        random.sample(cp_indices - cp_train_indices - cp_val_indices, len_test)
    )

    ncp_train_indices = set(random.sample(ncp_indices, len_train))
    ncp_val_indices = set(random.sample(ncp_indices - ncp_train_indices, len_val))
    ncp_test_indices = set(
        random.sample(ncp_indices - ncp_train_indices - ncp_val_indices, len_test)
    )

    normal_train_indices = set(random.sample(normal_indices, len_train))
    normal_val_indices = set(
        random.sample(normal_indices - normal_train_indices, len_val)
    )
    normal_test_indices = set(
        random.sample(
            normal_indices - normal_train_indices - normal_val_indices,
            len_test,
        )
    )

    return (
        (cp_train_indices, cp_val_indices, cp_test_indices),
        (ncp_train_indices, ncp_val_indices, ncp_test_indices),
        (normal_train_indices, normal_val_indices, normal_test_indices),
    )


def balanced_active_learning_split(
    dataset: Generic[T], initial_pool: int = 3, split: list[float] = [0.7, 0.1, 0.2]
):
    """
    Get balanced train, test, val set and an initial pool w.r.t. train.
    Args:
        dataset (Data): The dataset.
        initial_pool (int): Number of instances to label.
        split: (List(float)): train, val, test split
    """
    if sum(split) != 1:
        return "split does not add up to 1: {}".format(split)

    (
        (cp_train_indices, cp_val_indices, cp_test_indices),
        (ncp_train_indices, ncp_val_indices, ncp_test_indices),
        (normal_train_indices, normal_val_indices, normal_test_indices),
    ) = balanced_train_test_val_split(dataset, split)

    label_per_cls = int(initial_pool / 3)

    cp_labelled_indices = random.sample(cp_train_indices, label_per_cls)
    ncp_labelled_indices = random.sample(ncp_train_indices, label_per_cls)
    normal_labelled_indices = random.sample(normal_train_indices, label_per_cls)

    train_indices = list(
        cp_train_indices.union(ncp_train_indices, normal_train_indices)
    )
    val_indices = list(cp_val_indices.union(ncp_val_indices, normal_val_indices))
    test_indices = list(cp_test_indices.union(ncp_test_indices, normal_test_indices))

    initial_pool_indices = (
        cp_labelled_indices + ncp_labelled_indices + normal_labelled_indices
    )
    initial_pool = [train_indices.index(idx) for idx in list(initial_pool_indices)]

    return train_indices, val_indices, test_indices, initial_pool
