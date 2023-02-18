from copy import deepcopy
from typing import Callable, Optional

from torch.optim import Optimizer
from baal.active.dataset.base import Dataset
from baal import ModelWrapper

import numpy as np
import torch.utils.data as torchdata

class CustomModelWrapper(ModelWrapper):
    def __init__(self, model, criterion, replicate_in_memory=True):
        super(CustomModelWrapper, self).__init__(model, criterion, replicate_in_memory)

    def train_and_test_on_datasets(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        optimizer: Optimizer,
        batch_size: int,
        epoch: int,
        use_cuda: bool,
        workers: int = 4,
        collate_fn: Optional[Callable] = None,
        regularizer: Optional[Callable] = None,
        return_best_weights=False,
        patience=None,
        max_sample=-1,
        min_epoch_for_es=0,
    ):
        """
        Train and test the model on both Dataset `train_dataset`, `test_dataset`.
        Args:
            train_dataset (Dataset): Dataset to train on.
            test_dataset (Dataset): Dataset to evaluate on.
            optimizer (Optimizer): Optimizer to use during training.
            batch_size (int): Batch size used.
            epoch (int): Number of epoch to train on.
            use_cuda (bool): Use Cuda or not.
            workers (int): Number of workers to use.
            collate_fn (Optional[Callable]): The collate function to use.
            regularizer (Optional[Callable]): The loss regularization for training.
            return_best_weights (bool): If True, will keep the best weights and return them.
            patience (Optional[int]): If provided, will use early stopping to stop after
                                        `patience` epoch without improvement.
            min_epoch_for_es (int): Epoch at which the early stopping starts.
        Returns:
            History and best weights if required.
        """
        best_weight = None
        best_loss = 1e10
        best_epoch = 0
        hist = []
        for e in range(epoch):
            _ = self.train_on_dataset(
                train_dataset, optimizer, batch_size, 1, use_cuda, workers, collate_fn, regularizer
            )
            if max_sample != -1 and max_sample < len(test_dataset):
                indices = np.random.choice(len(test_dataset), max_sample, replace=False)
                test_dataset = torchdata.Subset(test_dataset, indices)
            te_loss = self.test_on_dataset(test_dataset, batch_size, use_cuda, workers, collate_fn)
            hist.append(self.get_metrics())
            if te_loss < best_loss:
                best_epoch = e
                best_loss = te_loss
                if return_best_weights:
                    best_weight = deepcopy(self.state_dict())

            if patience is not None and (e - best_epoch) > patience and (e > min_epoch_for_es):
                # Early stopping
                break

        if return_best_weights:
            return hist, best_weight
        else:
            return hist
