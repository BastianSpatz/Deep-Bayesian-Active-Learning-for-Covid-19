from baal import ModelWrapper

from copy import deepcopy
from typing import Callable, Optional

import structlog
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.optim import Optimizer
from baal.utils.cuda_utils import to_cuda

from src.dataset.dataset import CustomSubset

log = structlog.get_logger("ModelWrapper")


class CustomModelWrapper(ModelWrapper):
    def __init__(self, model, criterion, replicate_in_memory=True):
        super(CustomModelWrapper, self).__init__(model, criterion, replicate_in_memory)

    def train_on_batch(
        self,
        data,
        target,
        optimizer,
        cuda=False,
        regularizer: Optional[Callable] = None,
    ):
        """
        Train the current model on a batch using `optimizer`.

        Args:
            data (Tensor): The model input.
            target (Tensor): The ground truth.
            optimizer (optim.Optimizer): An optimizer.
            cuda (bool): Use CUDA or not.
            regularizer (Optional[Callable]): The loss regularization for training.


        Returns:
            Tensor, the loss computed from the criterion.
        """

        if cuda:
            data, target = to_cuda(data), to_cuda(target)
        optimizer.zero_grad()
        output = self.model(data)
        loss = self.criterion(output, target)

        if regularizer:
            regularized_loss = loss + regularizer()
            regularized_loss.backward()
        else:
            loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=0.5)
        optimizer.step()
        self._update_metrics(output, target, loss, filter="train")
        return loss

    def train_on_dataset(
        self,
        dataset,
        optimizer,
        batch_size,
        epoch,
        use_cuda,
        workers=4,
        collate_fn: Optional[Callable] = None,
        regularizer: Optional[Callable] = None,
    ):
        """
        Train for `epoch` epochs on a Dataset `dataset.

        Args:
            dataset (Dataset): Pytorch Dataset to be trained on.
            optimizer (optim.Optimizer): Optimizer to use.
            batch_size (int): The batch size used in the DataLoader.
            epoch (int): Number of epoch to train for.
            use_cuda (bool): Use cuda or not.
            workers (int): Number of workers for the multiprocessing.
            collate_fn (Optional[Callable]): The collate function to use.
            regularizer (Optional[Callable]): The loss regularization for training.

        Returns:
            The training history.
        """
        dataset_size = len(dataset)
        self.train()
        self.set_dataset_size(dataset_size)
        history = []
        log.info("Starting training", epoch=epoch, dataset=dataset_size)
        collate_fn = collate_fn or default_collate
        for _ in range(epoch):
            self._reset_metrics("train")
            for data, target, *_ in DataLoader(
                dataset, batch_size, True, num_workers=workers, collate_fn=collate_fn
            ):
                _ = self.train_on_batch(data, target, optimizer, use_cuda, regularizer)
            history.append(self.get_metrics("train")["train_loss"])

        optimizer.zero_grad()  # Assert that the gradient is flushed.
        log.info(
            "Training complete", train_loss=self.get_metrics("train")["train_loss"]
        )
        self.active_step(dataset_size, self.get_metrics("train"))
        return history

    def train_and_test_on_datasets(
        self,
        train_dataset: CustomSubset,
        test_dataset: CustomSubset,
        optimizer: Optimizer,
        scheduler,
        batch_size: int,
        epoch: int,
        use_cuda: bool,
        workers: int = 4,
        collate_fn: Optional[Callable] = None,
        regularizer: Optional[Callable] = None,
        learning_epoch=1,
        return_best_weights=False,
        patience=None,
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
                train_dataset,
                optimizer,
                batch_size,
                learning_epoch,
                use_cuda,
                workers,
                collate_fn,
                regularizer,
            )
            te_loss = self.test_on_dataset(
                test_dataset, batch_size, use_cuda, workers, collate_fn
            )
            hist.append(self.get_metrics())
            if te_loss < best_loss:
                best_epoch = e
                best_loss = te_loss
                if return_best_weights:
                    best_weight = deepcopy(self.state_dict())

            scheduler.step(te_loss)

            if (
                patience is not None
                and (e - best_epoch) > patience
                and (e > min_epoch_for_es)
            ):
                # Early stopping
                break
        if return_best_weights:
            return hist, best_weight
        else:
            return hist
