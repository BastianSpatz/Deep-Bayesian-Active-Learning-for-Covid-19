import argparse
import os
from pprint import pprint
import random
import csv
import sys
from dataclasses import dataclass

from torchvision import transforms
from torchvision.transforms.functional import rotate


import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning.loggers import CometLogger
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from src.configs.comet_configs import comet_config
from src.dataset.dataset import CustomDataset, collate_fn_padd, CustomFileDataset
from src.dataset.utils import get_active_learning_datasets, MyRotationTransform
from src.models.conv_net import CNN, CustomVGG16, MedicalNet, MnistExampleModel


@dataclass
class TrainConfig:
    epoch: int = 10
    batch_size: int = 8
    lr: float = 1e-4


class ConvModule(LightningModule):
    def __init__(self, model, args):
        super(ConvModule, self).__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.args = args
        self.lr = self.args.learning_rate
        self.train_accuracy = Accuracy(task="multiclass")
        self.val_accuracy = Accuracy(task="multiclass")
        self.train_transform = transforms.Compose(
            [  # transforms.Resize((512//2, 512//2)),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomAffine(
                    degrees=(30, 70), translate=(0.1, 0.1), scale=(0.5, 0.75)
                ),
                # transforms.ColorJitter(brightness=.5, hue=.3),
            ]
        )

        self.test_transform = transforms.Compose(
            [  # transforms.Resize((512//2, 512//2)),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        self.save_hyperparameters()
        self.dataset = CustomDataset(
            data_path=args.path_to_npy_data, target_path=args.path_to_npy_targets
        )
        (
            self.train_indices,
            self.val_indices,
            self.test_indices,
            _,
        ) = get_active_learning_datasets(self.dataset, split=[0.1, 0.8, 0.1])

    def training_step(self, batch, batch_idx):
        # loss = self.step(batch)
        volumes, labels = batch
        output = self.model(volumes)
        loss = self.criterion(output, labels)

        pred = torch.argmax(output, dim=1)
        self.train_accuracy.update(pred, labels)
        self.log(
            "train_loss", loss.detach(), on_epoch=True, batch_size=self.args.batch_size
        )

        return loss

    def training_epoch_end(self, outs):
        self.log("train_acc", self.train_accuracy.compute())

    def validation_step(self, batch, batch_idx):
        # loss = self.step(batch)
        volumes, labels = batch
        output = self.model(volumes)
        loss = self.criterion(output, labels)

        pred = torch.argmax(output, dim=1)
        self.val_accuracy.update(pred, labels)

        self.log(
            "lr",
            self.scheduler.get_last_lr()[0],
            on_epoch=True,
            on_step=False,
            logger=True,
            batch_size=self.args.batch_size,
        )
        self.log(
            "val_loss", loss.detach(), on_epoch=True, batch_size=self.args.batch_size
        )

    def validation_epoch_end(self, outs):
        self.log("val_acc", self.val_accuracy.compute())

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=5, gamma=0.25
        )
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}
        # return self.optimizer

    def train_dataloader(self):
        # We use -1 to specify that the data is unlabeled.
        train_datset = CustomFileDataset(
            [self.dataset.data_paths[idx] for idx in list(self.train_indices)],
            [self.dataset.targets[idx] for idx in list(self.train_indices)],
            self.train_transform,
        )
        # train_datset = torch.utils.data.Subset(self.dataset, list(self.train_indices))
        return DataLoader(
            dataset=train_datset,
            num_workers=self.args.num_workers,
            batch_size=self.args.batch_size,
            pin_memory=True,
        )

    def val_dataloader(self):
        test_datset = CustomFileDataset(
            [self.dataset.data_paths[idx] for idx in list(self.test_indices)],
            [self.dataset.targets[idx] for idx in list(self.test_indices)],
            self.test_transform,
        )
        # test_datset = torch.utils.data.Subset(self.dataset, list(self.test_indices))
        return DataLoader(
            dataset=test_datset,
            num_workers=self.args.num_workers,
            batch_size=self.args.batch_size,
            pin_memory=True,
        )


def train(args):

    random.seed(28)
    torch.manual_seed(28)

    # model = MnistExampleModel()
    model = CustomVGG16()

    # for param_name, param in model.named_parameters():
    #     if param_name.startswith("conv_seg"):
    #         param.requires_grad = False
    #     else:
    #         param.requires_grad = True

    if args.load_from_checkpoint:
        conv_module = ConvModule.load_from_checkpoint(
            args.ckp_path, model=model, args=args
        )
    else:
        conv_module = ConvModule(model, args)

    logger = CometLogger(
        api_key=comet_config["api_key"],
        project_name=comet_config["project_name"],
        workspace=comet_config["workspace"],
        experiment_name="fullyConvModel_{}_{}".format(args.epochs, args.learning_rate),
    )

    ckpt_callback = ModelCheckpoint(
        dirpath=args.save_model_path, save_top_k=-1, filename="{epoch}-{val_loss:.2f}"
    )
    trainer = Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,
        logger=logger,
        # val_check_interval=args.valid_every,
        callbacks=ckpt_callback,
        resume_from_checkpoint=args.ckp_path,
    )

    trainer.fit(conv_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # distributed training setup
    parser.add_argument(
        "-g", "--gpus", default=1, type=int, help="number of gpus per node"
    )
    parser.add_argument(
        "-w",
        "--num_workers",
        default=4,
        type=int,
        help="n data loading workers, default 0 = main process only",
    )

    # train and valid
    parser.add_argument(
        "--path_to_npy_data",
        default="data/NPY/volumes/",
        type=str,
        help="json file to load training data",
    )
    parser.add_argument(
        "--path_to_npy_targets",
        default="data/NPY/labels/",
        type=str,
        help="json file to load testing data",
    )
    parser.add_argument(
        "--valid_every",
        default=750,
        required=False,
        type=int,
        help="valid after every N iteration",
    )

    # dir and path for models and logs
    parser.add_argument(
        "--save_model_path", default="models/", type=str, help="path to save model"
    )
    parser.add_argument(
        "--ckp_path",
        default=None,
        required=False,
        type=str,
        help="path to load a pretrain model to continue training",
    )
    parser.add_argument(
        "--load_from_checkpoint",
        default=False,
        required=False,
        type=bool,
        help="check path to resume from",
    )

    # general
    parser.add_argument(
        "--epochs", default=100, type=int, help="number of total epochs to run"
    )
    parser.add_argument("--batch_size", default=32, type=int, help="size of batch")
    parser.add_argument(
        "--learning_rate", default=9e-7, type=float, help="learning rate"
    )
    parser.add_argument(
        "--pct_start",
        default=0.3,
        type=float,
        help="percentage of growth phase in one cycle",
    )
    parser.add_argument(
        "--div_factor", default=100, type=int, help="div factor for one cycle"
    )
    parser.add_argument(
        "--hparams_override",
        default={},
        type=str,
        required=False,
        help='override the hyper parameters, should be in form of dict. ie. {"attention_layers": 16 }',
    )

    args = parser.parse_args()
    # args.hparams_override = ast.literal_eval(args.hparams_override)
    if args.save_model_path:
        if not os.path.isdir(os.path.dirname(args.save_model_path)):
            raise Exception(
                "the directory for path {} does not exist".format(args.save_model_path)
            )

    train(args)
