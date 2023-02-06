import argparse
import os
import random
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core.module import LightningModule
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

from src.configs.comet_configs import comet_config
from src.dataset.dataset import CustomDataset, collate_fn_padd
from src.dataset.utils import train_test_validation_split
from src.models.conv_net import ConvNN, MnistExampleModel


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
        self.accuracy = Accuracy(task='multiclass')
        self.save_hyperparameters()
        self.dataset = CustomDataset(path_to_npy_data=args.path_to_npy_data,
                             path_to_npy_targets=args.path_to_npy_targets)
        self.train_indices, self.test_indices, _= self.dataset.get_active_learning_datasets()

    def training_step(self, batch, batch_idx):
        # loss = self.step(batch)
        volumes, labels = batch
        output = self.model(volumes)
        loss = self.criterion(output, labels)
        acc = self.accuracy(output, labels)
        self.log("train_accuracy",  acc, on_step=False, on_epoch=True, batch_size=self.args.batch_size)
        self.log("lr", self.scheduler.get_last_lr()[0], on_epoch=True, on_step=False, logger=True, batch_size=self.args.batch_size)
        self.log("train_loss",  loss.detach(), on_epoch=True, batch_size=self.args.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        # loss = self.step(batch)
        volumes, labels = batch
        output = self.model(volumes)
        loss = self.criterion(output, labels)
        acc = self.accuracy(output, labels)
        self.log("val_accuracy",  acc, on_epoch=True, batch_size=self.args.batch_size)
        self.log("val_loss",  loss.detach(), on_epoch=True, batch_size=self.args.batch_size)

    def configure_optimizers(self):
        self.optimizer = optim.SGD(self.model.parameters(), 
                                    lr=self.lr, 
                                    momentum=0.9, # Annealing applied to learning rate after each epoch
                                    nesterov=True,
                                    weight_decay = 1e-5)  # Initial Weight Decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.25)
        return {
            'optimizer': self.optimizer,
            'lr_scheduler': self.scheduler, # Changed scheduler to lr_scheduler
                    }

    def train_dataloader(self):
        train_datset = torch.utils.data.Subset(self.dataset, self.train_indices)
        return DataLoader(dataset=train_datset,
                            num_workers=self.args.num_workers,
                            batch_size=self.args.batch_size,
                            pin_memory=True,
                            collate_fn=collate_fn_padd)

    def val_dataloader(self):
        test_datset = torch.utils.data.Subset(self.dataset, self.test_indices)
        return DataLoader(dataset=test_datset,
                            num_workers=self.args.num_workers,
                            batch_size=self.args.batch_size,
                            pin_memory=True,
                            collate_fn=collate_fn_padd)

def train(args):

    random.seed(28)
    torch.manual_seed(28)

    model = MnistExampleModel()
    if args.load_from_checkpoint:
        conv_module = ConvModule.load_from_checkpoint(args.ckp_path, model=model, args=args)
    else:
        conv_module = ConvModule(model, args)
    # speech_module = SpeechModule(num_cnn_layers=1, num_rnn_layers=2, rnn_dim=512, num_classes=29, n_feats=128)

    logger = CometLogger(
        api_key=comet_config["api_key"],
        project_name=comet_config["project_name"],
        workspace=comet_config["workspace"],
        experiment_name="fullyConvModel_{}_{}".format(args.epochs, args.learning_rate))

    ckpt_callback = ModelCheckpoint(dirpath=args.save_model_path, save_top_k=-1, filename='{epoch}-{val_loss:.2f}')
    trainer = Trainer(
		max_epochs=args.epochs, 
		accelerator='gpu',
		devices=1,
		logger=logger,
		# val_check_interval=args.valid_every,
		callbacks=ckpt_callback,
		resume_from_checkpoint=args.ckp_path,
        auto_lr_find=True
		)

    trainer.fit(conv_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
	# distributed training setup
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-w', '--num_workers', default=4, type=int,
                        help='n data loading workers, default 0 = main process only')

    # train and valid
    parser.add_argument('--path_to_npy_data', default="data/NPY/volumes/", type=str,
                        help='json file to load training data')
    parser.add_argument('--path_to_npy_targets', default="data/NPY/labels/", type=str,
                        help='json file to load testing data')
    parser.add_argument('--valid_every', default=750, required=False, type=int,
                        help='valid after every N iteration')

    # dir and path for models and logs
    parser.add_argument('--save_model_path', default="models/", type=str,
                        help='path to save model')
    parser.add_argument('--ckp_path', default=None, required=False, type=str,
                        help='path to load a pretrain model to continue training')
    parser.add_argument('--load_from_checkpoint', default=False, required=False, type=bool,
                        help='check path to resume from')

    # general
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=2, type=int, help='size of batch')
    parser.add_argument('--learning_rate', default=9e-4, type=float, help='learning rate')
    parser.add_argument('--pct_start', default=0.3, type=float, help='percentage of growth phase in one cycle')
    parser.add_argument('--div_factor', default=100, type=int, help='div factor for one cycle')
    parser.add_argument("--hparams_override", default={},
                        type=str, required=False, 
                        help='override the hyper parameters, should be in form of dict. ie. {"attention_layers": 16 }')


    args = parser.parse_args()
    # args.hparams_override = ast.literal_eval(args.hparams_override)
    if args.save_model_path:
        if not os.path.isdir(os.path.dirname(args.save_model_path)):
            raise Exception("the directory for path {} does not exist".format(args.save_model_path))

    train(args)
        