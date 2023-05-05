import wandb

from datetime import date, datetime
import os
from comet_ml import Experiment


import pickle
import random
from copy import deepcopy
import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.active_learning.active_config import ExperimentConfig
from src.active_learning.model_wrapper import CustomModelWrapper

from baal.active import ActiveLearningDataset, get_heuristic, ActiveLearningLoop
from src.active_learning.metrics import ClassificationReport
from baal.utils.metrics import Accuracy
from baal.active.heuristics import BALD
from baal.active.heuristics.stochastics import PowerSampling
from baal.bayesian.dropout import patch_module

from torchvision import transforms
from tqdm import tqdm

from src.configs.comet_configs import comet_config
from src.dataset.dataset import Data, VolumeDataset
from src.dataset.utils import (
    balanced_active_learning_split,
    get_balanced_initial_pool,
    get_class_files,
    train_test_val_split,
)
from src.features.convert_images_to_npy import process_cncb_data
from src.models.models import CustomVGG19

log = structlog.get_logger("ActiveLearningModule")


def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


class ActiveLearningModule:
    """
    Active Learning Loop object.
    Args:
        model: The model to train.
        dataset: The full dataset.
        train_transforms (Optional[Callable]): torchvision.transform pipeline for training.
        test_transforms (Optional[Callable]): torchvision.transform pipeline for testing.
    """

    def __init__(
        self,
        model,
        dataset: Data,
        cfg: ExperimentConfig,
        train_transforms,
        test_transforms,
        use_cuda: bool = True,
        load_state: dict = {"datasets": None, "model": None, "uncertainty": None},
        experiment_name: str = "default_experiment_name",
        resume_learning: bool = False,
    ) -> None:
        self.model = model
        self.use_cuda = use_cuda
        self.cfg = cfg
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.dataset = dataset
        self.experiment_name = experiment_name
        self.resume_learning = resume_learning

        self.load_dataset = load_state["datasets"]
        self.load_model = load_state["model"]
        self.load_uncertainty = load_state["uncertainty"]

        self.experiment_path = os.path.join("./checkpoints", self.experiment_name)

    def weights_reset(self, model):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def get_datasets(self, verbose: bool = True, save_state: bool = True):
        if self.load_dataset is not None:
            with open(self.load_dataset, "rb") as handle:
                datasets = pickle.load(handle)
                X_train = datasets["X_train"]
                y_train = datasets["y_train"]
                X_test = datasets["X_test"]
                y_test = datasets["y_test"]
                X_val = datasets["X_val"]
                y_val = datasets["y_val"]
                initial_pool = datasets["inital_pool"]
        else:
            X_train, X_test, X_val, y_train, y_test, y_val = train_test_val_split(
                self.dataset.files, self.dataset.lbls, train_size=0.7, test_size=0.2
            )
            initial_pool = get_balanced_initial_pool(
                y_train, initial_pool=self.cfg.initial_pool
            )
        # val set balancing
        num_samples = np.unique(y_val, return_counts=True)[1][-1]
        indices = []
        samples = [0, 0, 0]
        for idx, cls in enumerate(y_val):
            if samples[cls] < num_samples:
                indices.append(idx)
                samples[cls] += 1
            if np.sum(samples) == num_samples * 3:
                break
        X_val = [X_val[idx] for idx in indices]
        y_val = [y_val[idx] for idx in indices]
        train = VolumeDataset(
            files=X_train, lbls=y_train, train_transform=self.train_transforms
        )
        val = VolumeDataset(
            files=X_val, lbls=y_val, train_transform=self.test_transforms
        )

        test = VolumeDataset(
            files=X_test, lbls=y_test, train_transform=self.test_transforms
        )

        if verbose:
            log.info("Train set len {}".format(len(train)))
            log.info("Test set len {}".format(len(test)))
            len_test_samples = np.unique(y_test, return_counts=True)[1]
            len_val_samples = np.unique(y_val, return_counts=True)[1]
            log.info(
                "Test set balance cp:{} ncp: {} normal: {}".format(
                    len_test_samples[0], len_test_samples[1], len_test_samples[2]
                )
            )
            log.info(
                "Val set balance cp:{} ncp: {} normal: {}".format(
                    len_val_samples[0], len_val_samples[1], len_val_samples[2]
                )
            )
        active_set = ActiveLearningDataset(
            train, pool_specifics={"train_transform": self.test_transforms}
        )

        if save_state:
            data = {}
            data["X_train"] = X_train
            data["y_train"] = y_train
            data["X_test"] = X_test
            data["y_test"] = y_test
            data["X_val"] = X_val
            data["y_val"] = y_val
            data["inital_pool"] = initial_pool
            # Store data (serialize)
            with open(
                os.path.join(self.experiment_path, "datasets.pickle"), "wb"
            ) as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        active_set.can_label = False

        active_set.label(initial_pool)
        return active_set, val, test

    def train(self):
        os.mkdir(self.experiment_path)
        os.mkdir(os.path.join(self.experiment_path, "uncertainty"))
        seed_all(123)
        experiment = Experiment(
            api_key=comet_config["api_key"],
            project_name=comet_config["project_name"],
            workspace=comet_config["workspace"],
        )
        experiment.set_name(self.experiment_name)
        torch.backends.cudnn.benchmark = True
        if not self.use_cuda:
            print("No CUDA found.")

        self.model = self.model.cuda()

        active_set, val_set, test_set = self.get_datasets()
        if self.cfg.heuristic == "powerbald":
            heuristic = PowerSampling(
                BALD(), query_size=self.cfg.query_size, temperature=1.0
            )
        else:
            heuristic = get_heuristic(self.cfg.heuristic)
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.cfg.lr,
            momentum=0.9,
        )

        self.model_wrapper = CustomModelWrapper(
            self.model, criterion, replicate_in_memory=False
        )

        if self.load_model is not None:
            self.model_wrapper.model.load_state_dict(
                torch.load(self.load_model)["model"]
            )
            active_set.load_state_dict(torch.load(self.load_model)["active_set"])

        self.model_wrapper.add_metric(
            name="cls_report", initializer=lambda: ClassificationReport(num_classes=3)
        )
        self.model_wrapper.add_metric(name="accuracy", initializer=lambda: Accuracy())

        experiment.log_parameters(self.cfg)

        active_loop = ActiveLearningLoop(
            active_set,
            self.model_wrapper.predict_on_dataset,
            heuristic,
            self.cfg.query_size,
            batch_size=2,
            iterations=self.cfg.iterations,
            use_cuda=self.use_cuda,
            max_sample=1000,  # parameter to tweak device based
            verbose=True,
            uncertainty_folder=os.path.join(self.experiment_path, "uncertainty"),
        )
        init_weights = deepcopy(self.model_wrapper.state_dict())

        labelling_progress = active_set._labelled.copy().astype(np.uint16)
        for epoch in tqdm(range(self.cfg.epoch)):
            self.model_wrapper.load_state_dict(init_weights)
            self.weights_reset(self.model_wrapper.model)
            optimizer.param_groups[0]["lr"] = self.cfg.lr
            scheduler = ReduceLROnPlateau(optimizer, "min", patience=3)
            hist, best_weight = self.model_wrapper.train_and_test_on_datasets(
                active_set,
                val_set,
                optimizer=optimizer,
                scheduler=scheduler,
                batch_size=self.cfg.batch_size,
                epoch=500,
                learning_epoch=self.cfg.learning_epoch,  # self.cfg.learning_epoch,
                use_cuda=self.use_cuda,
                return_best_weights=True,
                patience=self.cfg.patience,
                min_epoch_for_es=self.cfg.min_epoch_for_es * epoch,
            )
            print(
                f'[{epoch + 1}/ {self.cfg.epoch}] training acc: {(self.model_wrapper.get_metrics()["train_accuracy"])}'
            )
            print(
                f'[{epoch + 1}/ {self.cfg.epoch}] train cls data: {self.model_wrapper.metrics["train_cls_report"].class_data}'
            )
            experiment.log_metrics(
                {"val_loss": self.model_wrapper.get_metrics()["test_loss"]},
                epoch=epoch,
            )

            experiment.log_metrics(
                {"val_accuracy": self.model_wrapper.get_metrics()["test_accuracy"]},
                epoch=epoch,
            )

            self.model_wrapper.load_state_dict(best_weight)

            self.model_wrapper.test_on_dataset(
                test_set, self.cfg.batch_size, self.use_cuda
            )
            print(
                f'[{epoch + 1}/ {self.cfg.epoch}] test acc: {np.mean(self.model_wrapper.get_metrics()["test_cls_report"]["accuracy"])}'
            )
            print(
                f'[{epoch + 1}/ {self.cfg.epoch}] test cls data: {self.model_wrapper.metrics["test_cls_report"].class_data}'
            )
            experiment.log_metrics(
                {"train_loss": self.model_wrapper.get_metrics()["train_loss"]},
                epoch=epoch,
            )
            experiment.log_metrics(
                {"test_loss": self.model_wrapper.get_metrics()["test_loss"]},
                epoch=epoch,
            )

            experiment.log_metrics(
                {"test_accuracy": self.model_wrapper.get_metrics()["test_accuracy"]},
                epoch=epoch,
            )
            experiment.log_metrics(
                {"train_accuracy": self.model_wrapper.get_metrics()["train_accuracy"]},
                epoch=epoch,
            )

            experiment.log_metrics({"dataset_len": active_set.n_labelled}, epoch=epoch)
            experiment.log_metrics(
                {
                    "test_cls_report": self.model_wrapper.get_metrics()[
                        "test_cls_report"
                    ]
                },
                epoch=epoch,
            )
            experiment.log_metrics(
                {
                    "train_cls_report": self.model_wrapper.get_metrics()[
                        "train_cls_report"
                    ]
                },
                epoch=epoch,
            )
            should_continue = active_loop.step()
            # Keep track of progress
            labelling_progress += active_set._labelled.astype(np.uint16)

            torch.save(
                {
                    "model": best_weight,
                    "labelling_progress": labelling_progress,
                    "active_set": active_set.state_dict(),
                },
                os.path.join(
                    self.experiment_path,
                    "model_{}_{}_checkpoint.pth".format(epoch, self.cfg.heuristic),
                ),
            )
            if not should_continue:
                break

        torch.save(
            {
                "model": best_weight,
                "labelling_progress": labelling_progress,
                "active_set": active_set.state_dict(),
            },
            os.path.join(
                self.experiment_path,
                "model_{}_{}_checkpoint.pth".format(epoch, self.cfg.heuristic),
            ),
        )
        return


if __name__ == "__main__":
    from torch.hub import load_state_dict_from_url
    from torchvision import models

    # model = densenet121(weights=DenseNet121_Weights.DEFAULT, drop_rate=0.5)
    # model.classifier = nn.Sequential(
    #     nn.Linear(in_features=1024, out_features=512, bias=True),
    #     nn.ReLU(inplace=True),
    #     nn.Dropout(p=0.5, inplace=False),
    #     nn.Linear(in_features=512, out_features=3, bias=True),
    # )
    def vgg16(num_classes):
        model = models.vgg16(pretrained=False, num_classes=num_classes)
        weights = load_state_dict_from_url(
            "https://download.pytorch.org/models/vgg16-397923af.pth"
        )
        weights = {k: v for k, v in weights.items() if "classifier.6" not in k}
        model.load_state_dict(weights, strict=False)
        return model

    model = vgg16(num_classes=3)
    model = patch_module(model)
    root_dir = "./data/"
    files, classes = process_cncb_data(root_dir=root_dir)
    random_subset = random.sample(range(0, len(files)), 10000)

    new_files = [files[idx] for idx in random_subset]
    new_classes = [classes[idx] for idx in random_subset]
    # dataset = Data(data_path="data/NPY/volumes/", target_path="data/NPY/labels/")
    dataset = VolumeDataset(files=files, lbls=classes)
    train_transform = transforms.Compose(
        [
            # transforms.Resize((224, 224), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(
                brightness=0, contrast=0, saturation=(0.5, 1.5), hue=(-0.1, 0.1)
            ),
            transforms.RandomErasing(p=0.25),
            transforms.RandomRotation(degrees=(-30, 10)),
        ]
    )

    load_state = {
        "datasets": None,  # "./checkpoints/VGG16_bald_100_0.0001_12_04_2023_6-25/datasets.pickle",
        "model": None,  # "./checkpoints/VGG_bald_100_0.0001_28_03_2023_10-0/model_7_bald_checkpoint.pth",
        "uncertainty": None,  # "./checkpoints/VGG_bald_100_0.0001_28_03_2023_10-0/uncertainty/uncertainty_pool=1000_labelled=240.pkl",
    }

    test_transform = transforms.Compose([])
    ActiveLearning = ActiveLearningModule(
        model=model,
        dataset=dataset,
        cfg=ExperimentConfig,
        train_transforms=train_transform,
        test_transforms=test_transform,
        load_state=load_state,
        experiment_name="VGG16_{}_{}_{}_{}_{}-{}".format(
            ExperimentConfig.heuristic,
            ExperimentConfig.epoch,
            ExperimentConfig.lr,
            date.today().strftime("%d_%m_%Y"),
            datetime.now().hour,
            datetime.now().minute,
        ),
    )

    ActiveLearning.train()
