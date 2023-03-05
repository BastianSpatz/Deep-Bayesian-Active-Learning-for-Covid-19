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
from src.active_learning.active_config import ExperimentConfig
from src.active_learning.model_wrapper import CustomModelWrapper

from baal.active import ActiveLearningDataset, get_heuristic
from src.active_learning.active_loop import ActiveLearningLoop
from baal.utils.metrics import Accuracy, ClassificationReport


from torchvision import transforms
from tqdm import tqdm

from src.configs.comet_configs import comet_config
from src.dataset.dataset import Data, CustomSubset
from src.dataset.utils import (
    get_balanced_initial_pool,
    train_test_val_split,
)
from src.models.models import CustomVGG16, MobileNetV2

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
        load_state: dict = {"datasets": None, "model": None},
        experiment_name: str = "default_experiment_name",
        resume_learning: bool = False,
    ) -> None:
        self.model = model
        self.use_cuda = use_cuda
        self.cfg = cfg
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.dataset = dataset
        self.load_state = load_state
        self.experiment_name = experiment_name
        self.resume_learning = resume_learning

        self.experiment_path = os.path.join("./checkpoints", self.experiment_name)

    def weights_reset(self, model):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def get_datasets(self, verbose: bool = True, save_state: bool = True):
        if self.load_state["datasets"] is not None:
            dataset_path = self.load_state["datasets"]
            with open(dataset_path, "rb") as handle:
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
        train = CustomSubset(
            self.dataset, X_train, y_train, transform=self.train_transforms
        )
        val = CustomSubset(self.dataset, X_val, y_val, transform=self.test_transforms)

        test = CustomSubset(
            self.dataset, X_test, y_test, transform=self.test_transforms
        )

        if verbose:
            log.info("Train set len {}".format(len(train)))
            log.info("Test set len {}".format(len(test)))

        active_set = ActiveLearningDataset(
            train, pool_specifics={"transform": self.test_transforms}
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

        heuristic = get_heuristic(self.cfg.heuristic)
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.cfg.lr,
            momentum=0.9,
        )
        self.model_wrapper = CustomModelWrapper(
            self.model, criterion, replicate_in_memory=False
        )

        if self.load_state["model"] is not None:
            self.model_wrapper.model.load_state_dict(
                torch.load(self.load_state["model"])["model"]
            )
            active_set._labelled = torch.load(self.load_state["model"])[
                "labelling_progress"
            ]

        self.model_wrapper.add_metric(
            name="accuracy", initializer=lambda: Accuracy(task="multiclass")
        )

        experiment.log_parameters(self.cfg)

        active_loop = ActiveLearningLoop(
            active_set,
            self.model_wrapper.predict_on_dataset,
            heuristic,
            self.cfg.query_size,
            batch_size=8,
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

            hist, best_weight = self.model_wrapper.train_and_test_on_datasets(
                active_set,
                val_set,
                optimizer=optimizer,
                batch_size=self.cfg.batch_size,
                epoch=100,
                learning_epoch=self.cfg.learning_epoch,
                use_cuda=self.use_cuda,
                return_best_weights=True,
                patience=self.cfg.patience,
                min_epoch_for_es=self.cfg.min_epoch_for_es,
            )
            print(
                f'[{epoch + 1}/ {self.cfg.epoch}] training acc: {self.model_wrapper.get_metrics()["train_accuracy"]}'
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
                f'[{epoch + 1}/ {self.cfg.epoch}] test acc: {self.model_wrapper.get_metrics()["test_accuracy"]}'
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
    model = CustomVGG16(active_learning_mode=True)
    # model = MobileNetV2()
    dataset = Data(data_path="data/NPY/volumes/", target_path="data/NPY/labels/")
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            # transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.1)),
            transforms.ColorJitter(
                brightness=0, contrast=0, saturation=(0.5, 1.5), hue=(-0.1, 0.1)
            ),
            # transforms.GaussianBlur(kernel_size=(5, 5), sigma=(10, 10)),
            transforms.RandomErasing(p=0.25),
        ]
    )

    load_state = {
        "datasets": "./checkpoints/VGG_bald_20_0.001_04_03_2023_15-4/datasets.pickle",
        "model": None,
    }

    test_transform = transforms.Compose([])
    ActiveLearning = ActiveLearningModule(
        model=model,
        dataset=dataset,
        cfg=ExperimentConfig,
        train_transforms=train_transform,
        test_transforms=test_transform,
        load_state=load_state,
        experiment_name="VGG_{}_{}_{}_{}_{}-{}".format(
            ExperimentConfig.heuristic,
            ExperimentConfig.epoch,
            ExperimentConfig.lr,
            date.today().strftime("%d_%m_%Y"),
            datetime.now().hour,
            datetime.now().minute,
        ),
    )
    ActiveLearning.train()
    # print(torch.load(load_state["model"])["labelling_progress"])
