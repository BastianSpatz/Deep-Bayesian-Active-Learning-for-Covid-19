import random
from copy import deepcopy
import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.optim as optim
from src.active_learning.active_config import ExperimentConfig

from baal import ModelWrapper
from baal.active import ActiveLearningDataset, get_heuristic
from baal.active.active_loop import ActiveLearningLoop
from baal.utils.metrics import Accuracy, ClassificationReport

# from torchmetrics import Accuracy
from comet_ml import Experiment
from torchvision import transforms
from tqdm import tqdm

from src.configs.comet_configs import comet_config
from src.dataset.dataset import Data, CustomSubset
from src.dataset.utils import balanced_active_learning_split
from src.models.conv_net import CustomVGG16

log = structlog.get_logger("CustomActiveLearningLoop")


def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


class CustomActiveLearningLoop:
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
    ) -> None:
        self.model = model
        self.use_cuda = use_cuda
        self.cfg = cfg
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.dataset = dataset

    def weights_reset(self, model):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def get_datasets(self, verbose: bool = True):
        (
            train_indices,
            val_indices,
            test_indices,
            initial_pool,
        ) = balanced_active_learning_split(
            dataset=self.dataset, initial_pool=self.cfg.initial_pool
        )
        # We use -1 to specify that the data is unlabeled.
        train = CustomSubset(
            self.dataset, train_indices, transform=self.train_transforms
        )
        # We use -1 to specify that the data is unlabeled.
        val = CustomSubset(
            self.dataset,
            val_indices,
            transform=self.test_transforms,
        )

        test = CustomSubset(
            self.dataset,
            test_indices,
            transform=self.test_transforms,
        )
        if verbose:
            log.info("Train set len", len=len(train))
            log.info("Test set len", len=len(test))

        active_set = ActiveLearningDataset(
            train, pool_specifics={"transform": self.test_transforms}
        )
        active_set.can_label = False

        active_set.label(initial_pool)
        return active_set, val, test

    def train(self):
        seed_all(123)
        experiment = Experiment(
            api_key=comet_config["api_key"],
            project_name=comet_config["project_name"],
            workspace=comet_config["workspace"],
        )
        torch.backends.cudnn.benchmark = True
        if not self.use_cuda:
            print("No CUDA found.")

        active_set, val_set, test_set = self.get_datasets()

        heuristic = get_heuristic(self.cfg.heuristic)
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.SGD(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=(1 - 0.5) * 0.5 / (active_set.n_unlabelled),
            momentum=0.9,
        )
        self.model_wrapper = ModelWrapper(
            self.model, criterion, replicate_in_memory=False
        )

        self.model_wrapper.add_metric(
            name="accuracy", initializer=lambda: Accuracy(task="multiclass")
        )

        experiment.log_parameters(self.cfg)

        active_loop = ActiveLearningLoop(
            active_set,
            self.model_wrapper.predict_on_dataset,
            heuristic,
            self.cfg.query_size,
            batch_size=5,
            iterations=self.cfg.iterations,
            use_cuda=self.use_cuda,
            max_sample=-1,  # parameter to tweak device based
            verbose=True,
        )
        init_weights = deepcopy(self.model_wrapper.state_dict())

        for epoch in tqdm(range(self.cfg.epoch)):
            self.model_wrapper.load_state_dict(init_weights)
            self.weights_reset(self.model_wrapper.model)

            _ = self.model_wrapper.train_on_dataset(
                active_set,
                optimizer=optimizer,
                batch_size=8,
                epoch=30 + epoch * 5,
                use_cuda=self.use_cuda,
            )

            self.model_wrapper.test_on_dataset(
                test_set, self.cfg.batch_size, self.use_cuda
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

            if not should_continue:
                break
        return


if __name__ == "__main__":

    model = CustomVGG16(active_learning_mode=True)
    dataset = Data(data_path="data/NPY/volumes/", target_path="data/NPY/labels/")
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=126, std=94),
            transforms.ColorJitter(
                brightness=0, contrast=0, saturation=(0.5, 1.5), hue=(-0.1, 0.1)
            ),
            transforms.RandomErasing(p=0.1),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Normalize(mean=126, std=94),
        ]
    )

    ActiveLearning = CustomActiveLearningLoop(
        model=model,
        dataset=dataset,
        cfg=ExperimentConfig,
        train_transforms=train_transform,
        test_transforms=test_transform,
    )
    ActiveLearning.train()
