from argparse import ArgumentParser
import torch
import wandb
import os
import pickle
from copy import deepcopy
from datetime import date, datetime

import numpy as np
import structlog
import torch.nn as nn
import torch.optim as optim
from baal.active import ActiveLearningDataset, ActiveLearningLoop, get_heuristic
from baal.active.heuristics import BALD
from baal.active.heuristics.stochastics import PowerSampling
from baal.utils.metrics import Accuracy
from torch.hub import load_state_dict_from_url
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from torchvision.transforms import transforms
from tqdm import tqdm

from src.active_learning.metrics import ClassificationReport
from src.active_learning.model_wrapper import CustomModelWrapper
from src.dataset.dataset import VolumeDataset
from src.dataset.utils import get_balanced_initial_pool, train_test_val_split
from src.features.convert_images_to_npy import process_cncb_data

log = structlog.get_logger("ActiveLearningModule")


def vgg16(num_classes):
    model = models.vgg16(pretrained=False, num_classes=num_classes)
    weights = load_state_dict_from_url(
        "https://download.pytorch.org/models/vgg16-397923af.pth"
    )
    weights = {k: v for k, v in weights.items() if "classifier.6" not in k}
    model.load_state_dict(weights, strict=False)
    return model.cuda()


def get_datasets(
    train_transforms,
    test_transforms,
    config,
    verbose: bool = True,
    save_state: bool = True,
    dataset_path: str = None,
):
    if dataset_path is not None:
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
        root_dir = "./data/"
        files, classes = process_cncb_data(root_dir=root_dir)

        X_train, X_test, X_val, y_train, y_test, y_val = train_test_val_split(
            files, classes, train_size=0.7, test_size=0.2
        )
        initial_pool = get_balanced_initial_pool(
            y_train, initial_pool=config.initial_pool
        )

    train = VolumeDataset(files=X_train, lbls=y_train, train_transform=train_transforms)
    val = VolumeDataset(files=X_val, lbls=y_val, train_transform=test_transforms)

    test = VolumeDataset(files=X_test, lbls=y_test, train_transform=test_transforms)

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
        train, pool_specifics={"train_transform": test_transforms}
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
            os.path.join(config.experiment_path, "datasets.pickle"), "wb"
        ) as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    active_set.can_label = False

    active_set.label(initial_pool)
    return active_set, val, test


def create_heuristic(config):
    if config.heuristic == "powerbald":
        heuristic = PowerSampling(BALD(), query_size=config.query_size, temperature=1.0)
    else:
        heuristic = get_heuristic(config.heuristic)
    return heuristic


def get_optmizer(model, config):
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=0.9,
    )
    return optimizer


def get_model_wrapper(model, criterion):
    model_wrapper = CustomModelWrapper(model, criterion, replicate_in_memory=False)
    return model_wrapper


def train(model, active_set, val_set, test_set, criterion, config):
    # Tell wandb to watch what the model gets up to: gradients, weights, and more!

    wandb.watch(model, criterion, log="all", log_freq=10)
    optimizer = get_optmizer(model, config)
    model_wrapper = get_model_wrapper(model, criterion)
    model_wrapper.add_metric(
        name="cls_report", initializer=lambda: ClassificationReport(num_classes=3)
    )
    model_wrapper.add_metric(name="accuracy", initializer=lambda: Accuracy())
    init_weights = deepcopy(model_wrapper.state_dict())
    heuristic = create_heuristic(config)
    active_loop = ActiveLearningLoop(
        active_set,
        model_wrapper.predict_on_dataset,
        heuristic,
        config.query_size,
        batch_size=2,
        iterations=config.iterations,
        use_cuda=True,
        max_sample=1000,  # parameter to tweak device based
        verbose=True,
    )
    labelling_progress = active_set._labelled.copy().astype(np.uint16)
    for epoch in tqdm(range(config.epochs)):
        model_wrapper.load_state_dict(init_weights)
        should_continue = train_epoch(
            epoch,
            model_wrapper,
            active_loop,
            active_set,
            val_set,
            test_set,
            optimizer,
            labelling_progress,
            config,
        )
        if not should_continue:
            break


def train_epoch(
    epoch,
    model_wrapper,
    active_loop,
    active_set,
    val_set,
    test_set,
    optimizer,
    labelling_progress,
    config,
):
    optimizer.param_groups[0]["lr"] = config.learning_rate
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=5)
    hist, best_weight = model_wrapper.train_and_test_on_datasets(
        active_set,
        val_set,
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=config.batch_size,
        epoch=500,
        learning_epoch=config.learning_epoch,  # config.learning_epoch,
        use_cuda=True,
        return_best_weights=True,
        patience=config.patience,
        min_epoch_for_es=0,
    )
    ## Log train and validation
    wandb.log({"train_accuracy": model_wrapper.get_metrics()["train_accuracy"]})
    wandb.log({"train_loss": model_wrapper.get_metrics()["train_loss"]})

    wandb.log({"val_accuracy": model_wrapper.get_metrics()["test_accuracy"]})
    wandb.log({"val_loss": model_wrapper.get_metrics()["test_loss"]})
    model_wrapper.load_state_dict(best_weight)

    model_wrapper.test_on_dataset(test_set, config.batch_size, True)

    wandb.log({"test_accuracy": model_wrapper.get_metrics()["test_accuracy"]})
    wandb.log({"test_loss": model_wrapper.get_metrics()["test_loss"]})

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
            config.experiment_path,
            "model_{}_{}_checkpoint.pth".format(epoch, config.heuristic),
        ),
    )
    log_artifact(
        os.path.join(
            config.experiment_path,
            "model_{}_{}_checkpoint.pth".format(epoch, config.heuristic),
        ),
        os.path.join(
            config.experiment_path,
            "model_{}_{}_checkpoint.pth".format(epoch, config.heuristic),
        ),
        model_wrapper.get_metrics()["test_accuracy"],
    )
    return should_continue


def log_artifact(filename, model_path, metric_val):
    artifact = wandb.Artifact(
        filename, type="model", metadata={"Test score": metric_val}
    )
    artifact.add_file(model_path)
    wandb.run.log_artifact(artifact)


def main(config=None):
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        experiment_name = "VGG16_{}_{}_{}_{}_{}-{}".format(
            config.heuristic,
            config.epochs,
            config.learning_rate,
            date.today().strftime("%d_%m_%Y"),
            datetime.now().hour,
            datetime.now().minute,
        )
        experiment_path = os.path.join("./checkpoints", experiment_name)
        config.experiment_path = experiment_path
        os.mkdir(experiment_path)
        os.mkdir(os.path.join(experiment_path, "uncertainty"))
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

        test_transform = transforms.Compose([])

        active_set, val, test = get_datasets(
            train_transforms=train_transform,
            test_transforms=test_transform,
            config=config,
        )

        model = vgg16(3)
        criterion = nn.CrossEntropyLoss()
        train(model, active_set, val, test, criterion, config)


if __name__ == "__main__":
    parser = ArgumentParser(description="vgg16 Training Script")
    parser.add_argument(
        "-n",
        "--epochs",
        dest="epochs",
        default=10,
        type=int,
        help="Number of epochs",
    )
    parser.add_argument(
        "-b", "--batch_size", dest="batch_size", default=64, type=int, help="Batch size"
    )
    parser.add_argument(
        "-l",
        "--learning_rate",
        dest="learning_rate",
        type=float,
        default=0.0002,
        help="Learning rate",
    )
    parser.add_argument(
        "-le",
        "--learning_epoch",
        dest="learning_epoch",
        type=int,
        default=1,
        help="Learning epoch",
    )
    parser.add_argument(
        "-p",
        "--patience",
        dest="patience",
        type=int,
        default=1,
        help="Patience",
    )
    parser.add_argument(
        "-it",
        "--iterations",
        dest="iterations",
        type=int,
        default=1,
        help="iterations",
    )
    parser.add_argument(
        "-ip",
        "--initial_pool",
        dest="initial_pool",
        type=int,
        default=3,
        help="initial_pool",
    )
    parser.add_argument(
        "-heur",
        "--heuristic",
        dest="heuristic",
        type=str,
        default="bald",
    )
    parser.add_argument(
        "-qs",
        "--query_size",
        dest="query_size",
        type=int,
        default=30,
        help="query_size",
    )
    args = parser.parse_args()

    main(args)
