import random

from comet_ml import Experiment
from copy import deepcopy
from tqdm import tqdm

from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim

from src.configs.comet_configs import comet_config
from src.dataset.utils import train_test_validation_split
from src.dataset.dataset import collate_fn_padd

from baal.active import get_heuristic, ActiveLearningDataset
from baal.active.active_loop import ActiveLearningLoop
from baal.utils.metrics import Accuracy
from baal import ModelWrapper
from baal.bayesian.dropout import MCDropoutModule

class ActiveLearningTrainLoop:

    def __init__(self, model, dataset, cfg) -> None: 
        self.model = model
        self.dataset = dataset
        self.cfg = cfg

    def weights_reset(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    def get_datasets(self):
        train, test, valid = train_test_validation_split(dataset=self.dataset)
        # In a real application, you will want a validation set here.

        # Here we set `pool_specifics`, where we set the transform attribute for the pool.
        active_set = ActiveLearningDataset(train)

        # We start labeling randomly.
        active_set.label_randomly(self.cfg.initial_pool)
        return active_set, test, valid

    def train(self):
        random.seed(128)
        torch.manual_seed(128)

        experiment = Experiment(
                api_key=comet_config["api_key"],
                project_name=comet_config["project_name"],
                workspace=comet_config["workspace"],
                experiment_name="thursday")
        use_cuda = torch.cuda.is_available()
        torch.backends.cudnn.benchmark = True
        random.seed(28)
        torch.manual_seed(28)
        if not use_cuda:
            print("No CUDA found.")


        active_set, test_set, validation_set = self.get_datasets(self.cfg.initial_pool)

        heuristic = get_heuristic(self.cfg.heuristic)
        criterion = nn.CrossEntropyLoss()

        model = MCDropoutModule(self.model)
        optimizer = optim.Adam(model.parameters(), lr=self.cfg.lr)

        model=ModelWrapper(model, criterion, replicate_in_memory=False)

        model.add_metric(name='accuracy',initializer=lambda : Accuracy())

        experiment.log_parameters(self.cfg)

        active_loop = ActiveLearningLoop(
                        active_set,
                        model.predict_on_dataset,
                        heuristic,
                        self.cfg.query_size,
                        batch_size=1,
                        iterations=self.cfg.iterations,
                        use_cuda=use_cuda,
                        max_sample=100,
                        uncertainty_folder="output",
                        collate_fn=collate_fn_padd
                        )
        init_weights = deepcopy(model.state_dict())

        for epoch in tqdm(range(self.cfg.epochs)):
            model.load_state_dict(init_weights)
            model.train_on_dataset(
                active_set,
                optimizer,
                self.cfg["batch_size"],
                self.cfg["learning_epoch"],
                use_cuda,
                collate_fn=collate_fn_padd,
            )

            model.test_on_dataset(test_set, self.cfg["batch_size"], use_cuda, collate_fn=collate_fn_padd)
        
            experiment.log_metrics({"train_loss": model.get_metrics()["train_loss"]}, epoch=epoch)
            experiment.log_metrics({"test_loss": model.get_metrics()["test_loss"]}, epoch=epoch)
            experiment.log_metrics({"test_accuracy": model.get_metrics()["test_accuracy"]}, epoch=epoch)
            experiment.log_metrics({"train_accuracy": model.get_metrics()["train_accuracy"]}, epoch=epoch)
            
            should_continue = active_loop.step()
            # Keep track of progress
            # labelling_progress = active_set.labelled_map.astype(np.uint16)
            if not should_continue:
                break
        return 