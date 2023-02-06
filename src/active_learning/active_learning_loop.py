import random
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.active_learning.active_config import ExperimentConfig
from baal import ModelWrapper
from baal.active import ActiveLearningDataset, get_heuristic
from baal.active.active_loop import ActiveLearningLoop
from baal.bayesian.dropout import MCDropoutModule
from baal.utils.metrics import Accuracy
from comet_ml import Experiment
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import CometLogger
from tqdm import tqdm

from src.configs.comet_configs import comet_config
from src.dataset.dataset import CustomDataset, collate_fn_padd
from src.dataset.utils import train_test_validation_split
from src.models.conv_net import ConvNN, MnistExampleModel

def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


class ActiveLearningTrainLoop:

    def __init__(self, model, data_path, target_path, cfg) -> None: 
        self.model = model.cuda()
        self.cfg = cfg

        self.data_paths = []
        self.targets = []
        self.CP_files = []
        self.NCP_files = []
        self.Normal_files = []
        for idx, data_file_name in enumerate(os.listdir(data_path)):
            self.data_paths.append(data_path + data_file_name)
            # self.target_paths.append(path_to_npy_targets + target_file_name)
            self.targets.append(
                np.load(target_path + "label" + data_file_name[3:]))
            if "_CP_" in data_file_name:
                self.CP_files.append(idx)
            elif "_NCP_" in data_file_name:
                self.NCP_files.append(idx)
            elif "_Normal_" in data_file_name:
                self.Normal_files.append(idx)
            else:
                print("Weird path found: {}".format(data_file_name))

    
    def balanced_train_test_split(self, split):
        seed_all(self.seed)

        num_normal_files_train = int(len(self.Normal_files)*split[0])
        num_normal_files_test = int(len(self.Normal_files)*split[1])

        random.shuffle(self.CP_files)
        random.shuffle(self.NCP_files)
        random.shuffle(self.Normal_files)

        random_cp_files = set(self.CP_files)
        random_ncp_files = set(self.NCP_files)
        random_normal_files = set(self.Normal_files)

        self.cp_train_samples = set(random.sample(
            random_cp_files, num_normal_files_train))
        self.cp_test_samples = set(random.sample(
            random_cp_files - self.cp_train_samples, num_normal_files_test))

        self.ncp_train_samples = set(random.sample(
            random_ncp_files, num_normal_files_train))
        self.ncp_test_samples = set(random.sample(
            random_ncp_files - self.ncp_train_samples, num_normal_files_test))

        self.normal_train_samples = set(random.sample(
            random_normal_files, num_normal_files_train))
        self.normal_test_samples = set(random.sample(
            random_normal_files - self.normal_train_samples, num_normal_files_test))
        print("Number of samples per class: {}".format(num_normal_files_train))

    
    def get_active_learning_datasets(self, initial_pool=3, split=[0.8, 0.2]):
        self.balanced_train_test_split(split)
        num_class_samples = int(initial_pool/3)

        cp_labelled_samples = random.sample(self.cp_train_samples, num_class_samples)
        ncp_labelled_samples = random.sample(self.ncp_train_samples, num_class_samples)
        normal_labelled_samples = random.sample(self.normal_train_samples, num_class_samples)
        

        train_paths = self.cp_train_samples.union(self.ncp_train_samples, self.normal_train_samples)
        test_paths = self.cp_test_samples.union(self.ncp_test_samples, self.normal_test_samples)
        
        train_indices = [self.data_paths.index(path) for path in list(train_paths)]
        test_indices = [self.data_paths.index(path) for path in list(test_paths)]

        initial_pool_paths = cp_labelled_samples + ncp_labelled_samples + normal_labelled_samples
        initial_pool = [list(train_paths).index(path) for path in list(initial_pool_paths)]
        # initial_pool = [self.data_paths.index(path) for path in list(initial_pool_paths)]

        print("size train files: {}".format(len(train_indices)))
        print("size test files: {}".format(len(test_indices)))
        print("size initial pool files: {}".format(len(initial_pool)))
        
        return train_indices, test_indices, initial_pool

    def weights_reset(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

    def get_datasets(self):        
        train_indices, test_indices, initial_pool = self.dataset.get_active_learning_datasets(self.cfg.initial_pool)
        train = torch.utils.data.Subset(self.dataset, train_indices)
        test = torch.utils.data.Subset(self.dataset, test_indices)
        print("train set length {}".format(len(train)))
        print("test set length {}".format(len(test)))
        # Here we set `pool_specifics`, where we set the transform attribute for the pool.
        active_set = ActiveLearningDataset(train)

        # We start labeling randomly.
        active_set.label(initial_pool)
        return active_set, test

    def train(self):
        random.seed(128)
        torch.manual_seed(128)

        experiment = Experiment(
                api_key=comet_config["api_key"],
                project_name=comet_config["project_name"],
                workspace=comet_config["workspace"])
        use_cuda = torch.cuda.is_available()
        torch.backends.cudnn.benchmark = True
        random.seed(28)
        torch.manual_seed(28)
        if not use_cuda:
            print("No CUDA found.")


        active_set, test_set = self.get_datasets()


        heuristic = get_heuristic(self.cfg.heuristic)
        criterion = nn.CrossEntropyLoss()

        model = MCDropoutModule(self.model)
        optimizer = optim.SGD(model.parameters(), lr=self.cfg.lr,
                                    momentum=0.9, # Annealing applied to learning rate after each epoch
                                    nesterov=True,
                                    weight_decay = 1e-5)
        model=ModelWrapper(model, criterion, replicate_in_memory=False)

        model.add_metric(name='accuracy', initializer=lambda : Accuracy())

        experiment.log_parameters(self.cfg)

        active_loop = ActiveLearningLoop(
                        active_set,
                        model.predict_on_dataset,
                        heuristic,
                        self.cfg.query_size,
                        batch_size=1,
                        iterations=self.cfg.iterations,
                        use_cuda=use_cuda,
                        max_sample=200, # parameter to tweak device based
                        # uncertainty_folder="output",
                        collate_fn=collate_fn_padd
                        )
        init_weights = deepcopy(model.state_dict())

        for epoch in tqdm(range(self.cfg.epoch)):
            model.load_state_dict(init_weights)
            self.weights_reset(model)
            _ = model.train_on_dataset(
                active_set,
                optimizer,
                self.cfg.batch_size,
                self.cfg.learning_epoch + epoch,
                use_cuda,
                collate_fn=collate_fn_padd,
            )

            # prediction = model.predict_on_dataset(dataset=test_set, 
            #                                         batch_size=self.cfg.batch_size, 
            #                                         iterations=1,
            #                                         use_cuda=use_cuda, 
            #                                         collate_fn=collate_fn_padd)
            # print(prediction.shape)
            # np.savetxt("preds_{}.csv".format(epoch), np.argmax(prediction.squeeze(-1), axis=1), delimiter=",")

            model.test_on_dataset(test_set, self.cfg.batch_size, use_cuda, collate_fn=collate_fn_padd)
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


if __name__=="__main__":
    model = MnistExampleModel()
    dataset = CustomDataset(path_to_npy_data="data/NPY/volumes/", path_to_npy_targets="data/NPY/labels/")
    ActiveLearning=ActiveLearningTrainLoop(model=model, dataset=dataset, cfg=ExperimentConfig)
    ActiveLearning.train()

