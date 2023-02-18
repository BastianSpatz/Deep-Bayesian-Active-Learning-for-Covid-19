import random
from copy import deepcopy
from pprint import pprint
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.active_learning.active_config import ExperimentConfig
from baal import ModelWrapper
from baal.active import ActiveLearningDataset, get_heuristic
from baal.active.active_loop import ActiveLearningLoop
from baal.bayesian.dropout import MCDropoutModule, patch_module
from baal.utils.metrics import  Accuracy
# from torchmetrics import Accuracy
from comet_ml import Experiment
from torchvision import transforms
from tqdm import tqdm

from src.configs.comet_configs import comet_config
from src.dataset.dataset import CustomDataset, collate_fn_padd, CustomFileDataset
from src.dataset.utils import get_active_learning_datasets
from src.models.conv_net import CustomVGG16
from src.active_learning.model_wrapper import CustomModelWrapper

def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


class CustomActiveLearningTrainLoop:

    def __init__(self, model, dataset, cfg, train_transforms, test_transforms) -> None: 
        self.model = model.cuda()
        self.cfg = cfg
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.dataset = dataset


    def weights_reset(self, model):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def get_datasets(self):        
        train_indices, val_indices, test_indices, initial_pool = get_active_learning_datasets(dataset=self.dataset, initial_pool=self.cfg.initial_pool)
        # We use -1 to specify that the data is unlabeled.
        train = CustomFileDataset([dataset.data_paths[idx] for idx in train_indices],
                                        [dataset.targets[idx] for idx in train_indices],
                                            self.train_transforms)
        # We use -1 to specify that the data is unlabeled.
        val = CustomFileDataset([dataset.data_paths[idx] for idx in val_indices],
                                        [dataset.targets[idx] for idx in val_indices],
                                            self.test_transforms)
        
        test = CustomFileDataset([dataset.data_paths[idx] for idx in test_indices],
                                        [dataset.targets[idx] for idx in test_indices],
                                            self.test_transforms)
        print("train set length {}".format(len(train)))
        print("test set length {}".format(len(test)))
        # Here we set `pool_specifics`, where we set the transform attribute for the pool.
        active_set = ActiveLearningDataset(train)
        active_set.can_label = False

        active_set.label(initial_pool)
        return active_set, val, test

    def train(self):
        # random.seed(128)
        # torch.manual_seed(128)

        experiment = Experiment(
                api_key=comet_config["api_key"],
                project_name=comet_config["project_name"],
                workspace=comet_config["workspace"])
        use_cuda = torch.cuda.is_available()
        torch.backends.cudnn.benchmark = True
        if not use_cuda:
            print("No CUDA found.")


        active_set, val_set, test_set = self.get_datasets()


        heuristic = get_heuristic(self.cfg.heuristic)
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=(1-.5)*.5/(active_set.n_unlabelled + active_set.n_labelled))
        model = CustomModelWrapper(self.model, criterion, replicate_in_memory=False)

        model.add_metric(name='accuracy', initializer=lambda :  Accuracy(task='multiclass'))

        experiment.log_parameters(self.cfg)

        active_loop = ActiveLearningLoop(
                        active_set,
                        model.predict_on_dataset,
                        heuristic,
                        self.cfg.query_size,
                        batch_size=5,
                        iterations=self.cfg.iterations,
                        use_cuda=use_cuda,
                        max_sample=-1, # parameter to tweak device based
                        uncertainty_folder="src/active_learning_logs",
                        verbose=True,
                        )
        init_weights = deepcopy(model.state_dict())

        for epoch in tqdm(range(self.cfg.epoch)):
            model.load_state_dict(init_weights)
            self.weights_reset(model.model)

            hist, best_weigth = model.train_and_test_on_datasets(
                active_set,
                val_set,
                optimizer,
                batch_size=self.cfg.batch_size,
                epoch=100,
                use_cuda=use_cuda,
                workers = 4,
                return_best_weights=True,
                patience= 10,
                min_epoch_for_es=10*epoch,
                max_sample=-1,
            )
            experiment.log_metrics({"val_accuracy": model.get_metrics()["test_accuracy"]}, epoch=epoch)
            model.load_state_dict(best_weigth)
            # _ = model.train_on_dataset(
            #     active_set,
            #     optimizer,
            #     self.cfg.batch_size,
            #     self.cfg.learning_epoch + self.cfg.query_size,
            #     use_cuda,
            #     collate_fn=collate_fn_padd,
            # )

            # prediction = model.predict_on_dataset(dataset=test_set, 
            #                                         batch_size=self.cfg.batch_size, 
            #                                         iterations=1,
            #                                         use_cuda=use_cuda, 
            #                                         collate_fn=collate_fn_padd)
            # print(prediction.shape)
            # np.savetxt("preds_{}.csv".format(epoch), np.argmax(prediction.squeeze(-1), axis=1), delimiter=",")
            model.test_on_dataset(test_set, self.cfg.batch_size, use_cuda)
           
            # pprint(model.metrics["train_classification_report"].value)
            experiment.log_metrics({"train_loss": model.get_metrics()["train_loss"]}, epoch=epoch)
            experiment.log_metrics({"test_loss": model.get_metrics()["test_loss"]}, epoch=epoch)

            
            experiment.log_metrics({"test_accuracy": model.get_metrics()["test_accuracy"]}, epoch=epoch)
            experiment.log_metrics({"train_accuracy":model.get_metrics()["train_accuracy"]}, epoch=epoch)  
            
            experiment.log_metrics({"dataset_len": active_set.n_labelled}, epoch=epoch)
            should_continue = active_loop.step()
            # Keep track of progress
            # labelling_progress = active_set.labelled_map.astype(np.uint16)
            if not should_continue:
                break
        return 


if __name__=="__main__":
    seed_all(123)
    model = CustomVGG16(active_learning_mode=True)
    dataset = CustomDataset(data_path="data/NPY/volumes/",target_path="data/NPY/labels/")
    train_transform = transforms.Compose([#transforms.Resize((512//2, 512//2)),
                                    #             transforms.Normalize((0.5, ), (0.5, )),
                                    #     transforms.RandomHorizontalFlip(), 
                                    #     transforms.RandomVerticalFlip(),
                                    #     transforms.RandomAffine(degrees=(30, 70), translate=(0.1, 0.1), scale=(0.5, 0.75)), 
                                    # # transforms.ColorJitter(brightness=.5, hue=.3),
                                    ])

    test_transform = transforms.Compose([#transforms.Resize((512//2, 512//2)),
                                            #transforms.Normalize((0.5, ), (0.5, )),
                                        ])

    ActiveLearning = CustomActiveLearningTrainLoop(model=model, 
                                           dataset=dataset, 
                                           cfg=ExperimentConfig,
                                           train_transforms=train_transform,
                                           test_transforms=test_transform)
    ActiveLearning.train()

