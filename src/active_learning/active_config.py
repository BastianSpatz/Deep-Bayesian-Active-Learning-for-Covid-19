from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    epoch: int = 20
    batch_size: int = 12
    initial_pool: int = 90
    query_size: int = 60
    lr: float = 0.005
    heuristic: str = "bald"
    iterations: int = 50
    learning_epoch: int = 1
    patience: int = 20
    min_epoch_for_es: int = 10
