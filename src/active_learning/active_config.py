from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    epoch: int = 200
    batch_size: int = 8
    initial_pool: int = 60
    query_size: int = 20
    lr: float = 0.0001
    heuristic: str = "random"
    iterations: int = 1 if heuristic == "random" else 40
    learning_epoch: int = 5
    patience: int = 15
    min_epoch_for_es: int = 0
