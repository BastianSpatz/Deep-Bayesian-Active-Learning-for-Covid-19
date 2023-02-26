from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    epoch: int = 50
    batch_size: int = 8
    initial_pool: int = 33
    query_size: int = 9
    lr: float = 0.001
    heuristic: str = "random"
    iterations: int = 50
    learning_epoch: int = 30
