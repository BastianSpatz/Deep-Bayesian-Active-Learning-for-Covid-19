from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    epoch: int = 10
    batch_size: int = 8
    initial_pool: int = 33
    query_size: int = 33
    lr: float = 0.0005
    heuristic: str = "bald"
    iterations: int = 50
    learning_epoch: int = 30
