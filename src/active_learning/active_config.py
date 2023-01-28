from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    epoch: int = 50
    batch_size: int = 3
    initial_pool: int = 10
    query_size: int = 1
    lr: float = 9e-4
    heuristic: str = "bald"
    iterations: int = 15
    learning_epoch: int = 20