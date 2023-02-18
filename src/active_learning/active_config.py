from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    epoch: int = 50
    batch_size: int = 8
    initial_pool: int = 9
    query_size: int = 9
    lr: float = 5e-5
    heuristic: str = "bald"
    iterations: int = 10
    learning_epoch: int = 5