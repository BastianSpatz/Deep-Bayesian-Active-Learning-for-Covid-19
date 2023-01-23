from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    epoch: int = 40
    batch_size: int = 4
    initial_pool: int = 500
    query_size: int = 20
    lr: float = 1e-4
    heuristic: str = "bald"
    iterations: int = 40
    learning_epoch: int = 20