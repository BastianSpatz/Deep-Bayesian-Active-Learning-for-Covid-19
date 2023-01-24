from dataclasses import dataclass

@dataclass
class ExperimentConfig:
    initial_pool: int = 500
    query_size: int = 20