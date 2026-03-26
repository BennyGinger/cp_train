from dataclasses import dataclass


@dataclass
class TrainSettings:
    pretrained_model: str | None = "cyto3"
    model_name: str | None = None
    n_epochs: int = 300
    weight_decay: float = 1e-4
    learning_rate: float = 0.1
    SGD: bool = True