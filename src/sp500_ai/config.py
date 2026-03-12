from dataclasses import dataclass


@dataclass
class TrainConfig:
    seq_len: int = 64
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-5
    epochs: int = 50
    hidden_dim: int = 128
    n_heads: int = 4
    n_layers: int = 3
    dropout: float = 0.2
    val_ratio: float = 0.2
    seed: int = 42
    grad_clip: float = 1.0
    early_stopping_patience: int = 0


REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
