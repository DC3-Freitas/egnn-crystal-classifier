from dataclasses import dataclass


@dataclass
class HParams:
    num_buckets: int = 24
    num_hidden: int = 64
    num_reg_layers: int = 2
    num_classes: int = 6

    dropout_prob: float = 0.05

    train_split_frac: float = 0.8
    train_eval_sample_frac: float = 0.25
    batch_size: int = 1024

    epochs: int = 100
    lr: float = 0.01
    label_smoothing: float = 0.1

    checkpoint_freq = 5
