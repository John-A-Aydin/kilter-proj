from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import tomllib


@dataclass
class DataConfig:
    source: str = "data/processed/climbs.jsonl"
    vocabulary: str = "data/processed/hold_vocabulary.csv"
    cache_dir: str = "data/experiments"
    split_seed: int = 42
    train_fraction: float = 0.9
    validation_fraction: float = 0.05
    max_seq_len: int = 128
    include_computed_grades: bool = True


@dataclass
class ModelConfig:
    d_model: int | str = "auto"
    n_layers: int = 2
    n_heads: int = 4
    ff_multiplier: int = 4
    dropout: float = 0.1
    positional_embeddings: bool = True
    parameters_per_example: float = 0.5


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 10
    batch_size: int = 256
    learning_rate: float = 0.0003
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    warmup_fraction: float = 0.05
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    patience: int = 3
    device: str = "auto"
    threads: int = 4
    num_workers: int = 0
    length_bucketing: bool = True
    log_every: int = 100
    # Optional bounded runs; zero means the entire split. Never used for sizing.
    max_train_batches: int = 0
    max_eval_batches: int = 0


@dataclass
class Config:
    name: str = "ar_transformer"
    output_dir: str = "runs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        for key, typ in (("data", DataConfig), ("model", ModelConfig), ("train", TrainConfig)):
            values[key] = typ(**values.get(key, {}))
        config = cls(**values)
        config.validate()
        return config

    def validate(self):
        d, m, t = self.data, self.model, self.train
        if not (0 < d.train_fraction < 1 and 0 < d.validation_fraction < 1 - d.train_fraction):
            raise ValueError("Split fractions must leave nonempty train, validation, and test fractions")
        if d.max_seq_len < 2 or m.n_layers < 1 or m.n_heads < 1 or m.ff_multiplier < 1:
            raise ValueError("Sequence length and model dimensions must be positive")
        if m.d_model != "auto" and (not isinstance(m.d_model, int) or m.d_model < 1 or m.d_model % m.n_heads):
            raise ValueError("d_model must be 'auto' or a positive multiple of n_heads")
        if not 0 <= m.dropout < 1 or m.parameters_per_example <= 0:
            raise ValueError("Invalid dropout or parameter target")
        if min(t.epochs, t.batch_size, t.threads, t.log_every, t.patience) < 1:
            raise ValueError("Epochs, batch size, threads, log interval, and patience must be positive")
        if t.learning_rate <= 0 or t.weight_decay < 0 or t.grad_clip <= 0:
            raise ValueError("Invalid optimizer settings")
        if not (0 <= t.warmup_fraction < 1 and 0 <= t.min_lr_ratio <= 1):
            raise ValueError("Invalid scheduler settings")
        if not (0 <= t.beta1 < 1 and 0 <= t.beta2 < 1):
            raise ValueError("Adam betas must be in [0, 1)")
        if min(t.max_train_batches, t.max_eval_batches, t.num_workers) < 0:
            raise ValueError("Batch limits and workers must be nonnegative")
        if not all(isinstance(value, bool) for value in (m.positional_embeddings, d.include_computed_grades, t.length_bucketing)):
            raise ValueError("Boolean options must be true or false")
        if not self.name or Path(self.name).name != self.name or self.name in {".", ".."}:
            raise ValueError("name must be a single directory name")


def load_config(path, overrides=()):
    with Path(path).open("rb") as source:
        raw = tomllib.load(source)
    config = Config.from_dict(raw).to_dict()
    for override in overrides:
        key, separator, value = override.partition("=")
        if not separator:
            raise ValueError("Overrides must have the form section.key=value")
        try:
            parsed = tomllib.loads("value = " + value)["value"]
        except tomllib.TOMLDecodeError:
            parsed = value
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f"Unknown config key: {key}")
        target[parts[-1]] = parsed
    return Config.from_dict(config)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)
