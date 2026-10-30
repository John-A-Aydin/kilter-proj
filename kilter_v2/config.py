from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import tomllib


@dataclass
class DataConfig:
    source: str = "data/processed/climbs.jsonl"
    vocabulary: str = "data/processed/hold_vocabulary.csv"
    cache_dir: str = "data/experiments_v2"
    # Same UUID hash and seed as kilter/, so validation/test rows match the baseline.
    split_seed: int = 42
    train_fraction: float = 0.9
    validation_fraction: float = 0.05
    max_holds: int = 40
    include_computed_grades: bool = True
    # Drop training rows whose exact hold set also appears in validation or test.
    remove_split_leakage: bool = True


@dataclass
class ModelConfig:
    d_model: int = 128
    n_layers: int = 4
    n_heads: int = 4
    ff_multiplier: int = 4
    dropout: float = 0.1
    coordinate_frequencies: int = 6


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 8
    batch_size: int = 256
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.98
    warmup_fraction: float = 0.03
    min_lr_ratio: float = 0.05
    grad_clip: float = 1.0
    patience: int = 2
    device: str = "auto"
    threads: int = 4
    # "log" weights a row by log2(1 + ascents), capped at 6 (63 ascents); "none" weights rows equally.
    ascent_weighting: str = "none"
    # How each training row's given holds are chosen; resampled every epoch.
    p_scratch: float = 0.3
    p_starts: float = 0.45
    p_random: float = 0.25
    log_every: int = 100
    max_train_batches: int = 0
    max_eval_batches: int = 0


@dataclass
class Config:
    name: str = "climb_v2"
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
        if d.max_holds < 3 or min(m.n_layers, m.n_heads, m.ff_multiplier, m.coordinate_frequencies) < 1:
            raise ValueError("max_holds must be at least 3 and model dimensions positive")
        if m.d_model < 1 or m.d_model % m.n_heads:
            raise ValueError("d_model must be a positive multiple of n_heads")
        if not 0 <= m.dropout < 1:
            raise ValueError("Invalid dropout")
        if min(t.epochs, t.batch_size, t.threads, t.log_every, t.patience) < 1:
            raise ValueError("Epochs, batch size, threads, log interval, and patience must be positive")
        if t.learning_rate <= 0 or t.weight_decay < 0 or t.grad_clip <= 0:
            raise ValueError("Invalid optimizer settings")
        if not (0 <= t.warmup_fraction < 1 and 0 <= t.min_lr_ratio <= 1 and 0 <= t.beta1 < 1 and 0 <= t.beta2 < 1):
            raise ValueError("Invalid scheduler or Adam settings")
        if t.ascent_weighting not in {"none", "log"}:
            raise ValueError("ascent_weighting must be 'none' or 'log'")
        if min(t.p_scratch, t.p_starts, t.p_random) < 0 or abs(t.p_scratch + t.p_starts + t.p_random - 1) > 1e-6:
            raise ValueError("p_scratch, p_starts, and p_random must be nonnegative and sum to 1")
        if min(t.max_train_batches, t.max_eval_batches) < 0:
            raise ValueError("Batch limits must be nonnegative")
        if not all(isinstance(v, bool) for v in (d.include_computed_grades, d.remove_split_leakage)):
            raise ValueError("Boolean options must be true or false")
        if not self.name or Path(self.name).name != self.name or self.name in {".", ".."}:
            raise ValueError("name must be a single directory name")


def load_config(path, overrides=()):
    with Path(path).open("rb") as source:
        config = Config.from_dict(tomllib.load(source)).to_dict()
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
