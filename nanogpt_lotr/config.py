from dataclasses import dataclass, asdict


@dataclass
class Config:
    # --- run ---
    name: str = "default"
    seed: int = 1337
    out_dir: str = "checkpoints"

    # --- data ---
    data_path: str = "data/processed/lotr.txt"
    train_split: float = 0.9

    # --- model ---
    block_size: int = 256
    n_embd: int = 256
    n_head: int = 8
    n_layer: int = 6
    dropout: float = 0.2

    # --- optimisation ---
    batch_size: int = 64
    max_iters: int = 5000
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 500
    lr_decay_iters: int = 5000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.99

    # --- eval / logging ---
    eval_interval: int = 500
    eval_iters: int = 200

    # --- sampling ---
    max_new_tokens: int = 500
    temperature: float = 0.8
    top_k: int = 200

    # --- experiment tracking (optional) ---
    use_wandb: bool = False
    wandb_project: str = "nanogpt-lotr"

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise KeyError(f"unknown config keys in {path}: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)
