"""Config-driven training.

    python -m nanogpt_lotr.train --config config/default.yaml
    python -m nanogpt_lotr.train --config config/default.yaml n_layer=4 max_iters=2000

Each run:
  * saves a best + final checkpoint (model + tokenizer + config) to out_dir/
  * appends one summary row to results/runs.csv          (cross-run comparison)
  * writes the eval trajectory to results/history/<name>.csv  (loss curves)
"""

import os
import csv
import math
import time
import argparse

import torch

from .config import Config
from .data import load_data, get_batch
from .model import GPT


def get_lr(it, cfg):
    if it < cfg.warmup_iters:
        return cfg.learning_rate * (it + 1) / cfg.warmup_iters
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    ratio = (it - cfg.warmup_iters) / max(cfg.lr_decay_iters - cfg.warmup_iters, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def _autocast(device):
    dev_type = "cuda" if device.startswith("cuda") else "cpu"
    return torch.amp.autocast(device_type=dev_type, enabled=device.startswith("cuda"))


@torch.no_grad()
def estimate_loss(model, train_data, val_data, cfg, device):
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            X, Y = get_batch(data, cfg.block_size, cfg.batch_size, device)
            with _autocast(device):
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(model, tok, cfg, it, losses, tag):
    os.makedirs(cfg.out_dir, exist_ok=True)
    path = os.path.join(cfg.out_dir, f"{cfg.name}_{tag}.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "tokenizer": tok.state_dict(),
            "config": cfg.to_dict(),
            "iter": it,
            "losses": losses,
        },
        path,
    )
    return path


def log_run(cfg, result, n_params, wall):
    os.makedirs("results", exist_ok=True)
    path = "results/runs.csv"
    row = {
        **cfg.to_dict(),
        "n_params": n_params,
        "final_train_loss": round(result["train"], 4),
        "final_val_loss": round(result["val"], 4),
        "wall_clock_s": round(wall, 1),
    }
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def save_history(cfg, history):
    os.makedirs("results/history", exist_ok=True)
    path = f"results/history/{cfg.name}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["iter", "train_loss", "val_loss", "lr"])
        w.writeheader()
        w.writerows(history)


def train(cfg: Config):
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  run={cfg.name}")

    train_data, val_data, tok = load_data(cfg)
    model = GPT(tok.vocab_size, cfg).to(device)
    n_params = model.num_params()
    print(f"{n_params / 1e6:.3f}M parameters | vocab={tok.vocab_size}")

    optimizer = model.configure_optimizers(
        cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2)
    )
    scaler = torch.amp.GradScaler(enabled=device.startswith("cuda"))

    wandb = None
    if cfg.use_wandb:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project=cfg.wandb_project, name=cfg.name, config=cfg.to_dict())

    history = []
    best_val = float("inf")
    last = {"train": float("nan"), "val": float("nan")}
    t0 = time.time()

    for it in range(cfg.max_iters):
        lr = get_lr(it, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        if it % cfg.eval_interval == 0 or it == cfg.max_iters - 1:
            last = estimate_loss(model, train_data, val_data, cfg, device)
            gap = last["train"] - last["val"]
            print(f"step {it:5d} | train {last['train']:.4f} | val {last['val']:.4f} "
                  f"| gap {gap:+.4f} | lr {lr:.2e}")
            history.append({"iter": it, "train_loss": round(last["train"], 4),
                            "val_loss": round(last["val"], 4), "lr": lr})
            if wandb:
                wandb.log({"train_loss": last["train"], "val_loss": last["val"],
                           "lr": lr, "iter": it})
            if last["val"] < best_val:
                best_val = last["val"]
                save_checkpoint(model, tok, cfg, it, last, tag="best")

        xb, yb = get_batch(train_data, cfg.block_size, cfg.batch_size, device)
        with _autocast(device):
            _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

    wall = time.time() - t0
    save_checkpoint(model, tok, cfg, cfg.max_iters - 1, last, tag="final")
    save_history(cfg, history)
    log_run(cfg, last, n_params, wall)
    print(f"done in {wall:.1f}s | best val {best_val:.4f}")
    if wandb:
        wandb.finish()
    return last


def apply_overrides(cfg: Config, overrides):
    for ov in overrides:
        key, sep, val = ov.partition("=")
        if not sep or not hasattr(cfg, key):
            raise KeyError(f"bad override (expected key=value with a known key): {ov!r}")
        cur = getattr(cfg, key)
        if isinstance(cur, bool):
            val = val.lower() in ("1", "true", "yes")
        elif isinstance(cur, int):
            val = int(val)
        elif isinstance(cur, float):
            val = float(val)
        setattr(cfg, key, val)
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None,
                   help="path to a YAML config; omit to use defaults")
    p.add_argument("overrides", nargs="*", help="key=value overrides, e.g. n_layer=4")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    cfg = apply_overrides(cfg, args.overrides)
    train(cfg)


if __name__ == "__main__":
    main()
