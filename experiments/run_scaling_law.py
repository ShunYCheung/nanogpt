"""Train a sweep of model sizes; results land in results/runs.csv.

This is the spine of the writeup: vary capacity, hold data fixed, and watch
val loss flatten once the 424K-char corpus becomes the binding constraint.

    python experiments/run_scaling_law.py
    # then open experiments/analyze.ipynb to fit val_loss vs params
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanogpt_lotr.config import Config  # noqa: E402
from nanogpt_lotr.train import train  # noqa: E402

# (name, n_layer, n_embd, n_head) -- roughly 0.2M -> 11M params
SWEEP = [
    ("scale_xs", 2, 64, 4),
    ("scale_s", 4, 128, 4),
    ("scale_m", 6, 256, 8),
    ("scale_l", 8, 384, 8),
]


def main():
    for name, n_layer, n_embd, n_head in SWEEP:
        cfg = Config(name=name, n_layer=n_layer, n_embd=n_embd, n_head=n_head)
        print(f"\n=== {name}: L={n_layer} E={n_embd} H={n_head} ===")
        train(cfg)


if __name__ == "__main__":
    main()
