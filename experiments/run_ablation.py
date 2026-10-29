"""Regularisation ablation: how dropout and weight decay move the train/val gap.

Holds the 6-layer/256-wide model fixed and sweeps regularisation strength.

    python experiments/run_ablation.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanogpt_lotr.config import Config 
from nanogpt_lotr.train import train

DROPOUTS = [0.0, 0.1, 0.2, 0.3]
WEIGHT_DECAYS = [0.0, 0.1]


def main():
    for wd in WEIGHT_DECAYS:
        for do in DROPOUTS:
            name = f"abl_do{do}_wd{wd}"
            cfg = Config(name=name, dropout=do, weight_decay=wd)
            print(f"\n=== {name} ===")
            train(cfg)


if __name__ == "__main__":
    main()
