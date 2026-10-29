# nanoGPT-LotR

This is a character-level GPT trained from scratch on the Lord of the Rings screenplay.
I made this to learn how GPT works. This includes stuff like transformers, 
multi-head self attention and automatic mixed precision (AMP). This is not meant to be a working chatGPT demo.

This model and code is based on Andrej Karpathy's Youtube tutorial [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY).
This is a great tutorial to learn the basics of GPT and how to build a small GPT yourself.
This code is a modified version and is trained on the Lord of the Rings instead of Shakespeare. 

I wrote a [blog](https://shunycheung.github.io/posts/2026/06/chatGPT_make_another_LOTR_movie/) on the results of this project.

---

## Quickstart

```bash
pip install -e .            # or: pip install -r requirements.txt

python data/prepare.py

# train the baseline
python -m nanogpt_lotr.train --config config/default.yaml

# generate
python -m nanogpt_lotr.sample --ckpt checkpoints/default_best.pt \
    --prompt "GANDALF" --temperature 0.8 --top-k 200
```

Config values can be overridden inline without editing files:

```bash
python -m nanogpt_lotr.train --config config/default.yaml n_layer=4 max_iters=2000
```

---

## Reproducing the study

```bash
python experiments/run_scaling_law.py   # ~0.2M -> ~11M params, corpus fixed
python experiments/run_ablation.py      # dropout x weight-decay grid
# then open experiments/analyze.ipynb to regenerate all figures
```

Results accumulate in `results/runs.csv` (one row per run) and
`results/history/<name>.csv` (the eval trajectory per run).

---

## Repo layout

```
nanogpt_lotr/        core package
  config.py          loads settings from YAML file
  data.py            loads and prepares data. Batches data.
  model.py           GPT: heads, blocks, optimizer grouping, top-k sampling
  train.py           training loop, cosine LR, checkpointing, run ledger
  sample.py          generate from a checkpoint
config/              YAML configs; experiments/ holds one file per run
experiments/         run_scaling_law.py, run_ablation.py, analyze.ipynb
data/prepare.py      cleans a raw corpus -> data/processed/lotr.txt
results/             runs.csv (ledger) + history/ (curves) + figures/
notebooks/           test.ipynb (the original notebook with my first attempt)
```

---

## License

MIT — see `LICENSE`.
