"""Generate text from a trained checkpoint.

    python -m nanogpt_lotr.sample --ckpt checkpoints/default_best.pt \
        --prompt "GANDALF" --temperature 0.8 --top-k 200
"""

import argparse

import torch

from .config import Config
from .data import CharTokenizer
from .model import GPT


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = Config(**ckpt["config"])
    tok = CharTokenizer.from_state_dict(ckpt["tokenizer"])
    model = GPT(tok.vocab_size, cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tok, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", default="")
    p.add_argument("--max-new-tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, _ = load_model(args.ckpt, device)

    if args.prompt:
        idx = torch.tensor([tok.encode(args.prompt)], dtype=torch.long, device=device)
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)

    out = model.generate(idx, args.max_new_tokens,
                         temperature=args.temperature, top_k=args.top_k)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
