"""Character-level tokenizer and batching.

The tokenizer is saved inside every checkpoint so that sampling reconstructs the
exact same vocab without needing the original corpus.
"""

import torch


class CharTokenizer:
    def __init__(self, text: str):
        self.chars = sorted(list(set(text)))
        self._build()

    def _build(self):
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def state_dict(self):
        return {"chars": self.chars}

    @classmethod
    def from_state_dict(cls, sd):
        obj = cls.__new__(cls)
        obj.chars = list(sd["chars"])
        obj._build()
        return obj


def load_data(cfg, tokenizer: "CharTokenizer | None" = None):
    """Return (train_data, val_data, tokenizer) as long tensors."""
    with open(cfg.data_path, "r", encoding="utf-8") as f:
        text = f.read()
    if tokenizer is None:
        tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(cfg.train_split * len(data))
    return data[:n], data[n:], tokenizer


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    if device.startswith("cuda"):
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
