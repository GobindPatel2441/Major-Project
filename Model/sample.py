import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from train import GPT, GPTConfig


@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=40):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.config.block_size :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return idx


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    tokenizer = Tokenizer.from_file(str(data_dir / "tokenizer.json"))

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = GPTConfig(**ckpt["config"])
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    ids = tokenizer.encode(args.prompt).ids
    if not ids:
        ids = [0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = generate(model, idx, args.max_new_tokens)
    text = tokenizer.decode(out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
