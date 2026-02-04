import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm


class GPTConfig:
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get("vocab_size", 8000)
        self.block_size = kwargs.get("block_size", 128)
        self.n_layers = kwargs.get("n_layers", 4)
        self.n_heads = kwargs.get("n_heads", 4)
        self.d_model = kwargs.get("d_model", 256)
        self.dropout = kwargs.get("dropout", 0.1)
        self.batch_size = kwargs.get("batch_size", 16)
        self.grad_accum_steps = kwargs.get("grad_accum_steps", 4)
        self.learning_rate = kwargs.get("learning_rate", 5e-4)
        self.max_steps = kwargs.get("max_steps", 20000)
        self.eval_interval = kwargs.get("eval_interval", 500)
        self.log_interval = kwargs.get("log_interval", 50)
        self.save_interval = kwargs.get("save_interval", 1000)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.proj = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        y = self.dropout(y)
        return y


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.block_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def load_config(config_path: Path):
    if not config_path.exists():
        return GPTConfig()
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return GPTConfig(**data)


def load_memmap(path: Path, dtype: np.dtype):
    return np.memmap(path, dtype=dtype, mode="r")


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1 : i + block_size + 1]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, config, device):
    model.eval()
    losses = {}
    for split, data in [("train", train_data), ("val", val_data)]:
        split_losses = []
        for _ in range(10):
            xb, yb = get_batch(data, config.block_size, config.batch_size, device)
            _, loss = model(xb, yb)
            split_losses.append(loss.item())
        losses[split] = sum(split_losses) / len(split_losses)
    model.train()
    return losses


def save_checkpoint(model, optimizer, step, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": model.config.__dict__,
    }
    torch.save(ckpt, output_dir / f"step_{step}.pt")
    torch.save(ckpt, output_dir / "latest.pt")


def move_optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.json"))
    parser.add_argument("--out_dir", default=str(Path(__file__).parent / "checkpoints"))
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    meta_path = data_dir / "meta.json"
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    dtype = np.dtype(meta["dtype"])
    train_data = load_memmap(data_dir / "train.bin", dtype)
    val_data = load_memmap(data_dir / "val.bin", dtype)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = out_dir / "latest.pt"
    start_step = 1
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu")
        config = GPTConfig(**ckpt["config"])
        model = GPT(config)
        model.load_state_dict(ckpt["model"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        print(f"Resuming from step {ckpt['step']}")
    else:
        config = load_config(Path(args.config))
        model = GPT(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    model.to(device)
    move_optimizer_to_device(optimizer, device)

    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    for step in range(start_step, config.max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(config.grad_accum_steps):
            xb, yb = get_batch(train_data, config.block_size, config.batch_size, device)
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                _, loss = model(xb, yb)
                loss = loss / config.grad_accum_steps
            scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        if step % config.log_interval == 0:
            print(f"step {step} | loss {loss.item():.4f}")

        if step % config.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, config, device)
            print(f"eval | train {losses['train']:.4f} | val {losses['val']:.4f}")

        if step % config.save_interval == 0:
            save_checkpoint(model, optimizer, step, out_dir)

    save_checkpoint(model, optimizer, config.max_steps, out_dir)


if __name__ == "__main__":
    main()
