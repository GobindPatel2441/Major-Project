import argparse
import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm


CSV_TEXT_COLUMNS = ["text", "content", "sentence", "review", "message"]


def iter_csv_texts(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return

        # Handle leading empty column
        if header and header[0] == "":
            header = header[1:]
            offset = 1
        else:
            offset = 0

        header_lc = [h.strip().lower() for h in header]
        text_idx = None
        for col in CSV_TEXT_COLUMNS:
            if col in header_lc:
                text_idx = header_lc.index(col)
                break

        # Fallback to first non-empty column if no known text column
        if text_idx is None:
            for i, name in enumerate(header_lc):
                if name:
                    text_idx = i
                    break

        if text_idx is None:
            return

        for row in reader:
            if not row:
                continue
            if offset:
                row = row[offset:]
            if text_idx >= len(row):
                continue
            text = row[text_idx].strip()
            if text:
                yield text


def iter_txt_lines(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def write_corpus(output_path: Path, csv_paths, txt_paths):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        for csv_path in csv_paths:
            for text in iter_csv_texts(csv_path):
                out.write(text.replace("\n", " ") + "\n")
        for txt_path in txt_paths:
            for text in iter_txt_lines(txt_path):
                out.write(text.replace("\n", " ") + "\n")


def train_tokenizer(corpus_path: Path, tokenizer_path: Path, vocab_size: int):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = ByteLevel()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
        show_progress=True,
    )
    tokenizer.train([str(corpus_path)], trainer)
    tokenizer.save(str(tokenizer_path))
    return tokenizer


def encode_to_memmap(tokenizer: Tokenizer, input_path: Path, output_path: Path):
    # First pass: count tokens
    total = 0
    for line in tqdm(iter_txt_lines(input_path), desc=f"Counting {input_path.name}"):
        total += len(tokenizer.encode(line).ids)

    vocab_size = tokenizer.get_vocab_size()
    dtype = np.uint16 if vocab_size <= 65535 else np.uint32
    memmap = np.memmap(output_path, dtype=dtype, mode="w+", shape=(total,))

    idx = 0
    for line in tqdm(iter_txt_lines(input_path), desc=f"Encoding {input_path.name}"):
        ids = tokenizer.encode(line).ids
        if not ids:
            continue
        memmap[idx: idx + len(ids)] = ids
        idx += len(ids)

    memmap.flush()
    return dtype, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--output_dir", default=str(Path(__file__).parent / "data"))
    parser.add_argument("--vocab_size", type=int, default=8000)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(dataset_dir.glob("*.csv"))
    txt_paths = sorted(dataset_dir.glob("*.txt"))

    train_txt = output_dir / "train.txt"
    val_txt = output_dir / "val.txt"
    test_txt = output_dir / "test.txt"

    # Use provided val/test if present; otherwise split later
    src_val = dataset_dir / "val.txt"
    src_test = dataset_dir / "test.txt"
    src_train = dataset_dir / "train.txt"

    val_lines = []
    test_lines = []
    if src_val.exists():
        val_lines = list(iter_txt_lines(src_val))
        val_txt.write_text("\n".join(val_lines), encoding="utf-8")
    if src_test.exists():
        test_lines = list(iter_txt_lines(src_test))
        test_txt.write_text("\n".join(test_lines), encoding="utf-8")

    # Build training corpus from CSVs + train.txt if present
    train_corpus_lines = []
    for csv_path in csv_paths:
        train_corpus_lines.extend(list(iter_csv_texts(csv_path)))
    if src_train.exists():
        train_corpus_lines.extend(list(iter_txt_lines(src_train)))

    # If no provided val/test, create a small split
    if not val_lines or not test_lines:
        rng = np.random.default_rng(42)
        rng.shuffle(train_corpus_lines)
        n = len(train_corpus_lines)
        n_val = max(1, int(n * 0.01))
        n_test = max(1, int(n * 0.01))
        val_lines = train_corpus_lines[:n_val]
        test_lines = train_corpus_lines[n_val: n_val + n_test]
        train_corpus_lines = train_corpus_lines[n_val + n_test:]
        val_txt.write_text("\n".join(val_lines), encoding="utf-8")
        test_txt.write_text("\n".join(test_lines), encoding="utf-8")

    train_txt.write_text("\n".join(train_corpus_lines), encoding="utf-8")

    # Train tokenizer on train.txt
    tokenizer_path = output_dir / "tokenizer.json"
    tokenizer = train_tokenizer(train_txt, tokenizer_path, args.vocab_size)

    # Encode splits
    dtype_train, train_tokens = encode_to_memmap(tokenizer, train_txt, output_dir / "train.bin")
    dtype_val, val_tokens = encode_to_memmap(tokenizer, val_txt, output_dir / "val.bin")
    dtype_test, test_tokens = encode_to_memmap(tokenizer, test_txt, output_dir / "test.bin")

    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "train_tokens": int(train_tokens),
        "val_tokens": int(val_tokens),
        "test_tokens": int(test_tokens),
        "dtype": str(np.dtype(dtype_train)),
        "tokenizer": str(tokenizer_path),
    }
    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Data prep complete.")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
