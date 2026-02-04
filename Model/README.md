# Small From-Scratch LLM (Local)

This folder contains a **small from-scratch language model** pipeline that
trains on *all* provided datasets under `Dataset/`. It is designed to run on
a single GPU (RTX 3050 4GB) by using a tiny Transformer and mixed precision.

## What This Builds
- A small GPT-style model trained from scratch
- A tokenizer trained on your training text
- Checkpoints and a simple sampling script

## Setup
1. Create a virtual environment (recommended)
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Prepare Data
This will:
- Read all CSVs and TXT files from `Dataset/`
- Build `Model/data/train.txt`, `val.txt`, `test.txt`
- Train a tokenizer
- Encode splits to token `.bin` files

```bash
python prepare_data.py --dataset_dir "C:\Users\MY ASUS\Downloads\Project1\Dataset"
```

## Train
```bash
python train.py --data_dir "C:\Users\MY ASUS\Downloads\Project1\Model\data"
```

Training will auto-resume from `Model/checkpoints/latest.pt` if it exists.

## Sample
```bash
python sample.py --data_dir "C:\Users\MY ASUS\Downloads\Project1\Model\data" --checkpoint "C:\Users\MY ASUS\Downloads\Project1\Model\checkpoints\latest.pt"
```

## Notes
- This is a **small model** (not ChatGPT-scale).
- Training from scratch on a single GPU will take time.
- You can adjust model size in `config.json`.
