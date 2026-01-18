# glm-dna2vec-dualhead

Mutation-sensitive DNA embeddings using **dna2vec** with a **staged metric-learning**-style dual-head projection to optimize **CD / CDD / PCC**.

This repo is intentionally kept minimal and **single-backbone (dna2vec) focused**.

## Directory layout

```
.
├─ src/glm_dna2vec/          # reusable modules (backbone, head, losses, datasets)
└─ scripts/                  # runnable entrypoints
   ├─ train_dualhead.py
   ├─ eval_dualhead.py
   └─ infer_submission.py
```

## Setup

Install deps (example):

```bash
pip install torch transformers numpy pandas
```

> The scripts add `./src` to `PYTHONPATH` automatically, so you can run them from the repo root without packaging.

## Train

```bash
python scripts/train_dualhead.py \
  --train_path data/processed/clinvar_pairs_v3_train.jsonl \
  --out_dir runs/dna2vec_dualhead_best \
  --epochs 3 \
  --fp16
```

## Evaluate (CD / CDD / PCC)

```bash
python scripts/eval_dualhead.py \
  --data_path data/processed/clinvar_pairs_v3_dev_eval.jsonl \
  --head_ckpt runs/dna2vec_dualhead_best/head_epoch3.pt
```

## Infer submission

```bash
python scripts/infer_submission.py \
  --test_path data/raw/test.csv \
  --head_ckpt runs/dna2vec_dualhead_best/head_epoch3.pt \
  --output_path submissions/submission.csv \
  --mode z
```

- `--mode z` (best): uses the dual-head output `z` only.
- `--mode concat`: uses `[base || z]` (then L2-normalize). If dims exceed 2048, it truncates.

