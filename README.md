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

## Methodology

### Motivation: tension between PCC and CDD

In this task, **mutation count** and **functional impact** are not always aligned.
A single-nucleotide variant (SNV) can be highly disruptive (large embedding shift),
while multiple substitutions can be benign (small shift).

This creates a practical tension between objectives:
- **PCC** encourages distances to be *monotonic* with respect to mutation magnitude (e.g., `n_mut`) within a genomic window/group.
- **CDD** can reward separating clinically meaningful variants even when `n_mut` is small, which may break strict monotonicity.

We therefore adopt a **dual-head projection** and a **staged metric-learning style** objective:
one branch focuses on stable global scaling (CD/CDD), while the other enforces within-group ranking behavior for PCC,
reducing gradient interference between partially competing signals.

This repo uses a frozen **dna2vec** backbone and trains a lightweight **dual-head projection** to shape the embedding geometry for **CD / CDD / PCC**.

### Backbone embedding

Given a DNA sequence $x$, dna2vec produces token-level hidden states. We mean-pool over the sequence length and apply L2-normalization:



$$ h(x) = \mathrm{meanpool}(\mathrm{dna2vec}(x)), \qquad \bar{h}(x) = \frac{h(x)}{\lVert h(x)\rVert_2}. $$



In the best setup, the backbone is **frozen** and only the projection head is trained.

### Dual-head projection (best config)

We use two MLP branches (CD/CDD branch + PCC branch). Each branch output is L2-normalized, then concatenated and normalized again:



$$ u_{\mathrm{cd}} = f_{\mathrm{cd}}(\bar{h}), \quad u_{\mathrm{pcc}} = f_{\mathrm{pcc}}(\bar{h}) $$




$$ \hat{u}_{\mathrm{cd}} = \frac{u_{\mathrm{cd}}}{\lVert u_{\mathrm{cd}}\rVert_2}, \quad \hat{u}_{\mathrm{pcc}} = \frac{u_{\mathrm{pcc}}}{\lVert u_{\mathrm{pcc}}\rVert_2} $$




$$ z = \mathrm{concat}(\alpha \hat{u}_{\mathrm{cd}}, \beta \hat{u}_{\mathrm{pcc}}), \qquad \bar{z} = \frac{z}{\lVert z\rVert_2}. $$



- Learnable scalars: $\alpha, \beta$ (initialized to 1.0)
- Default dims (best): `cd_dim=768`, `pcc_dim=256` → `out_dim=1024`
- CD/CDD head: Linear($H\to 2H$) → LayerNorm → GELU → Dropout(0.2) → Linear($2H\to 768$)
- PCC head: Linear($H\to H$) → GELU → Linear($H\to 256$)

### Distance definition

All training/eval objectives use **cosine distance** on normalized embeddings:



$$ d(a,b) = 1 - \cos(a,b). $$



### Training objective (projection-head-only stage)

For each (ref, var) pair with mutation count $n_{\text{mut}}$, the total loss is:



$$ \mathcal{L} = \mathcal{L}_{\mathrm{reg}} + \lambda_{\mathrm{pcc}} \mathcal{L}_{\mathrm{pcc}} + \lambda_{\mathrm{cd}} \mathcal{L}_{\mathrm{scale}}. $$



Default weights (as in the training script):
- $\lambda_{\mathrm{pcc}} = 0.02$
- $\lambda_{\mathrm{cd}} = 0.25$

#### 1) Distance regression (mutation magnitude)

The main signal regresses cosine distance to a normalized mutation target:



$$ \mathcal{L}_{\mathrm{reg}} = \left(d(\bar{z}(r), \bar{z}(v)) - \tilde{n}\right)^2,\quad \tilde{n}=\min\left(\frac{n_{\text{mut}}}{\texttt{max\\_n\\_mut}}, 1\right). $$



Defaults: `max_n_mut=20.0` (target is clamped to $[0,1]$).

#### 2) PCC ranking loss (within-group monotonicity)

For each `group_id`, we sort samples by $n_{\text{mut}}$ and apply a hinge loss over adjacent pairs only (for efficiency). Let $d_i = d(\hat{u}_{\mathrm{pcc}}(r_i), \hat{u}_{\mathrm{pcc}}(v_i))$. For adjacent items with $\Delta n = n_{i+1} - n_i > 0$ and $\Delta d = d_{i+1} - d_i$:



$$ \mathcal{L}_{\mathrm{pcc}} = \mathrm{mean}\;\max(0, m(\Delta n) - \Delta d), \quad m(\Delta n) = \texttt{margin\\_scale}\cdot \Delta n. $$



Default: `margin_scale=1e-3`.

We denote this as $\gamma$ in the equation above.


#### 3) CD scale anchoring (global distance scale)

We keep the mean distance near a target CD:



$$ \mathcal{L}_{\mathrm{scale}} = \left(\mathbb{E}\left[d(\bar{z}(r),\bar{z}(v))\right] - \mu_{\text{target}}\right)^2. $$



Default: `target_cd=0.13`.

### Inference modes

- `--mode z` (best): output $\bar{z}$ (dual-head only, 1024 dims by default).
- `--mode concat`: output $\mathrm{L2Norm}([\bar{h} \,\|\, \bar{z}])$.
  If the concatenated dimension exceeds 2048, it is **truncated to 2048 before L2-normalization**.

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