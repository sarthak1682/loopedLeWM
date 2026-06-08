# LoopLeWM — Looped Transformer Predictor for LeWM

A drop-in **looped (weight-tied) transformer predictor** for
[LeWorldModel (LeWM)](https://github.com/lucas-maes/le-wm). The standard
`L`-layer next-state predictor is replaced by a *single* transformer block applied
`K` times with input injection and a per-loop timestep embedding (Yang et al.,
[*Looped Transformers are Better at Learning Learning Algorithms*](https://arxiv.org/abs/2311.12424),
ICLR 2024). Everything else in LeWM — encoder, SIGReg loss, training loop, planner
— is unchanged.

> This is a fork. For the base model, install, data pipeline, planning/eval, and
> pretrained checkpoints, see the upstream
> [LeWM README](https://github.com/lucas-maes/le-wm#readme). This document covers
> only what this fork adds.

## What's added

- `module.py` — `LoopedPredictor` / `LoopedTransformer` (one weight-tied
  `ConditionalBlock` looped `K` times, `h₀=0; h ← Block(h + x + τ_k, c)`) and a
  `build_predictor` factory. Identical `forward(x, c) → (B, T, D)` interface to the
  original `ARPredictor`, so nothing downstream changes.
- `config/train/lewm.yaml` — two flags: `predictor_type` (`standard` | `looped`)
  and `num_loops` (`K`, default `6`).
- `sanity_check.py` — offline checks (no training stack required).

## Usage

Install and prepare data per the upstream
[LeWM README](https://github.com/lucas-maes/le-wm#readme), then select the
predictor at launch:

```bash
python train.py data=pusht predictor_type=looped num_loops=6   # looped (K=6)
python train.py data=pusht predictor_type=standard             # baseline
```

The predictor's parameter count is logged at startup, so the standard-vs-looped
reduction is visible (≈6× fewer parameters at `K=6`).

### PushT data from Hugging Face

`quentinll/lewm-pusht` is published as **both** a model repo and a dataset repo.
The training data is in the dataset repo, and the existing
`config/train/data/pusht.yaml` already references it — fetch and decompress into
`$STABLEWM_HOME` (no data-pipeline changes needed):

```bash
hf download datasets/quentinll/lewm-pusht pusht_expert_train.h5.zst \
    --repo-type dataset --local-dir $STABLEWM_HOME
tar --zstd -xvf $STABLEWM_HOME/pusht_expert_train.h5.zst -C $STABLEWM_HOME
```

## Sanity checks

Runs with only `torch` (no `stable_worldmodel`):

```bash
python sanity_check.py
```

- **Parameter count** — standard (`depth=6`) vs looped (`K=6`); ≈6× reduction.
- **K=1 equivalence** — a `K=1` looped predictor is bit-identical to a single
  standard block (asserted with matched weights).

For loss-curve behavior, train a `looped` and a `standard` run and compare
`train/sigreg_loss` / `train/pred_loss` in WandB.

## Citation

This builds directly on LeWM and the looped-transformer formulation — please cite
both (see the [upstream README](https://github.com/lucas-maes/le-wm#readme) for the
LeWM BibTeX).
