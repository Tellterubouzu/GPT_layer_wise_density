# BI-Floor (method3)

This directory contains a minimal, runnable research codebase to train and evaluate the
"BI-Floor" regularizer described in `DOCS/method3_Floor_normalization.md`.

## Structure

- `configs/`: example JSON configs for training and evaluation.
- `scripts/`: entrypoints for training, evaluation, and plotting.
- `src/`: core implementation (losses, training loop, eval metrics, plotting helpers).
- `tests/`: small unit tests for BI-Floor.

## Quickstart

1) Train a small model with BI-Floor:

```bash
python scripts/train.py --config configs/train_small.json
```

Baseline (no BI-Floor):

```bash
python scripts/train.py --config configs/train_small_baseline.json
```

2) Run evaluation metrics (PPL, BI, layer drop, representation similarity):

```bash
python scripts/eval.py --config configs/eval_small.json --run_dir runs/exp_small_bifloor
```

3) Plot results (baseline vs BI-Floor):

```bash
python scripts/plot_results.py \
  --baseline_dir runs/exp_small_baseline \
  --bifloor_dir runs/exp_small_bifloor \
  --out_dir runs/plots
```

## Notes

- The training loop uses `output_hidden_states=True` to compute BI on the fly.
- Layer skipping uses forward hooks; it works for GPT-2/GPT-NeoX/LLaMA-style blocks.
- Example configs use tiny splits for sanity checks; scale up by adjusting splits and steps.
