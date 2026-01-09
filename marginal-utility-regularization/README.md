# Marginal Utility Regularization (method4)

This directory contains a minimal, runnable research codebase to train and evaluate the
"Marginal Utility Regularization (MUR)" method described in
`DOCS/method4_marginal_utility_regularization.md`.

## Structure

- `configs/`: example JSON configs for training and evaluation.
- `scripts/`: entrypoints for training, evaluation, and plotting.
- `src/`: core implementation (MUR utilities, training loop, eval metrics, plotting helpers).
- `tests/`: small unit tests for utility math and indexing.

## Quickstart

1) Train a small model (MUR loss mode):

```bash
python scripts/train.py --config configs/train_small_mur_loss.json
```

Baseline (no MUR):

```bash
python scripts/train.py --config configs/train_small_baseline.json
```

2) Run evaluation metrics (PPL, MUR stats, BI, layer drop):

```bash
python scripts/eval.py --config configs/eval_small.json --run_dir runs/exp_small_mur_loss
```

3) Plot results (baseline vs MUR):

```bash
python scripts/plot_results.py \
  --baseline_dir runs/exp_small_baseline \
  --mur_dir runs/exp_small_mur_loss \
  --out_dir runs/plots
```

## Notes

- `mur.mode=loss` uses `autograd.grad` to compute the per-layer gradient for the MUR loss.
- `mur.mode=update` uses a single backward pass and scales layer gradients before `optimizer.step`.
- Example configs use tiny splits for sanity checks; scale up by adjusting splits and steps.
