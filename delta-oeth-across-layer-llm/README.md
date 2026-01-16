# Delta-orth Across Layers (method2)

This directory contains a minimal, runnable research codebase to train and evaluate the
"delta-orth across layers" regularizer described in `DOCS/method2_delta_oeth_across_layers.md`.

## Structure

- `configs/`: example JSON configs for training and evaluation.
- `scripts/`: entrypoints for training, evaluation, and plotting.
- `src/`: core implementation (losses, training loop, eval metrics, plotting helpers).
- `tests/`: small unit tests for the delta-orth loss.

## Quickstart

1) Train a small model:

```bash
python scripts/train.py --config configs/train_small.json
```

Baseline (no delta-orth):

```bash
python scripts/train.py --config configs/train_small_baseline.json
```

2) Run evaluation metrics (PPL, delta stats, BI, layer drop, CKA):

```bash
python scripts/eval.py --config configs/eval_small.json --run_dir runs/exp_small
```

3) Plot results (baseline vs delta-orth):

```bash
python scripts/plot_results.py \
  --baseline_dir runs/exp_baseline \
  --delta_dir runs/exp_delta_orth \
  --out_dir runs/plots
```

## Notes

- The training loop uses `output_hidden_states=True` to compute deltas.
- Layer skipping uses forward hooks; it works for GPT-2/GPT-NeoX/LLaMA-style blocks.
- Example configs use tiny splits for sanity checks; scale up by adjusting splits and steps.
