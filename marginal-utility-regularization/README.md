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
python3 scripts/train.py --config configs/train_small_mur_loss.json
```

Baseline (no MUR):

```bash
python3 scripts/train.py --config configs/train_small_baseline.json
```

2) Run evaluation metrics (PPL, MUR stats, BI, layer drop):

```bash
python3 scripts/eval.py --config configs/eval_small.json --run_dir runs/exp_small_mur_loss
```

3) Plot results (baseline vs MUR):

```bash
python3 scripts/plot_results.py \
  --baseline_dir runs/exp_small_baseline \
  --mur_dir runs/exp_small_mur_loss \
  --out_dir runs/plots
```

## Dataset + Evaluation

- Training uses a streaming dataset (no train/val split, no shuffle by default).
- Evaluation is run only after training with `scripts/eval.py`.
- For FineWeb-Edu, PPL is computed over the first 100M tokens via `max_eval_tokens`.
- Training runs in bfloat16 (`bf16: true` in train configs).
- Training stops when `max_train_tokens` is reached; scheduler warmup is set by `warmup_ratio` (default 0.1).
- Training logs are also sent to Weights & Biases when `wandb_enabled: true`.

## Scaling experiments (FineWeb-Edu)

Scaling configs for GPT-2, Transformer++, and Llama (50M/100M/300M) live in `configs/scaling/`.
These configs default to the FineWeb-Edu dataset; adjust splits/steps as needed.

```bash
python3 scripts/train.py --config configs/scaling/train_gpt2_50m_baseline.json
python3 scripts/train.py --config configs/scaling/train_llama_100m_mur.json
```

Evaluation (FineWeb-Edu, first 100M tokens):

```bash
python3 scripts/eval.py --config configs/eval_fineweb.json --run_dir runs/llama_100m_mur
```

After running multiple experiments, generate a scaling plot and table:

```bash
python3 scripts/plot_scaling.py --runs_dir runs --out_dir runs/plots
```

## Notes

- Models can be initialized from scratch by providing a `model` block in the config (see scaling configs).
- Use `max_train_tokens` in the config to cap training by total seen tokens (stops early once reached).
- `mur.mode=loss` uses `autograd.grad` to compute the per-layer gradient for the MUR loss.
- `mur.mode=update` uses a single backward pass and scales layer gradients before `optimizer.step`.
- Transformer++ here uses Pre-LN + RMSNorm + SwiGLU (optional RoPE via config).
- All training configs use the `meta-llama/Llama-2-7b-hf` tokenizer by default.
- Example configs use tiny splits for sanity checks; scale up by adjusting splits and steps.
