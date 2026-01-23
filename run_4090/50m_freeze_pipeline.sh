source ~/miniconda3/etc/profile.d/conda.sh
conda activate esn

cd ../marginal-utility-regularization

python scripts/freezed_experiment.py --config configs/scaling/train_gpt2_50m_freeze.json
python scripts/freezed_experiment.py --config configs/scaling/train_llama_50m_freeze.json
python scripts/freezed_experiment.py --config configs/scaling/train_transformer_pp_50m_freeze.json

python scripts/eval.py \
  --config configs/eval_scaling/eval_gpt2_42m.json \
  --run_dir runs/gpt2_42m_baseline_freeze_layer0_20260121

python scripts/eval.py \
  --config configs/eval_scaling/eval_gpt2_42m.json \
  --run_dir runs/llama_58m_baseline_freeze_layer0_20260121
python scripts/eval.py \
  --config configs/eval_scaling/eval_gpt2_42m.json \
  --run_dir runs/transformer_pp_51m_baseline_freeze_layer0_20260121

python scripts/plot_freeze_results.py \
  --baseline_dir runs/gpt2_42m_baseline_20260110 \
  --freeze_dir runs/gpt2_42m_baseline_freeze_layer0_20260121 \
  --out_dir runs/plots/gpt2_42m_freeze_initial0121

python scripts/plot_freeze_results.py \
  --baseline_dir runs/llama_58m_baseline_20260110 \
  --freeze_dir runs/llama_58m_baseline_freeze_layer0_20260121 \
  --out_dir runs/plots/llama_58m_freeze_initial0121

python scripts/plot_freeze_results.py \
  --baseline_dir runs/transformer_pp_51m_baseline_20260110 \
  --freeze_dir runs/transformer_pp_51m_baseline_freeze_layer0_20260121 \
  --out_dir runs/plots/transformer_pp_51m_freeze_initial0121