source ~/miniconda3/etc/profile.d/conda.sh
conda activate esn

cd ../marginal-utility-regularization


python3 scripts/plot_results.py \
  --baseline_dir runs/gpt2_42m_baseline_20260110 \
  --mur_dir runs/gpt2_42m_mur_20260110 \
  --out_dir runs/plots/gpt2_42m

python3 scripts/plot_results.py \
  --baseline_dir runs/llama_58m_baseline_20260110 \
  --mur_dir runs/llama_58m_mur_20260110 \
  --out_dir runs/plots/llama_58m

python3 scripts/plot_results.py \
  --baseline_dir runs/transformer_pp_51m_baseline_20260110 \
  --mur_dir runs/transformer_pp_51m_mur_2026011 \
  --out_dir runs/plots/transformer_pp_51m