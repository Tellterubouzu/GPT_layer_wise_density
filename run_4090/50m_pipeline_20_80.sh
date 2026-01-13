source ~/miniconda3/etc/profile.d/conda.sh
conda activate esn

cd ../marginal-utility-regularization

python scripts/train.py --config configs/scaling/train_gpt2_50m_mur.json
python scripts/train.py --config configs/scaling/train_llama_50m_mur.json
python scripts/train.py --config configs/scaling/train_transformer_pp_50m_mur.json


python scripts/eval.py \
    --config configs/eval_scaling/eval_gpt2_42m.json \
    --run_dir runs/gpt2_42m_mur_layer20_80_20260113/ 
python scripts/eval.py \
    --config configs/eval_scaling/eval_llama_58m.json \
    --run_dir runs/llama_58m_mur_layer20_80_20260113/ 
python scripts/eval.py \
    --config configs/eval_scaling/eval_transformer_pp_51m.json \
    --run_dir runs/transformer_pp_51m_mur_layer20_80_20260113/

python3 scripts/plot_results.py \
  --baseline_dir runs/gpt2_42m_baseline_20260110 \
  --mur_dir runs/gpt2_42m_mur_layer20_80_20260113 \
  --out_dir runs/plots/gpt2_42m_mur_20_80

python3 scripts/plot_results.py \
  --baseline_dir runs/llama_58m_baseline_20260110 \
  --mur_dir runs/llama_58m_mur_layer20_80_20260113 \
  --out_dir runs/plots/llama_58m_mur_20_80

python3 scripts/plot_results.py \
  --baseline_dir runs/transformer_pp_51m_baseline_20260110 \
  --mur_dir runs/transformer_pp_51m_mur_layer20_80_20260113 \
  --out_dir runs/plots/transformer_pp_51m_mur_20_80
