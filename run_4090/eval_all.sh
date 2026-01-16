source ~/miniconda3/etc/profile.d/conda.sh
conda activate esn

cd ../marginal-utility-regularization

python scripts/eval.py \
    --config configs/eval_scaling/eval_gpt2_42m.json \
    --run_dir runs/gpt2_42m_baseline_20260110/
python scripts/eval.py \
    --config configs/eval_scaling/eval_gpt2_42m.json \
    --run_dir runs/gpt2_42m_mur_20260110/
python scripts/eval.py \
    --config configs/eval_scaling/eval_llama_58m.json \
    --run_dir runs/llama_58m_baseline_20260110/ 
python scripts/eval.py \
    --config configs/eval_scaling/eval_llama_58m.json \
    --run_dir runs/llama_58m_mur_20260110/ 
python scripts/eval.py \
    --config configs/eval_scaling/eval_transformer_pp_51m.json \
    --run_dir runs/transformer_pp_51m_baseline_20260110
python scripts/eval.py \
    --config configs/eval_scaling/eval_transformer_pp_51m.json \
    --run_dir runs/transformer_pp_51m_mur_20260111/
