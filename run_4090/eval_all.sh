source ~/miniconda3/etc/profile.d/conda.sh
conda activate esn

cd ../marginal-utility-regularization

python scripts/eval.py \
    --config configs/eval_fineweb.json \
    --run_dir runs/gpt2_42m_baseline_20260110/
python scripts/eval.py \
    --config configs/eval_fineweb.json \
    --run_dir runs/gpt2_42m_mur_20260110/
python scripts/eval.py \
    --config configs/eval_fineweb.json \
    --run_dir runs/llama_58m_baseline_20260110/ 
python scripts/eval.py \
    --config configs/eval_fineweb.json \
    --run_dir runs/llama_58m_mur_20260110/ 
python scripts/eval.py \
    --config configs/eval_fineweb.json \
    --run_dir runs/transformer_pp_51m_baseline_20260110                   
python scripts/eval.py \
    --config configs/eval_fineweb.json \
    --run_dir runs/transformer_pp_51m_mur_20260111/
