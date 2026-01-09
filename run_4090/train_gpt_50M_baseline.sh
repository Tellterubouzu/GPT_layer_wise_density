source ~/miniconda3/etc/profile.d/conda.sh
conda activate esn

cd ../marginal-utility-regularization

python scripts/train.py --config configs/scaling/train_gpt2_50m_baseline.json
python scripts/eval.py \
  --config configs/eval_fineweb.json \
  --run_dir runs/gpt2_50m_baseline/ \
  --output_file results/gpt2_50m_baseline_fineweb.json