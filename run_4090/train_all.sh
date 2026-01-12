source ~/miniconda3/etc/profile.d/conda.sh
conda activate esn

cd ../marginal-utility-regularization

python scripts/train.py --config configs/scaling/train_gpt2_50m_baseline.json
python scripts/train.py --config configs/scaling/train_gpt2_50m_mur.json
python scripts/train.py --config configs/scaling/train_llama_50m_baseline.json
python scripts/train.py --config configs/scaling/train_llama_50m_mur.json
python scripts/train.py --config configs/scaling/train_transformer_pp_50m_baseline.json
python scripts/train.py --config configs/scaling/train_transformer_pp_50m_mur.json
