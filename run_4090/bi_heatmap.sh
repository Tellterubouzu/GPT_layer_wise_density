source ~/miniconda3/etc/profile.d/conda.sh
conda activate esn

cd ../marginal-utility-regularization


python scripts/bi_experiment.py \
    --config configs/bi_config/bi_gpt2_baseline.json \
    --config configs/bi_config/bi_gpt2_mur.json \
    --config configs/bi_config/bi_llama_baseline.json \
    --config configs/bi_config/bi_llama_mur.json \
    --config configs/bi_config/bi_transformer_pp_baseline.json \
    --config configs/bi_config/bi_transformer_pp_mur.json \
    --max_train_tokens 500_000_000 \
    --max_eval_tokens 100000 \
    --early_checkpoint_tokens 327_860 \
    --early_checkpoint_interval 8192 \
    --late_checkpoint_interval 4_915_200
