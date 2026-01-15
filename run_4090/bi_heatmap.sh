source ~/miniconda3/etc/profile.d/conda.sh
conda activate esn

cd ../marginal-utility-regularization


python scripts/bi_experiment.py \
    --config configs/eval_scaling/eval_gpt2_42m.json \
    --max_train_tokens 500_000_000 \
    --max_eval_tokens 100000 \
    --early_checkpoint_tokens 163_840 \
    --early_checkpoint_interval 8192 \
    --late_checkpoint_interval 4_915_200

python scripts/bi_experiment.py \
    --config configs/eval_scaling/eval_llama_58m.json \
    --max_train_tokens 500_000_000 \
    --max_eval_tokens 100000 \
    --early_checkpoint_tokens 163_840 \
    --early_checkpoint_interval 8192 \
    --late_checkpoint_interval 4_915_200

python scripts/bi_experiment.py \
    --config configs/eval_scaling/eval_transformer_pp_51m.json \
    --max_train_tokens 500_000_000 \
    --max_eval_tokens 100000 \
    --early_checkpoint_tokens 163_840 \
    --early_checkpoint_interval 8192 \
    --late_checkpoint_interval 4_915_200
