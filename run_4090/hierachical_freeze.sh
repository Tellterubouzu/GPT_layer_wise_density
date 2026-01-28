source ~/miniconda3/etc/profile.d/conda.sh
conda activate esn

cd ../marginal-utility-regularization

UNFREEZE_TOKENS=${UNFREEZE_TOKENS:-10000000}

for mode in input output random; do
  python scripts/hierachical_freeze.py \
    --config configs/scaling/train_gpt2_50m_baseline.json \
    --override hierarchical_freeze.mode=${mode} \
    --override hierarchical_freeze.unfreeze_tokens=${UNFREEZE_TOKENS}

  python scripts/hierachical_freeze.py \
    --config configs/scaling/train_llama_50m_baseline.json \
    --override hierarchical_freeze.mode=${mode} \
    --override hierarchical_freeze.unfreeze_tokens=${UNFREEZE_TOKENS}

  python scripts/hierachical_freeze.py \
    --config configs/scaling/train_transformer_pp_50m_baseline.json \
    --override hierarchical_freeze.mode=${mode} \
    --override hierarchical_freeze.unfreeze_tokens=${UNFREEZE_TOKENS}
done

for mode in input output random; do
  latest_gpt2=$(ls -td runs/gpt2_*_hfreeze_${mode}_* 2>/dev/null | head -n 1)
  latest_llama=$(ls -td runs/llama_*_hfreeze_${mode}_* 2>/dev/null | head -n 1)
  latest_transformer_pp=$(ls -td runs/transformer_pp_*_hfreeze_${mode}_* 2>/dev/null | head -n 1)

  if [ -n "${latest_gpt2}" ]; then
    python scripts/eval.py \
      --config configs/eval_scaling/eval_gpt2_42m.json \
      --run_dir "${latest_gpt2}"
    python scripts/plot_freeze_results.py \
      --baseline_dir runs/gpt2_42m_baseline_20260110 \
      --freeze_dir "${latest_gpt2}" \
      --out_dir runs/plots/gpt2_42m_hfreeze_${mode}
  else
    echo "No gpt2 hfreeze run found for mode ${mode}; skipping eval/plot."
  fi

  if [ -n "${latest_llama}" ]; then
    python scripts/eval.py \
      --config configs/eval_scaling/eval_llama_58m.json \
      --run_dir "${latest_llama}"
    python scripts/plot_freeze_results.py \
      --baseline_dir runs/llama_58m_baseline_20260110 \
      --freeze_dir "${latest_llama}" \
      --out_dir runs/plots/llama_58m_hfreeze_${mode}
  else
    echo "No llama hfreeze run found for mode ${mode}; skipping eval/plot."
  fi

  if [ -n "${latest_transformer_pp}" ]; then
    python scripts/eval.py \
      --config configs/eval_scaling/eval_transformer_pp_51m.json \
      --run_dir "${latest_transformer_pp}"
    python scripts/plot_freeze_results.py \
      --baseline_dir runs/transformer_pp_51m_baseline_20260110 \
      --freeze_dir "${latest_transformer_pp}" \
      --out_dir runs/plots/transformer_pp_51m_hfreeze_${mode}
  else
    echo "No transformer_pp hfreeze run found for mode ${mode}; skipping eval/plot."
  fi
done
