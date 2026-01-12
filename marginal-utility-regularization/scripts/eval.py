import argparse
import json
import os
import re
import sys
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from eval.bi_metric import compute_bi_metric
from eval.layer_drop import compute_layer_drop
from eval.mur_stats import compute_mur_stats
from eval.ppl import compute_ppl
from models.checkpointing import load_model_state
from models.model_factory import build_model, build_tokenizer
from utils.config import apply_overrides, load_config
from utils.data import TokenLimitedLoader, build_streaming_dataloader
from utils.logging import setup_logger
from utils.seed import set_seed


def _parse_run_name(run_name: str) -> Optional[Tuple[str, str, str]]:
    for arch in ("transformer_pp", "gpt2", "llama"):
        prefix = f"{arch}_"
        if run_name.startswith(prefix):
            rest = run_name[len(prefix) :]
            parts = rest.split("_")
            if len(parts) >= 2:
                return arch, parts[0], parts[1]
    return None


def _parse_size(tag: str) -> Optional[int]:
    match = re.match(r"(\\d+)", tag)
    if not match:
        return None
    return int(match.group(1))


def _infer_model_config_from_run_name(run_name: str) -> Optional[Dict]:
    parsed = _parse_run_name(run_name)
    if parsed is None:
        return None
    arch, size_tag, variant = parsed
    size_value = _parse_size(size_tag)
    if size_value is None:
        return None

    configs_dir = os.path.join(os.path.dirname(__file__), "..", "configs", "scaling")
    if not os.path.isdir(configs_dir):
        return None

    candidates = []
    for fname in os.listdir(configs_dir):
        if not fname.endswith(".json"):
            continue
        if not fname.startswith(f"train_{arch}_") or f"_{variant}.json" not in fname:
            continue
        parts = fname.split("_")
        if len(parts) < 4:
            continue
        size_part = parts[2]
        candidate_size = _parse_size(size_part)
        if candidate_size is None:
            continue
        candidates.append((abs(candidate_size - size_value), os.path.join(configs_dir, fname)))

    if not candidates:
        return None

    _, best_path = sorted(candidates, key=lambda x: x[0])[0]
    return load_config(best_path)

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MUR experiments")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument("--run_dir", default=None, help="Directory with saved model")
    parser.add_argument("--override", action="append", default=[], help="Override config: key=value")
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_overrides(config, args.override)

    set_seed(config.get("seed", 42))

    output_dir = args.run_dir or "runs/eval"
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger("eval", os.path.join(output_dir, "eval.log"))
    model_config = config
    if args.run_dir:
        run_config_path = os.path.join(args.run_dir, "config.json")
        if os.path.exists(run_config_path):
            model_config = load_config(run_config_path)
            if "tokenizer_name" not in model_config and "tokenizer_name" in config:
                model_config["tokenizer_name"] = config["tokenizer_name"]
        if "model" not in model_config and "model_name" not in model_config:
            if "model" in config:
                model_config["model"] = config["model"]
            if "model_name" in config:
                model_config["model_name"] = config["model_name"]
        if "model" not in model_config and "model_name" not in model_config:
            run_name = os.path.basename(os.path.normpath(args.run_dir))
            fallback = _infer_model_config_from_run_name(run_name)
            if fallback is not None:
                model_config = fallback
                if "tokenizer_name" not in model_config and "tokenizer_name" in config:
                    model_config["tokenizer_name"] = config["tokenizer_name"]
                logger.warning("Using fallback model config from configs/scaling for %s", run_name)

    tokenizer = build_tokenizer(model_config)
    model = build_model(model_config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.run_dir:
        loaded = load_model_state(model, args.run_dir, map_location=device)
        if not loaded:
            logger.warning("Model state not found in %s; using random initialization", args.run_dir)

    eval_split = config.get("eval_split") or config.get("train_split") or "train"
    eval_loader = build_streaming_dataloader(
        dataset_path=config["dataset_name"],
        dataset_config=config.get("dataset_config"),
        split=eval_split,
        tokenizer=tokenizer,
        batch_size=config["eval_batch_size"],
        seq_len=config["seq_len"],
        num_workers=config.get("num_workers", 0),
        shuffle_buffer=config.get("streaming_shuffle_buffer", 0),
        seed=config.get("seed", 42),
    )
    max_eval_tokens = config.get("max_eval_tokens")
    limited_loader = TokenLimitedLoader(eval_loader, max_eval_tokens)

    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    ppl = compute_ppl(model, limited_loader, device)
    with open(os.path.join(metrics_dir, "ppl.json"), "w", encoding="utf-8") as f:
        json.dump(ppl, f, indent=2, ensure_ascii=True)

    if config.get("mur_stats", {}).get("enabled", False):
        mur = config["mur_stats"]
        stats = compute_mur_stats(
            model,
            limited_loader,
            device,
            metric=mur.get("metric", "cos"),
            mid_start=mur.get("mid_start", 0.0),
            mid_end=mur.get("mid_end", 1.0),
            tau=mur.get("tau", 0.0),
            sample_k=mur.get("sample_k", 64),
            eps=mur.get("eps", 1e-6),
            fp32_dot=mur.get("fp32_dot", True),
        )
        with open(os.path.join(metrics_dir, "mur_stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=True)

    if config.get("bi_metric", {}).get("enabled", False):
        bi = compute_bi_metric(model, limited_loader, device)
        with open(os.path.join(metrics_dir, "bi_metric.json"), "w", encoding="utf-8") as f:
            json.dump(bi, f, indent=2, ensure_ascii=True)

    if config.get("layer_drop", {}).get("enabled", False):
        drop = compute_layer_drop(model, TokenLimitedLoader(eval_loader, max_eval_tokens), device)
        with open(os.path.join(metrics_dir, "layer_drop.json"), "w", encoding="utf-8") as f:
            json.dump(drop, f, indent=2, ensure_ascii=True)

    logger.info("Evaluation completed")


if __name__ == "__main__":
    main()
