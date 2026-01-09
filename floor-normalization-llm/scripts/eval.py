import argparse
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from eval.bi_metric import compute_bi_metric
from eval.layer_drop import compute_layer_drop
from eval.ppl import compute_ppl
from eval.repr_similarity import compute_adjacent_cka, compute_adjacent_cos
from utils.config import apply_overrides, load_config
from utils.data import build_dataloader
from utils.logging import setup_logger
from utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BI-Floor experiments")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument("--run_dir", default=None, help="Directory with saved model")
    parser.add_argument("--override", action="append", default=[], help="Override config: key=value")
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_overrides(config, args.override)

    set_seed(config.get("seed", 42))

    model_path = args.run_dir or config["model_name"]
    output_dir = args.run_dir or "runs/eval"
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger("eval", os.path.join(output_dir, "eval.log"))
    logger.info("Loading model: %s", model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    raw_eval = load_dataset(config["dataset_name"], config.get("dataset_config"), split=config["eval_split"])
    eval_loader = build_dataloader(
        raw_eval,
        tokenizer,
        config["eval_batch_size"],
        config["block_size"],
        shuffle=False,
        text_column=config.get("text_column", "text"),
    )

    metrics_dir = os.path.join(output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    ppl = compute_ppl(model, eval_loader, device)
    with open(os.path.join(metrics_dir, "ppl.json"), "w", encoding="utf-8") as f:
        json.dump(ppl, f, indent=2, ensure_ascii=True)

    if config.get("bi_metric", {}).get("enabled", False):
        bi_cfg = config.get("bi_metric", {})
        bi = compute_bi_metric(
            model,
            eval_loader,
            device,
            token_reduce=bi_cfg.get("token_reduce", "mean"),
            sample_k=bi_cfg.get("sample_k", 32),
            eps=bi_cfg.get("eps", 1e-6),
            fp32=bi_cfg.get("fp32", True),
        )
        with open(os.path.join(metrics_dir, "bi_metric.json"), "w", encoding="utf-8") as f:
            json.dump(bi, f, indent=2, ensure_ascii=True)

    if config.get("layer_drop", {}).get("enabled", False):
        drop = compute_layer_drop(model, eval_loader, device)
        with open(os.path.join(metrics_dir, "layer_drop.json"), "w", encoding="utf-8") as f:
            json.dump(drop, f, indent=2, ensure_ascii=True)

    if config.get("repr_similarity", {}).get("enabled", False):
        repr_cfg = config.get("repr_similarity", {})
        method = repr_cfg.get("method", "cos")
        if method == "cka":
            repr_metrics = compute_adjacent_cka(
                model,
                eval_loader,
                device,
                max_samples=repr_cfg.get("max_samples", 2048),
            )
            out_name = "repr_cka.json"
        else:
            repr_metrics = compute_adjacent_cos(
                model,
                eval_loader,
                device,
                token_reduce=repr_cfg.get("token_reduce", "mean"),
                sample_k=repr_cfg.get("sample_k", 32),
                eps=repr_cfg.get("eps", 1e-6),
                fp32=repr_cfg.get("fp32", True),
            )
            out_name = "repr_cos.json"
        with open(os.path.join(metrics_dir, out_name), "w", encoding="utf-8") as f:
            json.dump(repr_metrics, f, indent=2, ensure_ascii=True)

    logger.info("Evaluation completed")


if __name__ == "__main__":
    main()
