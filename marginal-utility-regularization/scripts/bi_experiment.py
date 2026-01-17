import argparse
import json
import math
import os
import sys
import time
import shutil
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import wandb
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

from eval.bi_metric import compute_bi_metric
from eval.layer_drop import compute_layer_drop
from models.block_access import get_transformer_blocks
from models.checkpointing import save_model_state
from models.hidden_state_utils import hidden_states_to_deltas, resolve_layer_indices
from models.model_factory import build_model, build_tokenizer
from mur.optimizer_scaling import compute_layer_multipliers, scale_layer_grads
from train.train_clm import (
    MurAccumulator,
    _compute_layer_means,
    _compute_mur_loss,
    _make_grad_hook,
    _mur_strength,
    _resolve_mur_cfg,
)
from utils.config import apply_overrides, load_config, save_config
from utils.data import TokenLimitedLoader, build_streaming_dataloader
from utils.logging import setup_logger
from utils.metrics_store import MetricsStore
from utils.seed import set_seed


DEFAULT_CONFIGS = {
    "gpt2_baseline": "configs/bi_config/bi_gpt2_baseline.json",
    "gpt2_mur": "configs/bi_config/bi_gpt2_mur.json",
    "llama_baseline": "configs/bi_config/bi_llama_baseline.json",
    "llama_mur": "configs/bi_config/bi_llama_mur.json",
    "transformer_pp_baseline": "configs/bi_config/bi_transformer_pp_baseline.json",
    "transformer_pp_mur": "configs/bi_config/bi_transformer_pp_mur.json",
}


def _cycle(loader):
    while True:
        for batch in loader:
            yield batch


def _resolve_arch(config: Dict[str, Any]) -> str:
    model_cfg = config.get("model", {})
    arch = model_cfg.get("arch", "gpt2")
    if arch == "transformer++":
        arch = "transformer_pp"
    return arch


def _infer_run_name(arch: str, param_count_m: float, variant: str, total_tokens: int) -> str:
    date_tag = datetime.now().strftime("%Y%m%d")
    size_tag = f"{int(round(param_count_m))}M"
    total_tokens_mt = int(round(total_tokens / 1_000_000))
    return f"{arch}_{size_tag}_{variant}_{total_tokens_mt}MT_{date_tag}"


def _normalize_training_config(config: Dict[str, Any]) -> List[str]:
    warnings: List[str] = []

    if "dataset_name" not in config:
        raise ValueError("dataset_name is required for training")
    if "seq_len" not in config:
        raise ValueError("seq_len is required for training")

    def _set_default(key: str, value: Any, reason: str) -> None:
        if key not in config:
            config[key] = value
            warnings.append(f"Missing {key}; using {value} ({reason})")

    train_split = config.get("eval_split", "train")
    _set_default("train_split", train_split, "fallback to eval_split")
    _set_default("batch_size", config.get("eval_batch_size", 32), "fallback to eval_batch_size or 1")
    _set_default("learning_rate", 1e-3, "default")
    _set_default("weight_decay", 0.1, "default")
    _set_default("log_steps", 2500, "default")
    _set_default("max_grad_norm", 1.0, "default")
    _set_default("gradient_accumulation_steps", 0, "default")
    _set_default("warmup_ratio", 0.1, "default")

    return warnings


def _resolve_variant(config: Dict[str, Any]) -> str:
    mur_cfg = config.get("mur", {})
    return "mur" if mur_cfg.get("enabled", False) else "baseline"


def _build_checkpoint_schedule(
    max_tokens: int,
    early_until: int,
    early_interval: int,
    late_interval: int,
) -> List[int]:
    if max_tokens <= 0:
        return []
    schedule = []
    early_end = min(max_tokens, early_until)
    if early_interval > 0:
        current = early_interval
        while current <= early_end:
            schedule.append(current)
            current += early_interval
    if max_tokens > early_until and late_interval > 0:
        current = early_until + late_interval
        while current <= max_tokens:
            schedule.append(current)
            current += late_interval
    if not schedule:
        schedule.append(max_tokens)
    elif schedule[-1] != max_tokens:
        schedule.append(max_tokens)
    return schedule


def _format_tokens(value: int) -> str:
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.0f}M"
    return str(value)


def _save_checkpoint(model: torch.nn.Module, output_dir: str, token_target: int) -> str:
    ckpt_dir = os.path.join(output_dir, "checkpoints", f"tokens_{token_target}")
    os.makedirs(ckpt_dir, exist_ok=True)
    save_model_state(model, ckpt_dir)
    return ckpt_dir


def _evaluate_bi(
    model: torch.nn.Module,
    eval_loader,
    max_eval_tokens: Optional[int],
    device: torch.device,
) -> Dict[str, List[float]]:
    limited_loader = TokenLimitedLoader(eval_loader, max_eval_tokens)
    return compute_bi_metric(model, limited_loader, device)


def _evaluate_layer_drop(
    model: torch.nn.Module,
    eval_loader,
    max_eval_tokens: int,
    device: torch.device,
) -> Dict[str, List[float]]:
    limited_loader = TokenLimitedLoader(eval_loader, max_eval_tokens)
    return compute_layer_drop(model, limited_loader, device)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def _plot_bi_heatmap(records: List[Dict[str, Any]], out_path: str, title: str) -> None:
    if not records:
        return
    records = sorted(records, key=lambda r: r["checkpoint_tokens"])
    tokens = [r["checkpoint_tokens"] for r in records]
    bi_values = [r["bi"] for r in records]
    matrix = list(map(list, zip(*bi_values)))
    plt.figure(figsize=(10, 6))
    plt.imshow(matrix, aspect="auto", origin="lower")
    tick_stride = max(1, len(tokens) // 6)
    tick_indices = list(range(0, len(tokens), tick_stride))
    if (len(tokens) - 1) not in tick_indices:
        tick_indices.append(len(tokens) - 1)
    plt.xticks(tick_indices, [_format_tokens(tokens[i]) for i in tick_indices])
    plt.xlabel("Tokens trained")
    plt.ylabel("Layer")
    plt.title(title)
    plt.colorbar(label="Block Influence")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _plot_bi_layer_traces(records: List[Dict[str, Any]], out_path: str, title: str) -> None:
    if not records:
        return
    records = sorted(records, key=lambda r: r["checkpoint_tokens"])
    tokens = [r["checkpoint_tokens"] for r in records]
    bi_values = [r["bi"] for r in records]
    num_layers = len(bi_values[0])
    indices = [0]
    if num_layers > 1:
        indices.extend(
            sorted(
                set(
                    [
                        num_layers // 4,
                        num_layers // 2,
                        (3 * num_layers) // 4,
                        num_layers - 1,
                    ]
                )
            )
        )
    indices = indices[:6]
    x_vals = [t / 1_000_000_000 for t in tokens]
    plt.figure(figsize=(10, 5))
    for idx in indices:
        series = [bi[idx] for bi in bi_values]
        plt.plot(x_vals, series, label=f"layer {idx}")
    plt.xlabel("Tokens trained (B)")
    plt.ylabel("Block Influence")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _plot_bi_mean(records: List[Dict[str, Any]], out_path: str, title: str) -> None:
    if not records:
        return
    records = sorted(records, key=lambda r: r["checkpoint_tokens"])
    tokens = [r["checkpoint_tokens"] for r in records]
    means = []
    for record in records:
        values = record["bi"]
        means.append(sum(values) / max(1, len(values)))
    x_vals = [t / 1_000_000_000 for t in tokens]
    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, means)
    plt.xlabel("Tokens trained (B)")
    plt.ylabel("Mean Block Influence")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _plot_bi_results(run_dir: str, run_label: str) -> None:
    metrics_path = os.path.join(run_dir, "metrics", "bi_over_time.jsonl")
    if not os.path.exists(metrics_path):
        return
    records = _load_jsonl(metrics_path)
    plots_dir = os.path.join(run_dir, "plots")
    _plot_bi_heatmap(records, os.path.join(plots_dir, "bi_heatmap.png"), f"BI heatmap ({run_label})")
    _plot_bi_layer_traces(
        records,
        os.path.join(plots_dir, "bi_layer_traces.png"),
        f"BI layer traces ({run_label})",
    )
    _plot_bi_mean(records, os.path.join(plots_dir, "bi_mean.png"), f"BI mean ({run_label})")


def _plot_layer_drop_heatmap(records: List[Dict[str, Any]], out_path: str, title: str) -> None:
    if not records:
        return
    records = sorted(records, key=lambda r: r["checkpoint_tokens"])
    tokens = [r["checkpoint_tokens"] for r in records]
    deltas = [r["delta_ppl"] for r in records]
    matrix = list(map(list, zip(*deltas)))
    plt.figure(figsize=(10, 6))
    plt.imshow(matrix, aspect="auto", origin="lower")
    tick_stride = max(1, len(tokens) // 6)
    tick_indices = list(range(0, len(tokens), tick_stride))
    if (len(tokens) - 1) not in tick_indices:
        tick_indices.append(len(tokens) - 1)
    plt.xticks(tick_indices, [_format_tokens(tokens[i]) for i in tick_indices])
    plt.xlabel("Tokens trained")
    plt.ylabel("Layer")
    plt.title(title)
    plt.colorbar(label="Delta PPL")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _plot_layer_drop_mean(records: List[Dict[str, Any]], out_path: str, title: str) -> None:
    if not records:
        return
    records = sorted(records, key=lambda r: r["checkpoint_tokens"])
    tokens = [r["checkpoint_tokens"] for r in records]
    means = []
    for record in records:
        values = record["delta_ppl"]
        means.append(sum(values) / max(1, len(values)))
    x_vals = [t / 1_000_000_000 for t in tokens]
    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, means)
    plt.xlabel("Tokens trained (B)")
    plt.ylabel("Mean delta PPL")
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _plot_layer_drop_traces(records: List[Dict[str, Any]], out_path: str, title: str) -> None:
    if not records:
        return
    records = sorted(records, key=lambda r: r["checkpoint_tokens"])
    tokens = [r["checkpoint_tokens"] for r in records]
    deltas = [r["delta_ppl"] for r in records]
    num_layers = len(deltas[0])
    indices = [0]
    if num_layers > 1:
        indices.extend(
            sorted(
                set(
                    [
                        num_layers // 4,
                        num_layers // 2,
                        (3 * num_layers) // 4,
                        num_layers - 1,
                    ]
                )
            )
        )
    indices = indices[:6]
    x_vals = [t / 1_000_000_000 for t in tokens]
    plt.figure(figsize=(10, 5))
    for idx in indices:
        series = [vals[idx] for vals in deltas]
        plt.plot(x_vals, series, label=f"layer {idx}")
    plt.xlabel("Tokens trained (B)")
    plt.ylabel("Delta PPL")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def _plot_layer_drop_results(run_dir: str, run_label: str) -> None:
    metrics_path = os.path.join(run_dir, "metrics", "layer_drop_over_time.jsonl")
    if not os.path.exists(metrics_path):
        return
    records = _load_jsonl(metrics_path)
    plots_dir = os.path.join(run_dir, "plots")
    _plot_layer_drop_heatmap(records, os.path.join(plots_dir, "layer_drop_heatmap.png"), f"Layer drop ({run_label})")
    _plot_layer_drop_traces(
        records,
        os.path.join(plots_dir, "layer_drop_traces.png"),
        f"Layer drop traces ({run_label})",
    )
    _plot_layer_drop_mean(records, os.path.join(plots_dir, "layer_drop_mean.png"), f"Layer drop mean ({run_label})")


def _cleanup_run_dir(run_dir: str, keep_names: Optional[List[str]] = None) -> None:
    if not os.path.isdir(run_dir):
        return
    keep = set(keep_names or [])
    for entry in os.listdir(run_dir):
        if entry in keep:
            continue
        path = os.path.join(run_dir, entry)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)


def _train_and_eval_bi(
    config: Dict[str, Any],
    output_dir: str,
    args,
    model: torch.nn.Module,
    tokenizer,
    param_count: int,
    param_count_m: float,
    warnings: Optional[List[str]] = None,
) -> None:
    model.train()

    logger = setup_logger(f"bi_experiment_{os.path.basename(output_dir)}", os.path.join(output_dir, "train.log"))
    if warnings:
        for warning in warnings:
            logger.warning(warning)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    use_bf16 = config.get("bf16", True)
    if use_bf16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 requested but CUDA device does not support bf16")
    if use_bf16:
        model.to(dtype=torch.bfloat16)

    train_loader = build_streaming_dataloader(
        dataset_path=config["dataset_name"],
        dataset_config=config.get("dataset_config"),
        split=config["train_split"],
        tokenizer=tokenizer,
        batch_size=config["batch_size"],
        seq_len=config["seq_len"],
        num_workers=config.get("num_workers", 0),
        shuffle_buffer=config.get("streaming_shuffle_buffer", 0),
        seed=config.get("seed", 42),
    )
    eval_split = config.get("eval_split") or config.get("train_split") or "train"
    eval_loader = build_streaming_dataloader(
        dataset_path=config["dataset_name"],
        dataset_config=config.get("dataset_config"),
        split=eval_split,
        tokenizer=tokenizer,
        batch_size=config.get("eval_batch_size", config["batch_size"]),
        seq_len=config["seq_len"],
        num_workers=config.get("num_workers", 0),
        shuffle_buffer=config.get("eval_shuffle_buffer", 0),
        seed=config.get("seed", 42),
    )

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    grad_accum = int(config.get("gradient_accumulation_steps", 0) or 0)
    accum_enabled = grad_accum > 0
    tokens_per_optimizer_step = config["batch_size"] * config["seq_len"] * (grad_accum if accum_enabled else 1) * world_size
    max_train_tokens = int(args.max_train_tokens or config.get("max_train_tokens") or 0)
    num_train_steps = int(config.get("num_train_steps", 0) or 0)
    if max_train_tokens <= 0 and num_train_steps <= 0:
        raise ValueError("max_train_tokens or num_train_steps must be set")
    if max_train_tokens > 0:
        total_optimizer_steps = max(1, math.ceil(max_train_tokens / tokens_per_optimizer_step))
    else:
        total_optimizer_steps = max(1, num_train_steps)
    warmup_ratio = float(config.get("warmup_ratio", 0.1))
    warmup_steps = max(0, int(total_optimizer_steps * warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_optimizer_steps
    )

    log_steps = config.get("log_steps", 50)
    max_grad_norm = config.get("max_grad_norm", 1.0)
    max_eval_tokens = args.max_eval_tokens
    if max_eval_tokens is None:
        max_eval_tokens = config.get("max_eval_tokens", 1_000_000)

    wandb_enabled = config.get("wandb_enabled", True)
    if wandb_enabled:
        wandb.init(project=config["project_name"], name=config["run_name"], config=config)
        wandb.log(
            {
                "param_count": param_count,
                "param_count_m": param_count_m,
                "total_steps": total_optimizer_steps,
                "warmup_steps": warmup_steps,
            },
            step=0,
        )

    train_metrics = MetricsStore(os.path.join(output_dir, "metrics", "train.jsonl"))
    bi_metrics = MetricsStore(os.path.join(output_dir, "metrics", "bi_over_time.jsonl"))
    drop_metrics = MetricsStore(os.path.join(output_dir, "metrics", "layer_drop_over_time.jsonl"))

    checkpoint_targets = _build_checkpoint_schedule(
        max_train_tokens,
        early_until=args.early_checkpoint_tokens,
        early_interval=args.early_checkpoint_interval,
        late_interval=args.late_checkpoint_interval,
    )
    checkpoint_idx = 0
    drop_checkpoint_targets = _build_checkpoint_schedule(
        max_train_tokens,
        early_until=args.layer_drop_early_checkpoint_tokens,
        early_interval=args.layer_drop_early_checkpoint_interval,
        late_interval=args.layer_drop_late_checkpoint_interval,
    )
    drop_checkpoint_idx = 0

    train_iter = _cycle(train_loader)
    seen_tokens = 0
    step = 0
    optimizer_step = 0
    log_loss_sum = 0.0
    log_steps_seen = 0
    log_tokens = 0
    last_log_time = time.time()
    last_bi_mean: Optional[float] = None
    last_drop_mean: Optional[float] = None
    mur_cfg = _resolve_mur_cfg(config.get("mur", {}))
    layer_indices: Optional[List[int]] = None
    mur_accum: Optional[MurAccumulator] = None
    blocks = None

    logger.info("Starting BI experiment training")
    while True:
        if max_train_tokens > 0 and seen_tokens >= max_train_tokens:
            break
        if max_train_tokens <= 0 and num_train_steps > 0 and step >= num_train_steps:
            break
        step += 1
        batch = next(train_iter)
        input_ids = batch["input_ids"].to(device)
        autocast_ctx = (
            torch.autocast(device_type=device.type, dtype=torch.bfloat16) if use_bf16 else nullcontext()
        )
        with autocast_ctx:
            outputs = model(input_ids=input_ids, labels=input_ids, output_hidden_states=mur_cfg.enabled)
            lm_loss = outputs.loss
        mur_weight = _mur_strength(step, mur_cfg)
        mur_loss = torch.tensor(0.0, device=device)
        handles = []
        grad_store: Dict[int, torch.Tensor] = {}
        delta_store: Dict[int, torch.Tensor] = {}

        if mur_cfg.enabled and mur_weight > 0.0:
            hidden_states = outputs.hidden_states
            if hidden_states is None:
                raise ValueError("MUR enabled but hidden states are missing")
            num_layers = len(hidden_states) - 1
            if layer_indices is None:
                layer_indices = resolve_layer_indices(num_layers, mur_cfg.mid_start, mur_cfg.mid_end)
            if mur_cfg.mode == "loss":
                mur_loss, _ = _compute_mur_loss(lm_loss, hidden_states, layer_indices, mur_cfg)
            elif mur_cfg.mode == "update":
                if mur_accum is None:
                    mur_accum = MurAccumulator(len(layer_indices))
                if blocks is None:
                    blocks = get_transformer_blocks(model)
                deltas = hidden_states_to_deltas(hidden_states)
                for idx in layer_indices:
                    delta_store[idx] = deltas[idx].detach()
                    handles.append(hidden_states[idx + 1].register_hook(_make_grad_hook(grad_store, idx)))
            else:
                raise ValueError(f"Unknown MUR mode: {mur_cfg.mode}")

        total_loss = lm_loss + mur_weight * mur_loss
        total_loss_value = float(total_loss.detach().item())
        if accum_enabled:
            loss = total_loss / grad_accum
            loss.backward()
        else:
            total_loss.backward()

        if mur_cfg.enabled and mur_cfg.mode == "update" and mur_weight > 0.0:
            for handle in handles:
                handle.remove()
            if layer_indices is None or mur_accum is None:
                raise ValueError("MUR update mode missing layer indices")
            layer_means = _compute_layer_means(layer_indices, grad_store, delta_store, mur_cfg)
            mur_accum.update(layer_means)

        if accum_enabled:
            if step % grad_accum == 0:
                if mur_cfg.enabled and mur_cfg.mode == "update" and mur_weight > 0.0:
                    if layer_indices is None or mur_accum is None or blocks is None:
                        raise ValueError("MUR update mode missing state")
                    avg_means = mur_accum.average()
                    multipliers = compute_layer_multipliers(layer_indices, avg_means, mur_cfg.tau, mur_weight)
                    scale_layer_grads(blocks, multipliers)
                    mur_accum.reset()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                optimizer_step += 1
        else:
            if mur_cfg.enabled and mur_cfg.mode == "update" and mur_weight > 0.0:
                if layer_indices is None or mur_accum is None or blocks is None:
                    raise ValueError("MUR update mode missing state")
                avg_means = mur_accum.average()
                multipliers = compute_layer_multipliers(layer_indices, avg_means, mur_cfg.tau, mur_weight)
                scale_layer_grads(blocks, multipliers)
                mur_accum.reset()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            optimizer_step += 1

        tokens_in_step = input_ids.numel() * world_size
        seen_tokens += tokens_in_step
        log_loss_sum += lm_loss.item()
        log_steps_seen += 1
        log_tokens += tokens_in_step

        if step % log_steps == 0:
            elapsed = max(1e-8, time.time() - last_log_time)
            train_loss = log_loss_sum / max(1, log_steps_seen)
            record = {
                "project_name": config["project_name"],
                "run_name": config["run_name"],
                "step": step,
                "lm_loss": lm_loss.item(),
                "current_loss": lm_loss.item(),
                "current_perplexity": math.exp(lm_loss.item()),
                "train_loss": train_loss,
                "train_perplexity": math.exp(train_loss),
                "lr": scheduler.get_last_lr()[0],
                "current_lr": scheduler.get_last_lr()[0],
                "token_per_sec": log_tokens / elapsed,
                "VRAM_allocated_bytes": torch.cuda.memory_allocated(device) if device.type == "cuda" else 0,
                "seened_tokens": seen_tokens,
                "mur_loss": float(mur_loss.detach().item()),
                "mur_weight": mur_weight,
                "total_loss": total_loss_value,
                "mur_mode": mur_cfg.mode if mur_cfg.enabled else "disabled",
            }
            if last_bi_mean is not None:
                record["bi_mean"] = last_bi_mean
            if last_drop_mean is not None:
                record["layer_drop_mean"] = last_drop_mean
            train_metrics.write(record)
            logger.info(json.dumps(record, ensure_ascii=True))
            if wandb_enabled:
                wandb_record = {k: v for k, v in record.items() if not isinstance(v, list)}
                wandb.log(wandb_record, step=step)
            log_loss_sum = 0.0
            log_steps_seen = 0
            log_tokens = 0
            last_log_time = time.time()

        while checkpoint_idx < len(checkpoint_targets) and seen_tokens >= checkpoint_targets[checkpoint_idx]:
            checkpoint_tokens = checkpoint_targets[checkpoint_idx]
            ckpt_dir = _save_checkpoint(model, output_dir, checkpoint_tokens)
            bi = _evaluate_bi(model, eval_loader, max_eval_tokens, device)
            bi_values = bi.get("bi", [])
            bi_mean = sum(bi_values) / max(1, len(bi_values))
            last_bi_mean = bi_mean
            record = {
                "checkpoint_tokens": checkpoint_tokens,
                "seen_tokens": seen_tokens,
                "step": step,
                "optimizer_step": optimizer_step,
                "bi": bi_values,
                "bi_mean": bi_mean,
                "checkpoint_dir": ckpt_dir,
            }
            bi_metrics.write(record)
            logger.info(json.dumps(record, ensure_ascii=True))
            if wandb_enabled:
                wandb.log(
                    {
                        "bi_mean": bi_mean,
                        "checkpoint_tokens": checkpoint_tokens,
                    },
                    step=step,
                )
            checkpoint_idx += 1

        while drop_checkpoint_idx < len(drop_checkpoint_targets) and seen_tokens >= drop_checkpoint_targets[drop_checkpoint_idx]:
            checkpoint_tokens = drop_checkpoint_targets[drop_checkpoint_idx]
            drop = _evaluate_layer_drop(model, eval_loader, args.layer_drop_eval_tokens, device)
            delta_ppl = drop.get("delta_ppl", [])
            base_ppl = drop.get("base_ppl")
            drop_mean = sum(delta_ppl) / max(1, len(delta_ppl))
            last_drop_mean = drop_mean
            record = {
                "checkpoint_tokens": checkpoint_tokens,
                "seen_tokens": seen_tokens,
                "step": step,
                "optimizer_step": optimizer_step,
                "delta_ppl": delta_ppl,
                "base_ppl": base_ppl,
                "delta_ppl_mean": drop_mean,
            }
            drop_metrics.write(record)
            logger.info(json.dumps(record, ensure_ascii=True))
            if wandb_enabled:
                wandb.log(
                    {
                        "layer_drop_mean": drop_mean,
                        "layer_drop_base_ppl": base_ppl,
                        "layer_drop_checkpoint_tokens": checkpoint_tokens,
                    },
                    step=step,
                )
            drop_checkpoint_idx += 1

        if max_train_tokens > 0 and seen_tokens >= max_train_tokens:
            break

    save_model_state(model, output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Training completed")
    if wandb_enabled:
        wandb.finish()


def _run_from_config(config_path: str, args) -> str:
    config = load_config(config_path)
    config = apply_overrides(config, args.override)
    arch = _resolve_arch(config)
    variant = _resolve_variant(config)

    warnings = _normalize_training_config(config)
    set_seed(config.get("seed", 42))
    tokenizer = build_tokenizer(config)
    model = build_model(config)
    param_count = sum(p.numel() for p in model.parameters())
    param_count_m = param_count / 1e6
    max_train_tokens = int(args.max_train_tokens or config.get("max_train_tokens") or 0)
    run_name = _infer_run_name(arch, param_count_m, variant, max_train_tokens)

    output_dir = os.path.join(args.output_root, run_name)
    os.makedirs(output_dir, exist_ok=True)

    config["project_name"] = "gpt_bi_graph"
    config["run_name"] = run_name
    config["output_dir"] = output_dir
    config["max_train_tokens"] = max_train_tokens
    save_config(config, os.path.join(output_dir, "config.json"))

    _train_and_eval_bi(
        config,
        output_dir,
        args,
        model,
        tokenizer,
        param_count,
        param_count_m,
        warnings,
    )
    _plot_bi_results(output_dir, run_name)
    _plot_layer_drop_results(output_dir, run_name)
    if not args.keep_artifacts:
        _cleanup_run_dir(output_dir, keep_names=["metrics", "plots"])
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="BI experiment: train checkpoints and plot BI over time")
    parser.add_argument("--config", action="append", default=[], help="Path to JSON config (repeatable)")
    parser.add_argument("--override", action="append", default=[], help="Override config: key=value")
    parser.add_argument("--output_root", default="runs/bi_experiment", help="Root directory for experiment outputs")
    parser.add_argument("--max_train_tokens", type=int, default=10_000_000_000)
    parser.add_argument("--max_eval_tokens", type=int, default=None)
    parser.add_argument("--early_checkpoint_tokens", type=int, default=200_000_000)
    parser.add_argument("--early_checkpoint_interval", type=int, default=25_000_000)
    parser.add_argument("--late_checkpoint_interval", type=int, default=100_000_000)
    parser.add_argument("--layer_drop_eval_tokens", type=int, default=1_000_000)
    parser.add_argument("--layer_drop_early_checkpoint_tokens", type=int, default=200_000_000)
    parser.add_argument("--layer_drop_early_checkpoint_interval", type=int, default=100_000_000)
    parser.add_argument("--layer_drop_late_checkpoint_interval", type=int, default=200_000_000)
    parser.add_argument("--plot_only", action="store_true")
    parser.add_argument("--run_dir", action="append", default=[], help="Existing run dir for plotting")
    parser.add_argument("--keep_artifacts", action="store_true", help="Keep checkpoints and logs after plotting")
    args = parser.parse_args()

    if args.plot_only:
        run_dirs = args.run_dir
        if not run_dirs:
            run_dirs = [os.path.join(args.output_root, name) for name in os.listdir(args.output_root)]
        for run_dir in run_dirs:
            if not os.path.isdir(run_dir):
                continue
            _plot_bi_results(run_dir, os.path.basename(run_dir))
            _plot_layer_drop_results(run_dir, os.path.basename(run_dir))
            if not args.keep_artifacts:
                _cleanup_run_dir(run_dir, keep_names=["metrics", "plots"])
        return

    configs = args.config
    if not configs:
        configs = list(DEFAULT_CONFIGS.values())

    for config_path in configs:
        if not os.path.isabs(config_path):
            config_path = os.path.join(os.path.dirname(__file__), "..", config_path)
        _run_from_config(config_path, args)


if __name__ == "__main__":
    main()
