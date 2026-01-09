import json
import math
import os
import time
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

from models.block_access import get_transformer_blocks
from models.checkpointing import save_model_state
from models.hidden_state_utils import hidden_states_to_deltas, resolve_layer_indices
from models.model_factory import build_model, build_tokenizer
from mur.optimizer_scaling import compute_layer_multipliers, scale_layer_grads
from mur.regularizer import hinge_floor, schedule_value
from mur.utility import compute_metric_values, reduce_token_values
from utils.config import save_config
from utils.data import build_dataloader
from utils.logging import setup_logger
from utils.metrics_store import MetricsStore
from utils.seed import set_seed


@dataclass
class MurConfig:
    enabled: bool = False
    mode: str = "loss"
    metric: str = "cos"
    mid_start: float = 0.33
    mid_end: float = 0.67
    tau: float = 0.0
    lambda_max: float = 0.0
    alpha: float = 0.0
    warmup_steps: int = 0
    ramp_steps: int = 0
    token_reduce: str = "mean"
    sample_k: int = 32
    eps: float = 1e-6
    fp32_dot: bool = True
    log_layer_stats: bool = False


class MurAccumulator:
    def __init__(self, num_layers: int) -> None:
        self.sums = [0.0 for _ in range(num_layers)]
        self.count = 0

    def update(self, layer_means: Iterable[float]) -> None:
        for idx, value in enumerate(layer_means):
            self.sums[idx] += float(value)
        self.count += 1

    def average(self) -> List[float]:
        if self.count == 0:
            return [0.0 for _ in self.sums]
        return [value / self.count for value in self.sums]

    def reset(self) -> None:
        self.sums = [0.0 for _ in self.sums]
        self.count = 0


def _resolve_mur_cfg(raw: Dict[str, Any]) -> MurConfig:
    cfg = MurConfig()
    for key, value in raw.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def _mur_strength(step: int, cfg: MurConfig) -> float:
    if not cfg.enabled:
        return 0.0
    if cfg.mode == "update":
        return schedule_value(step, cfg.warmup_steps, cfg.ramp_steps, cfg.alpha)
    return schedule_value(step, cfg.warmup_steps, cfg.ramp_steps, cfg.lambda_max)


def _cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def _make_grad_hook(store: Dict[int, torch.Tensor], layer_idx: int):
    def _hook(grad: torch.Tensor) -> torch.Tensor:
        store[layer_idx] = grad.detach()
        return grad

    return _hook


def _compute_layer_means(
    layer_indices: List[int],
    grad_store: Dict[int, torch.Tensor],
    delta_store: Dict[int, torch.Tensor],
    cfg: MurConfig,
) -> List[float]:
    layer_means: List[float] = []
    for idx in layer_indices:
        g = grad_store.get(idx)
        d = delta_store.get(idx)
        if g is None or d is None:
            layer_means.append(0.0)
            continue
        values = compute_metric_values(g, d, cfg.metric, cfg.eps, cfg.fp32_dot)
        mean_value = reduce_token_values(values, cfg.token_reduce, cfg.sample_k)
        layer_means.append(float(mean_value.detach().item()))
    return layer_means


def _compute_mur_loss(
    lm_loss: torch.Tensor,
    hidden_states: Iterable[torch.Tensor],
    layer_indices: List[int],
    cfg: MurConfig,
) -> tuple[torch.Tensor, List[float]]:
    losses: List[torch.Tensor] = []
    layer_means: List[float] = []
    h_out = [hidden_states[idx + 1] for idx in layer_indices]
    grads = torch.autograd.grad(lm_loss, h_out, retain_graph=True, create_graph=False, allow_unused=True)

    for layer_idx, g in zip(layer_indices, grads):
        if g is None:
            continue
        delta = hidden_states[layer_idx + 1] - hidden_states[layer_idx]
        values = compute_metric_values(g, delta, cfg.metric, cfg.eps, cfg.fp32_dot)
        mean_value = reduce_token_values(values, cfg.token_reduce, cfg.sample_k)
        losses.append(hinge_floor(mean_value, cfg.tau))
        layer_means.append(float(mean_value.detach().item()))

    if not losses:
        return torch.tensor(0.0, device=lm_loss.device), layer_means

    return torch.stack(losses).mean(), layer_means


def _layer_stats(layer_means: List[float], tau: float) -> Dict[str, float]:
    if not layer_means:
        return {"mur_mean": 0.0, "mur_frac_below_tau": 0.0}
    mean_value = sum(layer_means) / len(layer_means)
    frac_below = sum(1 for value in layer_means if value < tau) / len(layer_means)
    return {"mur_mean": mean_value, "mur_frac_below_tau": frac_below}


def _infer_run_name(config: Dict[str, Any], param_count_m: float) -> str:
    model_cfg = config.get("model", {})
    arch = model_cfg.get("arch", "gpt2")
    if arch == "transformer++":
        arch = "transformer_pp"
    variant = "mur" if config.get("mur", {}).get("enabled", False) else "baseline"
    size_tag = f"{int(round(param_count_m))}m"
    date_tag = datetime.now().strftime("%Y%m%d")
    return f"{arch}_{size_tag}_{variant}_{date_tag}"


def train(config: Dict[str, Any]) -> None:
    set_seed(config.get("seed", 42))

    tokenizer = build_tokenizer(config)
    model = build_model(config)
    model.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    raw_train = load_dataset(config["dataset_name"], config.get("dataset_config"), split=config["train_split"])
    raw_eval = load_dataset(config["dataset_name"], config.get("dataset_config"), split=config["eval_split"])

    train_loader = build_dataloader(
        raw_train,
        tokenizer,
        config["batch_size"],
        config["block_size"],
        shuffle=True,
        text_column=config.get("text_column", "text"),
    )
    eval_loader = build_dataloader(
        raw_eval,
        tokenizer,
        config["eval_batch_size"],
        config["block_size"],
        shuffle=False,
        text_column=config.get("text_column", "text"),
    )

    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=config["warmup_steps"], num_training_steps=config["num_train_steps"]
    )

    param_count = sum(p.numel() for p in model.parameters())
    param_count_m = param_count / 1e6
    run_name = _infer_run_name(config, param_count_m)
    config["project_name"] = "marginal-utility-regularization"
    config["run_name"] = run_name

    output_dir = config.get("output_dir")
    if not output_dir or output_dir == "auto":
        output_dir = os.path.join("runs", run_name)
    elif "{run_name}" in output_dir:
        output_dir = output_dir.format(run_name=run_name)
    config["output_dir"] = output_dir
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger("train", os.path.join(output_dir, "train.log"))
    save_config(config, os.path.join(output_dir, "config.json"))

    with open(os.path.join(output_dir, "model_params.json"), "w", encoding="utf-8") as f:
        json.dump({"param_count": param_count, "param_count_m": param_count_m}, f, indent=2, ensure_ascii=True)

    model_cfg = config.get("model")
    if model_cfg and model_cfg.get("vocab_size") and model_cfg.get("vocab_size") != len(tokenizer):
        logger.warning(
            "Tokenizer vocab size (%d) does not match model vocab size (%d)",
            len(tokenizer),
            model_cfg.get("vocab_size"),
        )

    mur_cfg = _resolve_mur_cfg(config.get("mur", {}))
    train_metrics = MetricsStore(os.path.join(output_dir, "metrics", "train.jsonl"))
    eval_metrics = MetricsStore(os.path.join(output_dir, "metrics", "eval.jsonl"))

    train_iter = _cycle(train_loader)
    num_steps = config["num_train_steps"]
    grad_accum = config.get("gradient_accumulation_steps", 1)

    layer_indices: Optional[List[int]] = None
    mur_accum: Optional[MurAccumulator] = None
    blocks: Optional[List[torch.nn.Module]] = None

    seen_tokens = 0
    log_tokens = 0
    log_loss_sum = 0.0
    log_steps = 0
    last_log_time = time.time()

    logger.info("Starting training")

    optimizer.zero_grad()
    for step in range(1, num_steps + 1):
        batch = next(train_iter)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            output_hidden_states=mur_cfg.enabled,
        )
        lm_loss = outputs.loss
        mur_weight = _mur_strength(step, mur_cfg)

        mur_loss = torch.tensor(0.0, device=device)
        layer_means: List[float] = []
        handles: List[torch.utils.hooks.RemovableHandle] = []
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
                mur_loss, layer_means = _compute_mur_loss(lm_loss, hidden_states, layer_indices, mur_cfg)
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
        total_loss = total_loss / grad_accum
        total_loss.backward()

        if mur_cfg.enabled and mur_cfg.mode == "update" and mur_weight > 0.0:
            for handle in handles:
                handle.remove()
            if layer_indices is None or mur_accum is None:
                raise ValueError("MUR update mode missing layer indices")
            layer_means = _compute_layer_means(layer_indices, grad_store, delta_store, mur_cfg)
            mur_accum.update(layer_means)

        if step % grad_accum == 0:
            if mur_cfg.enabled and mur_cfg.mode == "update" and mur_weight > 0.0:
                if layer_indices is None or mur_accum is None or blocks is None:
                    raise ValueError("MUR update mode missing state")
                avg_means = mur_accum.average()
                multipliers = compute_layer_multipliers(layer_indices, avg_means, mur_cfg.tau, mur_weight)
                scale_layer_grads(blocks, multipliers)
                mur_accum.reset()

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 1.0))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        tokens_in_step = input_ids.numel()
        seen_tokens += tokens_in_step
        log_tokens += tokens_in_step
        log_loss_sum += lm_loss.item()
        log_steps += 1

        if step % config["log_steps"] == 0:
            elapsed = max(1e-8, time.time() - last_log_time)
            train_loss = log_loss_sum / max(1, log_steps)
            stats = _layer_stats(layer_means, mur_cfg.tau) if mur_cfg.enabled else {}
            record = {
                "project_name": config["project_name"],
                "run_name": config["run_name"],
                "step": step,
                "lm_loss": lm_loss.item(),
                "train_loss": train_loss,
                "train_perplexity": math.exp(train_loss),
                "token_per_sec": log_tokens / elapsed,
                "VRAM_allocated_bytes": torch.cuda.memory_allocated(device) if device.type == "cuda" else 0,
                "seened_tokens": seen_tokens,
                "current_lr": scheduler.get_last_lr()[0],
                "mur_loss": mur_loss.item(),
                "mur_weight": mur_weight,
                "total_loss": total_loss.item() * grad_accum,
                "lr": scheduler.get_last_lr()[0],
                "mur_mode": mur_cfg.mode if mur_cfg.enabled else "disabled",
            }
            record.update(stats)
            if mur_cfg.enabled and mur_cfg.log_layer_stats:
                record["mur_layer_means"] = layer_means
            train_metrics.write(record)
            logger.info(json.dumps(record, ensure_ascii=True))
            last_log_time = time.time()
            log_tokens = 0
            log_loss_sum = 0.0
            log_steps = 0

        if step % config["eval_steps"] == 0:
            eval_loss = evaluate(model, eval_loader, device)
            eval_record = {"step": step, "eval_loss": eval_loss, "eval_ppl": math.exp(eval_loss)}
            eval_metrics.write(eval_record)
            logger.info(json.dumps(eval_record, ensure_ascii=True))

        if step % config["save_steps"] == 0 or step == num_steps:
            save_model_state(model, output_dir)
            tokenizer.save_pretrained(output_dir)

    logger.info("Training completed")


def evaluate(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()
    model.train()
    return total_loss / max(1, total_tokens)
