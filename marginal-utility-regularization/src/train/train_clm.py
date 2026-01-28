import json
import math
import os
import random
import time
from contextlib import nullcontext
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.distributed as dist
import wandb
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from torch.optim import AdamW
from models.block_access import get_transformer_blocks
from models.checkpointing import save_model_state
from models.hidden_state_utils import hidden_states_to_deltas, resolve_layer_indices
from models.model_factory import build_model, build_tokenizer
from mur.optimizer_scaling import compute_layer_multipliers, scale_layer_grads
from mur.regularizer import hinge_floor, schedule_value
from mur.utility import compute_metric_values, reduce_token_values
from utils.config import save_config
from utils.data import build_streaming_dataloader
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


@dataclass
class FreezeConfig:
    enabled: bool = False
    mode: str = "random_near_input"
    layer_idx: Optional[int] = None
    near_input_ratio: float = 0.25
    near_input_layers: Optional[int] = None
    seed: Optional[int] = None


@dataclass
class HierarchicalFreezeConfig:
    enabled: bool = False
    mode: str = "input"
    unfreeze_tokens: int = 0
    seed: Optional[int] = None


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


def _resolve_freeze_cfg(raw: Dict[str, Any]) -> FreezeConfig:
    cfg = FreezeConfig()
    for key, value in raw.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def _resolve_hierarchical_freeze_cfg(raw: Dict[str, Any]) -> HierarchicalFreezeConfig:
    cfg = HierarchicalFreezeConfig()
    for key, value in raw.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def _build_unfreeze_order(num_layers: int, cfg: HierarchicalFreezeConfig, seed: int) -> List[int]:
    order = list(range(num_layers))
    if cfg.mode == "input":
        return order
    if cfg.mode == "output":
        return list(reversed(order))
    if cfg.mode == "random":
        rng = random.Random(seed)
        rng.shuffle(order)
        return order
    raise ValueError(f"Unknown hierarchical freeze mode: {cfg.mode}")


def _apply_hierarchical_freeze(
    model: torch.nn.Module,
    config: Dict[str, Any],
    cfg: HierarchicalFreezeConfig,
) -> tuple[List[torch.nn.Module], List[int]]:
    blocks = get_transformer_blocks(model)
    for block in blocks:
        for param in block.parameters():
            param.requires_grad = False
    seed = cfg.seed if cfg.seed is not None else int(config.get("seed", 42))
    order = _build_unfreeze_order(len(blocks), cfg, seed)
    return blocks, order


def _resolve_freeze_candidates(num_layers: int, cfg: FreezeConfig) -> List[int]:
    if num_layers <= 0:
        return []
    if cfg.near_input_layers is not None:
        count = int(cfg.near_input_layers)
    else:
        count = int(round(num_layers * float(cfg.near_input_ratio)))
    count = max(1, min(num_layers, count))
    return list(range(count))


def _select_freeze_layer(num_layers: int, cfg: FreezeConfig, seed: int) -> tuple[int, List[int]]:
    if num_layers <= 0:
        raise ValueError("No layers available to freeze")
    if cfg.mode == "fixed":
        if cfg.layer_idx is None:
            raise ValueError("freeze.layer_idx must be set when freeze.mode is fixed")
        if cfg.layer_idx < 0 or cfg.layer_idx >= num_layers:
            raise ValueError(f"freeze.layer_idx {cfg.layer_idx} out of range for {num_layers} layers")
        return cfg.layer_idx, [cfg.layer_idx]
    rng = random.Random(seed)
    if cfg.mode == "random":
        return rng.randrange(num_layers), list(range(num_layers))
    if cfg.mode == "random_near_input":
        candidates = _resolve_freeze_candidates(num_layers, cfg)
        return rng.choice(candidates), candidates
    raise ValueError(f"Unknown freeze mode: {cfg.mode}")


def _apply_freeze(
    model: torch.nn.Module,
    config: Dict[str, Any],
    cfg: FreezeConfig,
) -> tuple[Optional[int], Optional[List[int]]]:
    if not cfg.enabled:
        return None, None
    blocks = get_transformer_blocks(model)
    num_layers = len(blocks)
    seed = cfg.seed if cfg.seed is not None else int(config.get("seed", 42))
    layer_idx, candidates = _select_freeze_layer(num_layers, cfg, seed)
    for param in blocks[layer_idx].parameters():
        param.requires_grad = False
    return layer_idx, candidates


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


def _format_layer_label(mid_start: float, mid_end: float) -> str:
    if 0.0 < mid_start <= 1.0 and 0.0 < mid_end <= 1.0:
        start_pct = int(mid_start * 100)
        end_pct = int(mid_end * 100) - 1
        if end_pct < start_pct:
            end_pct = start_pct
        return f"{start_pct}_{end_pct}"
    return f"{int(mid_start)}_{int(mid_end)}"


def _infer_run_name(config: Dict[str, Any], param_count_m: float) -> str:
    model_cfg = config.get("model", {})
    arch = model_cfg.get("arch", "gpt2")
    if arch == "transformer++":
        arch = "transformer_pp"
    size_tag = f"{int(round(param_count_m))}m"
    date_tag = datetime.now().strftime("%Y%m%d")
    mur_cfg = config.get("mur", {})
    freeze_cfg = config.get("freeze", {})
    freeze_tag = None
    if freeze_cfg.get("enabled", False):
        layer_idx = freeze_cfg.get("selected_layer")
        if layer_idx is None:
            layer_idx = freeze_cfg.get("layer_idx")
        freeze_tag = f"freeze_layer{layer_idx}" if layer_idx is not None else "freeze"
    hfreeze_cfg = config.get("hierarchical_freeze", {})
    hfreeze_tag = None
    if hfreeze_cfg.get("enabled", False):
        mode = hfreeze_cfg.get("mode", "input")
        unfreeze_tokens = hfreeze_cfg.get("unfreeze_tokens")
        token_tag = f"t{int(unfreeze_tokens)}" if unfreeze_tokens is not None else None
        if token_tag:
            hfreeze_tag = f"hfreeze_{mode}_{token_tag}"
        else:
            hfreeze_tag = f"hfreeze_{mode}"
    if mur_cfg.get("enabled", False):
        mid_start = mur_cfg.get("mid_start", 0.33)
        mid_end = mur_cfg.get("mid_end", 0.67)
        layer_label = _format_layer_label(mid_start, mid_end)
        base_name = f"{arch}_{size_tag}_mur_layer{layer_label}"
    else:
        base_name = f"{arch}_{size_tag}_baseline"
    if freeze_tag:
        base_name = f"{base_name}_{freeze_tag}"
    if hfreeze_tag:
        base_name = f"{base_name}_{hfreeze_tag}"
    return f"{base_name}_{date_tag}"


def train(config: Dict[str, Any]) -> None:
    set_seed(config.get("seed", 42))

    tokenizer = build_tokenizer(config)
    model = build_model(config)
    model.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    use_bf16 = config.get("bf16", True)
    if use_bf16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 requested but CUDA device does not support bf16")
    if use_bf16:
        model.to(dtype=torch.bfloat16)

    hfreeze_cfg = _resolve_hierarchical_freeze_cfg(config.get("hierarchical_freeze", {}))
    hfreeze_blocks: Optional[List[torch.nn.Module]] = None
    hfreeze_order: Optional[List[int]] = None
    hfreeze_next_idx = 0
    hfreeze_interval: Optional[int] = None
    hfreeze_total_layers = 0
    hfreeze_total_tokens = 0

    freeze_cfg = _resolve_freeze_cfg(config.get("freeze", {}))
    freeze_layer: Optional[int] = None
    freeze_candidates: Optional[List[int]] = None

    if hfreeze_cfg.enabled:
        config.setdefault("hierarchical_freeze", {})
        hfreeze_blocks, hfreeze_order = _apply_hierarchical_freeze(model, config, hfreeze_cfg)
        hfreeze_total_layers = len(hfreeze_order)
        hfreeze_total_tokens = max(0, int(hfreeze_cfg.unfreeze_tokens))
        config["hierarchical_freeze"]["order"] = hfreeze_order
        if hfreeze_total_tokens == 0 and hfreeze_blocks:
            for idx in hfreeze_order:
                for param in hfreeze_blocks[idx].parameters():
                    param.requires_grad = True
            hfreeze_next_idx = hfreeze_total_layers
        else:
            hfreeze_interval = max(1, int(math.ceil(hfreeze_total_tokens / max(1, hfreeze_total_layers))))
    elif freeze_cfg.enabled:
        config.setdefault("freeze", {})
        freeze_layer, freeze_candidates = _apply_freeze(model, config, freeze_cfg)
        config["freeze"]["selected_layer"] = freeze_layer
        config["freeze"]["candidate_layers"] = freeze_candidates

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

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if hfreeze_cfg.enabled:
        optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    else:
        optimizer = AdamW(trainable_params, lr=config["learning_rate"], weight_decay=config["weight_decay"])

    param_count = sum(p.numel() for p in model.parameters())
    param_count_m = param_count / 1e6
    trainable_param_count = sum(p.numel() for p in trainable_params)
    trainable_param_count_m = trainable_param_count / 1e6
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
        json.dump(
            {
                "param_count": param_count,
                "param_count_m": param_count_m,
                "trainable_param_count": trainable_param_count,
                "trainable_param_count_m": trainable_param_count_m,
            },
            f,
            indent=2,
            ensure_ascii=True,
        )

    model_cfg = config.get("model")
    if model_cfg and model_cfg.get("vocab_size") and model_cfg.get("vocab_size") != len(tokenizer):
        logger.warning(
            "Tokenizer vocab size (%d) does not match model vocab size (%d)",
            len(tokenizer),
            model_cfg.get("vocab_size"),
        )

    mur_cfg = _resolve_mur_cfg(config.get("mur", {}))
    train_metrics = MetricsStore(os.path.join(output_dir, "metrics", "train.jsonl"))
    train_iter = _cycle(train_loader)
    num_steps = config.get("num_train_steps", 0)
    grad_accum = config.get("gradient_accumulation_steps", 1)
    max_train_tokens = config.get("max_train_tokens")
    warmup_ratio = float(config.get("warmup_ratio", 0.1))
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    tokens_per_optimizer_step = config["batch_size"] * config["seq_len"] * grad_accum * world_size
    if max_train_tokens is not None:
        total_steps = max(1, math.ceil(max_train_tokens / tokens_per_optimizer_step))
    else:
        total_steps = max(1, int(num_steps))
    warmup_steps = max(0, int(total_steps * warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    wandb_enabled = config.get("wandb_enabled", True)
    if wandb_enabled:
        wandb_name = config["run_name"]
        mur_cfg_dict = config.get("mur", {})
        if mur_cfg_dict.get("enabled", False) and "mur_layer" not in wandb_name:
            mid_start = mur_cfg_dict.get("mid_start", 0.33)
            mid_end = mur_cfg_dict.get("mid_end", 0.67)
            layer_label = _format_layer_label(mid_start, mid_end)
            if "_mur_" in wandb_name:
                wandb_name = wandb_name.replace("_mur_", f"_mur_layer{layer_label}_")
            else:
                wandb_name = f"{wandb_name}_mur_layer{layer_label}"
        wandb.init(project=config["project_name"], name=wandb_name, config=config)
        wandb.log(
            {
                "param_count": param_count,
                "param_count_m": param_count_m,
                "trainable_param_count": trainable_param_count,
                "trainable_param_count_m": trainable_param_count_m,
                "total_steps": total_steps,
                "warmup_steps": warmup_steps,
                "freeze_layer": freeze_layer,
                "hierarchical_freeze_unfrozen_layers": hfreeze_next_idx,
                "hierarchical_freeze_total_layers": hfreeze_total_layers,
                "hierarchical_freeze_total_tokens": hfreeze_total_tokens,
            },
            step=0,
        )

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
    stopped_early = False
    step = 0
    while True:
        if max_train_tokens is not None and seen_tokens >= max_train_tokens:
            stopped_early = True
            break
        if max_train_tokens is None and num_steps and step >= num_steps:
            break
        if max_train_tokens is not None and step >= total_steps:
            stopped_early = True
            break
        step += 1
        batch = next(train_iter)
        input_ids = batch["input_ids"].to(device)
        autocast_ctx = (
            torch.autocast(device_type=device.type, dtype=torch.bfloat16) if use_bf16 else nullcontext()
        )
        with autocast_ctx:
            outputs = model(
                input_ids=input_ids,
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

        tokens_in_step = input_ids.numel() * world_size
        seen_tokens += tokens_in_step
        log_tokens += tokens_in_step
        log_loss_sum += lm_loss.item()
        log_steps += 1

        if (
            hfreeze_cfg.enabled
            and hfreeze_blocks is not None
            and hfreeze_order is not None
            and hfreeze_interval is not None
            and hfreeze_next_idx < hfreeze_total_layers
        ):
            while hfreeze_next_idx < hfreeze_total_layers:
                threshold = (hfreeze_next_idx + 1) * hfreeze_interval
                if seen_tokens < threshold:
                    break
                layer_idx = hfreeze_order[hfreeze_next_idx]
                for param in hfreeze_blocks[layer_idx].parameters():
                    param.requires_grad = True
                hfreeze_next_idx += 1

        if max_train_tokens is not None and seen_tokens >= max_train_tokens:
            stopped_early = True

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
                "seened_token_global": seen_tokens,
                "current_lr": scheduler.get_last_lr()[0],
                "mur_loss": mur_loss.item(),
                "mur_weight": mur_weight,
                "total_loss": total_loss.item() * grad_accum,
                "lr": scheduler.get_last_lr()[0],
                "mur_mode": mur_cfg.mode if mur_cfg.enabled else "disabled",
                "freeze_layer": freeze_layer,
                "hierarchical_freeze_unfrozen_layers": hfreeze_next_idx,
                "hierarchical_freeze_total_layers": hfreeze_total_layers,
                "hierarchical_freeze_next_layer": (
                    hfreeze_order[hfreeze_next_idx] if hfreeze_order and hfreeze_next_idx < hfreeze_total_layers else None
                ),
            }
            record.update(stats)
            if mur_cfg.enabled and mur_cfg.log_layer_stats:
                record["mur_layer_means"] = layer_means
            train_metrics.write(record)
            logger.info(json.dumps(record, ensure_ascii=True))
            if wandb_enabled:
                wandb_record = {k: v for k, v in record.items() if not isinstance(v, list)}
                wandb.log(wandb_record, step=step)
            last_log_time = time.time()
            log_tokens = 0
            log_loss_sum = 0.0
            log_steps = 0

        if step % config["save_steps"] == 0 or (max_train_tokens is None and step == num_steps):
            save_model_state(model, output_dir)
            tokenizer.save_pretrained(output_dir)

        if stopped_early:
            break

    if stopped_early:
        save_model_state(model, output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("Training completed (stopped at token limit)")
    else:
        logger.info("Training completed")
    if wandb_enabled:
        wandb.finish()


def evaluate(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            outputs = model(input_ids=input_ids, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item() * input_ids.numel()
            total_tokens += input_ids.numel()
    model.train()
    return total_loss / max(1, total_tokens)
