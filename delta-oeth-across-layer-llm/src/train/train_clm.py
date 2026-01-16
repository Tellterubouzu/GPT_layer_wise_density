import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AdamW, AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from losses.delta_orth import delta_orth_loss
from models.delta_capture import adjacent_pairs, hidden_states_to_deltas, resolve_layer_indices
from utils.data import build_dataloader
from utils.logging import setup_logger
from utils.metrics_store import MetricsStore
from utils.seed import set_seed


@dataclass
class DeltaOrthConfig:
    enabled: bool = False
    lambda_max: float = 0.0
    lambda_schedule: str = "linear_ramp"
    warmup_steps: int = 0
    ramp_steps: int = 0
    mid_start: float = 0.25
    mid_end: float = 0.75
    detach_prev: bool = False
    token_reduce: str = "mean"
    sample_k: int = 32
    eps: float = 1e-6


def _lambda_schedule(step: int, cfg: DeltaOrthConfig) -> float:
    if not cfg.enabled:
        return 0.0
    if cfg.lambda_schedule == "constant":
        return cfg.lambda_max
    if step < cfg.warmup_steps:
        return 0.0
    if cfg.ramp_steps <= 0:
        return cfg.lambda_max
    progress = (step - cfg.warmup_steps) / float(cfg.ramp_steps)
    return cfg.lambda_max * min(1.0, max(0.0, progress))


def _resolve_delta_cfg(raw: Dict[str, Any]) -> DeltaOrthConfig:
    cfg = DeltaOrthConfig()
    for key, value in raw.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def _compute_delta_loss(hidden_states: Iterable[torch.Tensor], cfg: DeltaOrthConfig) -> torch.Tensor:
    deltas = hidden_states_to_deltas(hidden_states)
    layers = resolve_layer_indices(len(deltas), cfg.mid_start, cfg.mid_end)
    pairs = adjacent_pairs(layers)
    if not pairs:
        return torch.tensor(0.0, device=deltas[0].device)
    losses = []
    for idx, prev_idx in pairs:
        losses.append(
            delta_orth_loss(
                deltas[idx],
                deltas[prev_idx],
                eps=cfg.eps,
                detach_prev=cfg.detach_prev,
                reduce_tokens=cfg.token_reduce,
                sample_k=cfg.sample_k,
            )
        )
    return torch.stack(losses).mean()


def _cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def train(config: Dict[str, Any]) -> None:
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger("train", os.path.join(output_dir, "train.log"))
    set_seed(config.get("seed", 42))

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config["model_name"])
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

    delta_cfg = _resolve_delta_cfg(config.get("delta_orth", {}))
    train_metrics = MetricsStore(os.path.join(output_dir, "metrics", "train.jsonl"))
    eval_metrics = MetricsStore(os.path.join(output_dir, "metrics", "eval.jsonl"))

    train_iter = _cycle(train_loader)
    num_steps = config["num_train_steps"]
    grad_accum = config.get("gradient_accumulation_steps", 1)

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
            output_hidden_states=delta_cfg.enabled,
        )
        lm_loss = outputs.loss
        reg_lambda = _lambda_schedule(step, delta_cfg)
        reg_loss = torch.tensor(0.0, device=device)

        if delta_cfg.enabled and reg_lambda > 0.0:
            reg_loss = _compute_delta_loss(outputs.hidden_states, delta_cfg)

        loss = lm_loss + reg_lambda * reg_loss
        loss = loss / grad_accum
        loss.backward()

        if step % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 1.0))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % config["log_steps"] == 0:
            record = {
                "step": step,
                "lm_loss": lm_loss.item(),
                "reg_loss": reg_loss.item(),
                "reg_lambda": reg_lambda,
                "total_loss": loss.item() * grad_accum,
                "lr": scheduler.get_last_lr()[0],
            }
            train_metrics.write(record)
            logger.info(json.dumps(record, ensure_ascii=True))

        if step % config["eval_steps"] == 0:
            eval_loss = evaluate(model, eval_loader, device)
            eval_record = {"step": step, "eval_loss": eval_loss, "eval_ppl": math.exp(eval_loss)}
            eval_metrics.write(eval_record)
            logger.info(json.dumps(eval_record, ensure_ascii=True))

        if step % config["save_steps"] == 0 or step == num_steps:
            model.save_pretrained(output_dir)
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
