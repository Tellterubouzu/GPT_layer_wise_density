import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AdamW, AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from losses.bi_floor import bi_floor_loss, compute_bi_list
from models.bi_capture import resolve_layer_indices
from utils.data import build_dataloader
from utils.logging import setup_logger
from utils.metrics_store import MetricsStore
from utils.seed import set_seed


@dataclass
class BIFloorConfig:
    enabled: bool = False
    tau: float = 0.1
    mode: str = "hinge"
    beta: float = 20.0
    lambda_max: float = 0.0
    lambda_schedule: str = "linear_ramp"
    warmup_steps: int = 0
    ramp_steps: int = 0
    mid_start: float = 0.33
    mid_end: float = 0.67
    detach_input: bool = False
    detach_output: bool = False
    token_reduce: str = "mean"
    sample_k: int = 32
    eps: float = 1e-6
    fp32: bool = True
    log_bi_list: bool = False


def _lambda_schedule(step: int, cfg: BIFloorConfig) -> float:
    if not cfg.enabled:
        return 0.0
    if cfg.lambda_schedule == "constant":
        return cfg.lambda_max
    if cfg.lambda_schedule not in {"linear_ramp", "warmup_ramp"}:
        raise ValueError(f"Unknown lambda_schedule: {cfg.lambda_schedule}")
    if step < cfg.warmup_steps:
        return 0.0
    if cfg.ramp_steps <= 0:
        return cfg.lambda_max
    progress = (step - cfg.warmup_steps) / float(cfg.ramp_steps)
    return cfg.lambda_max * min(1.0, max(0.0, progress))


def _resolve_bifloor_cfg(raw: Dict[str, Any]) -> BIFloorConfig:
    cfg = BIFloorConfig()
    for key, value in raw.items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def _compute_bi_stats(bi_list: torch.Tensor, tau: float, log_bi_list: bool) -> Dict[str, Any]:
    if bi_list.numel() == 0:
        return {}
    below = bi_list < tau
    stats = {
        "bi_mean_mid": bi_list.mean().item(),
        "bi_min_mid": bi_list.min().item(),
        "bi_p10_mid": torch.quantile(bi_list, 0.1).item() if bi_list.numel() > 1 else bi_list.item(),
        "frac_below_tau_mid": below.float().mean().item(),
        "num_layers_below_tau_mid": int(below.sum().item()),
    }
    if log_bi_list:
        stats["bi_mid"] = [float(x) for x in bi_list.detach().cpu().tolist()]
    return stats


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

    bifloor_cfg = _resolve_bifloor_cfg(config.get("bifloor", {}))
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
            output_hidden_states=bifloor_cfg.enabled,
        )
        lm_loss = outputs.loss
        reg_lambda = _lambda_schedule(step, bifloor_cfg)
        reg_loss = torch.tensor(0.0, device=device)
        bi_list = torch.tensor([], device=device)

        if bifloor_cfg.enabled:
            hidden_states = outputs.hidden_states
            num_layers = len(hidden_states) - 1
            layers = resolve_layer_indices(num_layers, bifloor_cfg.mid_start, bifloor_cfg.mid_end)
            bi_list = compute_bi_list(
                hidden_states,
                layers,
                attention_mask=attention_mask,
                eps=bifloor_cfg.eps,
                token_reduce=bifloor_cfg.token_reduce,
                sample_k=bifloor_cfg.sample_k,
                fp32=bifloor_cfg.fp32,
                detach_input=bifloor_cfg.detach_input,
                detach_output=bifloor_cfg.detach_output,
            )
            if bi_list.numel() > 0:
                reg_loss = bi_floor_loss(bi_list, bifloor_cfg.tau, mode=bifloor_cfg.mode, beta=bifloor_cfg.beta)

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
            if bifloor_cfg.enabled:
                record.update(_compute_bi_stats(bi_list, bifloor_cfg.tau, bifloor_cfg.log_bi_list))
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
