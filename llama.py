#!/usr/bin/env python3
"""
Without DDP
python train_llama_ddp.py \
  --dataset_path /nvme/fineweb_tokens \
  --seq_len 2048 \
  --local_batch_size 32 \
  --total_tokens 15e9 \
  --wandb_project llama-scratch-pretrain \
  --api_file api.txt
Multiple GPU(With DDP)
torchrun --standalone --nproc_per_node=4 train_llama_ddp.py \
  --dataset_path /nvme/fineweb_tokens \
  --seq_len 2048 \
  --local_batch_size 32 \
  --total_tokens 15e9 \
  --wandb_project llama-scratch-pretrain \
  --api_file api.txt

"""


import argparse
import os
import math
import datetime
import time
import json
from pathlib import Path
from itertools import islice

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import torch.distributed as dist

import wandb
from datasets import load_dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from huggingface_hub import HfApi, upload_folder
from contextlib import nullcontext

try:
    import deepspeed
    _HAS_DS = True
except ImportError:
    _HAS_DS = False

# ---------- Constants ----------
MAGIC_NUMBER = 20240520
HEADER_INT32  = 256  # 1 KiB header = 256 * int32
VERSION = 1
HEADER_U16 = HEADER_INT32 *2
class BinShardsDataset(IterableDataset):
    """Load pre‑tokenised uint16 shards with zero‑copy mmap."""

    def __init__(self, shard_dir: str | Path, seq_len: int):
        self.seq_len = seq_len
        self.files = sorted(Path(shard_dir).glob("shard_*.bin"))
        if not self.files:
            raise FileNotFoundError(f"no shard_*.bin in {shard_dir}")
        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

    def _load_uint16_tokens(self, file: Path) -> torch.Tensor:
        header = torch.from_file(str(file), dtype=torch.int32, size=HEADER_INT32)
        if header[0].item() != MAGIC_NUMBER or header[1].item() != VERSION:
            raise ValueError(f"bad header in {file}")
        num_tok = int(header[2].item())
        tot_u16 = HEADER_U16 + num_tok
        mapped = torch.from_file(str(file), dtype=torch.uint16, size=tot_u16)
        return mapped[HEADER_U16:]

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        num_workers = worker.num_workers if worker else 1
        
        global_idx = self.rank * num_workers + worker_id
        global_stride = self.world_size * num_workers

        for i, f in enumerate(self.files):
            if i% global_stride != global_idx:
                continue
            toks = self._load_uint16_tokens(f)
            for j in range(0, len(toks) - self.seq_len + 1, self.seq_len):
                yield toks[j : j + self.seq_len].long()


def load_api_keys(path):
    """Read key=value pairs from api.txt and return dict."""
    keys = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                keys[k.strip()] = v.strip()
    return keys


def prepare_hf_repo_id(write_token, explicit_repo, default_prefix="gpt2-scratch"):
    """Helper to build repo id if not explicitly given."""
    api = HfApi(token=write_token)
    if explicit_repo:
        return explicit_repo, api
    me = api.whoami()
    owner = me["name"]
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    repo_id = f"{owner}/{default_prefix}-{timestamp}"
    return repo_id, api

# ---------- LLaMA-like model definitions (RMSNorm + SwiGLU + RoPE + Pre-Norm + GQA) ----------

class LlamaConfig:
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        max_position_embeddings: int,
        num_key_value_heads: int | None = None,   # GQA
        intermediate_size: int | None = None,
        multiple_of: int = 256,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = True,
    ):
        assert hidden_size % num_attention_heads == 0
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.max_position_embeddings = max_position_embeddings

        # GQA
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        assert self.num_attention_heads % self.num_key_value_heads == 0, \
            "num_attention_heads must be divisible by num_key_value_heads"

        # MLP size (LLaMA-style) + rounding
        self.multiple_of = multiple_of
        if intermediate_size is None:
            # LLaMA系でよくある: int( (2/3) * 4 * hidden )
            intermediate_size = int((2 * 4 * hidden_size) / 3)
            intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)
        else:
            intermediate_size = multiple_of * ((intermediate_size + multiple_of - 1) // multiple_of)
        self.intermediate_size = intermediate_size

        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.initializer_range = initializer_range
        self.tie_word_embeddings = tie_word_embeddings

        # Dropoutは完全に使わない（固定0相当）
        self.attention_dropout = 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return x * self.weight


class RotaryEmbedding(nn.Module):
    """
    RoPE (LLaMA系の典型実装)
    """
    def __init__(self, dim: int, max_position_embeddings: int, base: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "RoPE dim must be even"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.max_seq_len_cached = 0
        self.register_buffer("cos_cached", torch.empty(1), persistent=False)
        self.register_buffer("sin_cached", torch.empty(1), persistent=False)
        self._set_cache(max_position_embeddings, device=inv_freq.device, dtype=torch.float32)

    def _set_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)   # (T, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)             # (T, dim)
        cos = emb.cos().to(dtype)
        sin = emb.sin().to(dtype)
        self.cos_cached = cos[None, None, :, :]             # (1,1,T,dim)
        self.sin_cached = sin[None, None, :, :]

    def forward(self, x: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if (seq_len > self.max_seq_len_cached) or (self.cos_cached.device != x.device):
            self._set_cache(seq_len, device=x.device, dtype=torch.float32)
        cos = self.cos_cached[:, :, :seq_len, :].to(dtype=x.dtype, device=x.device)
        sin = self.sin_cached[:, :, :seq_len, :].to(dtype=x.dtype, device=x.device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    (B, H_kv, T, D) -> (B, H_kv*n_rep, T, D)
    """
    if n_rep == 1:
        return x
    B, H_kv, T, D = x.shape
    x = x[:, :, None, :, :].expand(B, H_kv, n_rep, T, D)
    return x.reshape(B, H_kv * n_rep, T, D)


class LlamaAttention(nn.Module):
    """
    GQA対応 self-attention
      - Q: num_attention_heads
      - K,V: num_key_value_heads
      - 最後にrepeat_kvして heads を合わせてから SDPA
    Dropoutは完全に0。
    """
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.n_rep = self.n_heads // self.n_kv_heads

        # LLaMA系: bias=False
        self.q_proj = nn.Linear(self.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim, config.max_position_embeddings, base=config.rope_theta)

        # Dropout完全0固定
        self.attn_dropout_p = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)       # (B,H,T,D)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)    # (B,Hkv,T,D)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)    # (B,Hkv,T,D)

        cos, sin = self.rotary_emb(q, seq_len=T)  # (1,1,T,D)
        q = apply_rotary_pos_emb(q, cos, sin)
        k = apply_rotary_pos_emb(k, cos, sin)

        # GQA: KV heads を Q heads に合わせる
        k = repeat_kv(k, self.n_rep)  # (B,H,T,D)
        v = repeat_kv(v, self.n_rep)  # (B,H,T,D)

        # SDPA（is_causal=True）: dropout_p=0.0で固定
        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True
            )  # (B,H,T,D)
        else:
            # fallback（古いPyTorch向け）
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,T,T)
            causal = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            scores = scores.masked_fill(~causal.view(1, 1, T, T), torch.finfo(scores.dtype).min)
            probs = torch.softmax(scores, dim=-1)
            y = probs @ v  # (B,H,T,D)

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(y)


class LlamaMLP(nn.Module):
    """
    SwiGLU: down( SiLU(gate(x)) * up(x) )
    Dropoutなし。
    """
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    """
    LLaMA系 Pre-Norm:
      x = x + Attn(RMSNorm(x))
      x = x + MLP(RMSNorm(x))
    """
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = LlamaAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = LlamaMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class LlamaForCausalLM(nn.Module):
    """
    Embedding dropoutなし。最終normあり（RMSNorm）。
    """
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def forward(self, input_ids: torch.LongTensor, labels: torch.LongTensor = None):
        x = self.embed_tokens(input_ids)  # (B,T,C)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1)
            )
        return logits, loss


@torch.no_grad()
def sample_sequence(
    model,
    input_ids: torch.LongTensor,
    max_new_tokens: int = 20,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    min_p: float = 0.0,
    device: torch.device = None,
):
    """Greedy/Top-k/Top-p sampling util."""
    device = device or next(model.parameters()).device
    generated = input_ids.to(device)

    for _ in range(max_new_tokens):
        outputs = model(generated)
        logits = outputs[0]
        next_logits = logits[:, -1, :] / max(temperature, 1e-8)

        if top_k > 0:
            values, _ = torch.topk(next_logits, top_k)
            min_value = values[:, -1].unsqueeze(-1)
            next_logits = torch.where(next_logits < min_value, torch.full_like(next_logits, float('-inf')), next_logits)

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            mask = cumulative_probs > top_p
            mask[..., 1:] = mask[..., :-1].clone()
            mask[..., 0] = False
            sorted_logits[mask] = float('-inf')
            next_logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        if min_p > 0.0:
            probs = F.softmax(next_logits, dim=-1)
            next_logits = torch.where(probs < min_p, torch.full_like(next_logits, float('-inf')), next_logits)

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_token], dim=1)

    return generated

class StreamingDataset(IterableDataset):
    def __init__(self, dataset_path, split, tokenizer, seq_len):
        self.ds = load_dataset(dataset_path, split=split, streaming=True).shuffle(buffer_size=10, seed=42)# 10_000_000
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

    def __iter__(self):
        worker = get_worker_info()
        worker_id, num_workers = (worker.id, worker.num_workers) if worker else (0, 1)
        buf = []
        for idx, ex in enumerate(self.ds):
            global_idx = self.rank * num_workers + worker_id
            global_stride = self.world_size * num_workers
            if idx % global_stride != global_idx:
                continue
            toks = self.tokenizer(ex["text"], return_attention_mask=False, add_special_tokens=True)["input_ids"]
            buf.extend(toks)
            while len(buf) >= self.seq_len:
                yield torch.tensor(buf[: self.seq_len], dtype=torch.long)
                buf = buf[self.seq_len:]

def get_validation_blocks(hf_dataset, tokenizer, seq_len, max_blocks=100):
    blocks = []
    buffer = []
    for sample in hf_dataset:
        text = sample.get("text", "")
        token_ids = tokenizer.encode(text, add_special_tokens=True)
        buffer.extend(token_ids)
        while len(buffer) >= seq_len and len(blocks) < max_blocks:
            block = buffer[:seq_len]
            blocks.append({
                "input_ids": torch.tensor(block, dtype=torch.long),
                "labels": torch.tensor(block, dtype=torch.long)
            })
            buffer = buffer[seq_len:]
        if len(blocks) >= max_blocks:
            break
    return blocks


def safe_decode(token_list, tokenizer):
    try:
        return tokenizer.decode(token_list, skip_special_tokens=True)
    except Exception:
        s = "".join([chr(max(32, t)) for t in token_list])
        return s.encode("utf-8", "replace").decode("utf-8")

# ---------- Data loader helper ----------

def get_train_loader(tokenizer_path: str, path: str, seq_len: int, batch_size: int):
    if Path(path).is_dir():
        ds = BinShardsDataset(path, seq_len)
        print(f"[data] BinShardsDataset with {len(list(Path(path).glob('shard_*.bin')))} files")
        tot = len(list(Path(path).glob('shard_*.bin')))
        per_rank = (tot + ds.world_size - 1) // ds.world_size
        print(f"[data] BinShardsDataset: {tot} shards → {per_rank} / rank (world={ds.world_size})")

    else:
        tok = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        ds = StreamingDataset(path, "train", tok, seq_len)  # original tokenizer‑on‑the‑fly dataset
        print("[data] HF streaming dataset", path)
    return DataLoader(ds, batch_size=batch_size, num_workers=2 if Path(path).is_dir() else 8,
                      pin_memory=True, drop_last=True)


# ---------- Evaluation ----------
@torch.no_grad()
def compute_mean_so_far_ppl(model, blocks, device, ks):
    """
    model   : GPT2LMHeadModel or deepspeed モデル
    blocks  : list of {"input_ids": Tensor[T], "labels": Tensor[T]}
    device  : torch.device
    ks      : list of int (計測したいトークン長のリスト)
    returns : {k: {"mean_nll": float, "mean_ppl": float}}
    """
    sum_logprob = {k: 0.0 for k in ks}
    token_count = {k: 0 for k in ks}

    model.eval()
    for block in blocks:
        ids    = block["input_ids"].to(device).unsqueeze(0)   # (1, T)
        labels = block["labels"].to(device).unsqueeze(0)     # (1, T)
        logits, _ = model(ids)                               # (1, T, V)
        log_probs = F.log_softmax(logits, dim=-1)            # (1, T, V)

        # shift して位置 i の log P を取得
        #  predict for token i → log_probs[0, i-1, label_i]
        lp = log_probs[0, :-1, :]                            # predict positions 1..T-1
        lbl = labels[0, 1:]                                  # true tokens at 1..T-1
        lp_i = lm.gather(2, lbl.unsqueeze(-1)).squeeze(-1)   # (T-1,) vector
 
        T = lp_i.size(1)
        for k in ks:
            k_trunc = min(k, T)
            sum_logprob[k] += lp_i[:k_trunc].sum().item()
            token_count[k] += k_trunc

    # 平均 NLL → PPL に
    results = {}
    for k in ks:
        mean_nll = - sum_logprob[k] / token_count[k]
        results[k] = {
            "mean_nll": mean_nll,
            "mean_ppl": math.exp(mean_nll)
        }
    return results

# ---------- main() skeleton ----------
def parse_args():
    parser = argparse.ArgumentParser(description="GPT2 scratch pretrain with optional DeepSpeed and CLI-configurable parameters")
    # Distributed and DeepSpeed options
    parser.add_argument("--local_rank", type=int, default=int(os.environ.get("LOCAL_RANK", 0)), help="Local rank for distributed training")
    parser.add_argument("--deepspeed_config", type=str, default="ds_config.json", help="Path to DeepSpeed config file")
    parser.add_argument("--use_deepspeed", action="store_true", help="Enable DeepSpeed if available and GPUs > 1")

    # Training parameters
    #parser.add_argument("--local_batch_size", type=int, default=4, help="Micro batch size per GPU")
    parser.add_argument("--local_batch_size", type=int, default=150, help="Micro batch size per GPU")
    parser.add_argument("--use_gpu_amount", type=int, default=max(torch.cuda.device_count(), 1), help="Total number of GPUs for global batch calculation, if use cpu, set to 1")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate")
    parser.add_argument("--validate_every_steps", type=int, default=200, help="Validate every N steps")
    parser.add_argument("--save_checkpoint_every_steps", type=int, default=2000, help="Save checkpoint every N steps")
    parser.add_argument("--generate_every", type=int, default=1000, help="Generate every N steps")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length")
    parser.add_argument("--total_tokens", type=float, default=100e6, help="Total number of tokens to train on")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs (streaming ignored)")
    parser.add_argument("--grad_clip_norm", type=float, default=1.0, help="Global norm for gradient clipping (<=0 to disable)")
    parser.add_argument("--beta1", type=float, default=0.9, help="Beta1 for AdamW")
    parser.add_argument("--beta2", type=float, default=0.95, help="Beta2 for AdamW")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay for AdamW")
    parser.add_argument("--tokenizer_path", type=str, default="meta-llama/Llama-2-7b-hf", help="Tokenizer path")
    # Model parameters
    parser.add_argument("--n_embd", type=int, default=768, help="Embedding dimension size")
    parser.add_argument("--n_layer", type=int, default=18, help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=8, help="Number of attention heads per layer")
    parser.add_argument("--num_key_value_heads", type=int, default=None,
                        help="GQA: number of KV heads. default: n_head (MHA)")
    parser.add_argument("--rope_theta", type=float, default=10000.0, help="RoPE theta (base)")
    parser.add_argument("--rms_norm_eps", type=float, default=1e-6, help="RMSNorm epsilon")
    parser.add_argument("--multiple_of", type=int, default=256, help="intermediate_size rounding multiple")
    parser.add_argument("--intermediate_size", type=int, default=None,
                        help="MLP intermediate size. If None, use LLaMA-style 2/3*4*hidden then round to multiple_of")

    # Dataset  "vesteinn/babylm"
    parser.add_argument("--dataset_path", type=str, default="HuggingFaceFW/fineweb-edu", help="Dataset path for Training")
    parser.add_argument("--val_dataset_path", type=str, default="vesteinn/babylm", help="Dataset path for Validation")
    # Weights & Biases
    parser.add_argument("--wandb_project", type=str, default="NGRC_LanguageModel", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name (default: generated by WandB)")
    parser.add_argument("--api_file", type=str, default="api.txt", help="API file path")

    parser.add_argument("--hf_repo", type=str, default=None, help="Upload destination like 'username/gpt2-scratch-64M'. If None, skip upload")
    parser.add_argument("--hf_private", action="store_true", help="Create HF repo as private")

    return parser.parse_args()

def main():
    args = parse_args()
    WANDB_AVAILABLE = False
    api_keys = load_api_keys(args.api_file)
    if "WANDB_API_KEY" in api_keys:
        os.environ["WANDB_API_KEY"] = api_keys["WANDB_API_KEY"]
    if "HF_READ_TOKEN" in api_keys:
        from huggingface_hub import login
        login(token=api_keys["HF_WRITE_TOKEN"])
    distributed = args.use_deepspeed and _HAS_DS and torch.cuda.is_available()
    if distributed:
        torch.cuda.set_device(args.local_rank)
        distributed = True
    else:
        distributed = False


    world_size = dist.get_world_size() if (distributed and dist.is_initialized()) else 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokenizer (needed for val & generation)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # model
    num_kv = args.num_key_value_heads if args.num_key_value_heads is not None else args.n_head

    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=args.n_embd,
        num_hidden_layers=args.n_layer,
        num_attention_heads=args.n_head,
        num_key_value_heads=num_kv,             # ← GQA
        max_position_embeddings=args.seq_len,
        intermediate_size=args.intermediate_size,
        multiple_of=args.multiple_of,           # ← 256丸め
        rms_norm_eps=args.rms_norm_eps,
        rope_theta=args.rope_theta,
    )
    model = LlamaForCausalLM(config)

    if device.type == "cuda" : 
        model = model.to(device = device, dtype = torch.bfloat16)
    else :
        model = model.to(device = device, dtype = torch.float16)

    param_millions = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"parameter count: {param_millions:.2f}M")

    if distributed:
        # Deepspeedを使うときは，deepspeedがoptimizer/ schedulerを内部に持つ
        model, optimizer, _, _ = deepspeed.initialize(
            args=args,
            model=model,
            model_parameters=model.parameters(),
        )
        scheduler = None  # handled by DeepSpeed
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,

        )
        total_steps = math.ceil(args.total_tokens / (args.local_batch_size * args.use_gpu_amount * args.seq_len))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=math.ceil(0.1 * total_steps),
            num_training_steps=total_steps,
        )
        model.to(device)

    train_loader = get_train_loader(args.tokenizer_path, args.dataset_path, args.seq_len, args.local_batch_size)


    # validation loader unchanged (SlimPajama slice)
    #val_ds = load_dataset("cerebras/SlimPajama-627B", split="train", streaming=True)
    val_ds = load_dataset(args.val_dataset_path, split="test", streaming=True)
    val_blocks = get_validation_blocks(list(islice(val_ds, 100)), tokenizer, args.seq_len, 1)
    val_loader = DataLoader(val_blocks, batch_size=args.local_batch_size, num_workers=0, pin_memory=True)

    max_steps = math.ceil(args.total_tokens / (args.local_batch_size * args.use_gpu_amount * args.seq_len))

    tokens_seen_local = 0
    tokens_seen_global = 0

    start_time = time.time()
    max_mem_mb = 0.0
    model.train()
    if not distributed or args.local_rank == 0:
            wandb_run_name = args.wandb_run_name or f"GPT(RoPE)_{param_millions:.2f}M_batch_size{args.local_batch_size}_seq_len{args.seq_len}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
            wandb.init(project=args.wandb_project, name=wandb_run_name, config=vars(args))
            WANDB_AVAILABLE = True


    for step, batch in enumerate(train_loader, start=1):
        ids = batch.to(device)

        amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        with amp_ctx:
            logits, loss = model(ids, labels=ids)

        if distributed:
            model.backward(loss)
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.module.parameters(), args.grad_clip_norm)
            model.step()
            #current_lr = optimizer.param_groups[0]["lr"]
            current_lr = model.get_lr()[0]
        else:
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            optimizer.zero_grad()

        step_time = time.time() - start_time
        tokens_seen_local += args.local_batch_size * args.seq_len
        #tokens_seen_global += args.local_batch_size * args.seq_len * dist.get_world_size()
        tokens_seen_global += args.local_batch_size * args.seq_len * world_size
        tokens_per_sec_global = tokens_seen_global / step_time if step_time > 0 else 0.0
        tokens_per_sec_local = tokens_seen_local / step_time if step_time > 0 else 0.0
        vram_mb = torch.cuda.max_memory_allocated() / 1e6
        max_mem_mb = max(max_mem_mb, vram_mb)

        if (not distributed) or args.local_rank == 0:
            wandb.log(
                {
                    "train_loss": loss.item(),
                    "train_perplexity": math.exp(loss.item()),
                    "current_lr": current_lr,
                    "seenedtokens": tokens_seen_local,
                    "seenedtokens_global": tokens_seen_global,
                    "tokens_per_sec": tokens_per_sec_local,
                    "tokens_per_sec_global": tokens_per_sec_global,
                    "max_mem_mb": max_mem_mb,
                    "step": step,
                    "max_steps": max_steps,
                },
                step=step,
            )
        if step % args.validate_every_steps == 0:
            model.eval()
            val_loss_list = []
            with torch.no_grad():
                for v_batch in islice(val_loader, args.validate_every_steps):
                    v_ids = v_batch["input_ids"].to(device)
                    v_labels = v_batch["labels"].to(device)
                    v_logits, v_loss = model(v_ids, labels=v_labels)
                    val_loss_list.append(v_loss.item())
            val_loss = sum(val_loss_list) / len(val_loss_list)
            if (not distributed) or args.local_rank == 0:
                wandb.log(
                    {
                        "val_loss": val_loss,
                        "val_perplexity": math.exp(val_loss),
                    }
                )
            model.train()

        if step % args.save_checkpoint_every_steps == 0 and args.local_rank==0:
            ckpt_name = f"checkpoint_step{step}_tokens{tokens_seen_global}.pt"
            save_dir = f"./checkpoint/{args.wandb_project}_{start_time}"
            os.makedirs(save_dir, exist_ok=True)
            if distributed:
                model.save_checkpoint(save_dir=f"./{save_dir}/{ckpt_name}", tag=f"step_{step}")
            else:
                torch.save(model.state_dict(), f"./{save_dir}/{ckpt_name}")

        if step % args.generate_every == 0 and ((not distributed) or args.local_rank == 0):
            model.eval()
            for prompt in ["Hello,", "I'm"]:
                inp_ids = tokenizer.encode(prompt, add_special_tokens=True)
                inp = torch.tensor(inp_ids, dtype=torch.long).to(device)
                generated = sample_sequence(
                    model if not distributed else model.module,
                    inp.unsqueeze(0),
                    max_new_tokens=20,
                    temperature=0.8,
                    top_k=40,
                    top_p=0.9,
                    min_p=0.01,
                )
                output_str = safe_decode(generated[0].tolist(), tokenizer)
                wandb.log({"generated": wandb.Html(f"<b>{prompt}</b>{output_str}")})
            model.train()
        # if step >= max_steps:
        #     break
        if tokens_seen_global >= args.total_tokens:
            break
        
    # ---------- training finished ----------
    total_train_time = time.time() - start_time
    is_master = (not distributed) or args.local_rank == 0
    if is_master:
        



        final_dir = "hf_upload"
        os.makedirs(final_dir, exist_ok=True)
        torch.save(
            model.module.state_dict() if distributed else model.state_dict(),
            os.path.join(final_dir, "pytorch_model.bin"),
        )
        with open(os.path.join(final_dir, "config.json"), "w") as f:
            json.dump(config.__dict__, f, indent=2)

        # upload to HF
        if "HF_WRITE_TOKEN" in api_keys:
            repo_id, api = prepare_hf_repo_id(api_keys["HF_WRITE_TOKEN"], args.hf_repo)
            api.create_repo(repo_id=repo_id, exist_ok=True, private=args.hf_private)
            upload_folder(repo_id=repo_id, folder_path=final_dir, path_in_repo=".", token=api_keys["HF_WRITE_TOKEN"], ignore_patterns=["*.pt"])
            print(f"✅ Model pushed to https://huggingface.co/{repo_id}")
        else:
            print("HF upload skipped (token absent or repo not specified).")

        # stats & report
        param_count = sum(p.numel() for p in model.parameters())
        report = {
            "run_name": wandb.run.name if WANDB_AVAILABLE else "offline_run",
            "hyperparameters": vars(args),
            "parameter_count": param_count,
            "max_gpu_memory_MB": max_mem_mb,
            "training_time_sec": total_train_time,
            "final_train_loss": loss.item(),
            "final_train_perplexity": math.exp(loss.item()),
            "final_val_loss": val_loss if 'val_loss' in locals() else None,
            "final_val_perplexity": math.exp(val_loss) if 'val_loss' in locals() else None,
        }

        # inference memory & speed test
        test_input = torch.randint(0, tokenizer.vocab_size, (1, args.seq_len), device=device)
        torch.cuda.reset_peak_memory_stats()
        t_inf_start = time.time()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _ = model(test_input)
        infer_time = time.time() - t_inf_start
        infer_tok_per_sec = 1024 / infer_time if infer_time > 0 else 0.0
        infer_mem_mb = torch.cuda.max_memory_allocated() / 1e6
        report.update({"inference_tok_per_sec": infer_tok_per_sec, "inference_mem_MB": infer_mem_mb})
        seq_len_test = args.seq_len * 8
        print(f"▶️ babylm テストセットで mean-so-far-PPL を最大 {seq_len_test} トークンまで計測します…")
        test_ds = load_dataset(args.val_dataset_path, split="test", streaming=False)
        buffer = []
        test_blocks = []
        for ex in test_ds:
            toks = tokenizer(ex["text"], return_attention_mask=False)["input_ids"]
            buffer.extend(toks)
            while len(buffer) >= seq_len_test:
                blk = buffer[:seq_len_test]
                test_blocks.append({
                    "input_ids": torch.tensor(blk, dtype=torch.long),
                    "labels":   torch.tensor(blk, dtype=torch.long),
                })
                buffer = buffer[seq_len_test:]
        print(f"→ {len(test_blocks)} ブロックを作成")
        test_blocks = test_blocks[:100]
        ks = list(range(1, seq_len_test + 1))

        torch.cuda.reset_peak_memory_stats()
        t_inf_start = time.time()
        mean_so_far = compute_mean_so_far_ppl(
            model if not distributed else model.module,
            test_blocks,
            device,
            ks
        )
        t_inf_end = time.time()
        inf_time = t_inf_end - t_inf_start
        total_inf_tokens = len(test_blocks) * seq_len_test
        infer_tok_per_sec = total_inf_tokens / inf_time if inf_time > 0 else 0.0
        infer_mem_mb = torch.cuda.max_memory_allocated() / 1e6
        report.update({
            "inference_tok_per_sec": infer_tok_per_sec,
            "inference_mem_MB": infer_mem_mb
        })
        print(f"Inference time: {inf_time:.2f}s, Tokens/sec: {infer_tok_per_sec:.2f}, Memory: {infer_mem_mb:.2f}MB")
        if WANDB_AVAILABLE:
            wandb.log({"inference_time": inf_time, "inference_tok_per_sec": infer_tok_per_sec, "inference_mem_MB": infer_mem_mb})

        report_ks = []
        k = 1
        while k <= seq_len_test:
            report_ks.append(k)
            k *=2
        report_ks = [k for k in report_ks if k <= seq_len_test]

        if WANDB_AVAILABLE:
            table = wandb.Table(columns = ["token_length", "perplexity"])
            for k in report_ks:
                table.add_data(k,mean_so_far[k]["mean_ppl"])
            chart = wandb.plot.line(
                table,
                x ="token_length",
                y ="perplexity",
                title="Mean-so-far PPL vs Token Length"
            )
            wandb.log({"mean_so_far_ppl_curve": chart})
        # for k in report_ks:
        #     v = mean_so_far[k]
        #     print(f" mean-so-far@{k:4d} →  NLL={v['mean_nll']:.4f},  PPL={v['mean_ppl']:.2f}")
        #     if WANDB_AVAILABLE:
        #         wandb.log({f"test_mean_so_far_ppl@{k}": v["mean_ppl"]})

        # 5) JSON レポートにも一括格納（必要なら全 ks を文字列化して保存）
        report["test_mean_so_far_ppl_curve"] = {
            k: mean_so_far[k]["mean_ppl"] for k in ks
        }

        report_path = f"./{report['run_name']}_report.txt"
        with open(report_path, "w") as f:
            f.write(json.dumps(report, indent=2))
        print(f"📄 Training report written to {report_path}")

    if WANDB_AVAILABLE:
        wandb.finish()

if __name__ == "__main__":
    main()


