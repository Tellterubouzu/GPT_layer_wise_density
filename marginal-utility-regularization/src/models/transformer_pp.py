from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from models.modeling_outputs import CausalLMOutput


@dataclass
class TransformerPPConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int
    max_position_embeddings: int
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    embd_dropout: float = 0.0
    rms_norm_eps: float = 1e-5
    use_rope: bool = False
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    use_bias: bool = True


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10000.0) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]

    def _rotate(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        rotated = torch.stack((-x2, x1), dim=-1)
        return rotated.flatten(-2)

    q_rot = (q * cos) + (_rotate(q) * sin)
    k_rot = (k * cos) + (_rotate(k) * sin)
    return q_rot, k_rot


class TransformerPPAttention(nn.Module):
    def __init__(self, config: TransformerPPConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=config.use_bias)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=config.use_bias)
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.resid_dropout)
        self.use_rope = config.use_rope
        if config.use_rope and self.head_dim % 2 != 0:
            raise ValueError("head_dim must be even when using RoPE")
        self.rotary = RotaryEmbedding(self.head_dim, base=config.rope_theta) if config.use_rope else None

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_rope and self.rotary is not None:
            cos, sin = self.rotary(seq_len, x.device)
            q, k = apply_rotary(q, k, cos, sin)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        attn_scores = attn_scores.masked_fill(causal_mask, float("-inf"))

        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :] == 0
            attn_scores = attn_scores.masked_fill(key_mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        attn_output = self.resid_dropout(self.out_proj(attn_output))
        return attn_output


class TransformerPPMLP(nn.Module):
    def __init__(self, config: TransformerPPConfig) -> None:
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.use_bias)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.use_bias)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.use_bias)
        self.dropout = nn.Dropout(config.resid_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.gate(x)) * self.up(x)
        return self.dropout(self.down(gated))


class TransformerPPBlock(nn.Module):
    def __init__(self, config: TransformerPPConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = TransformerPPAttention(config)
        self.norm2 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = TransformerPPMLP(config)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerPPForCausalLM(nn.Module):
    def __init__(self, config: TransformerPPConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_positions = None
        if not config.use_rope:
            self.embed_positions = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.embd_dropout)
        self.layers = nn.ModuleList([TransformerPPBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ):
        bsz, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device)
        hidden_states = self.embed_tokens(input_ids)
        if self.embed_positions is not None:
            hidden_states = hidden_states + self.embed_positions(positions)
        hidden_states = self.dropout(hidden_states)

        all_hidden_states = [hidden_states] if output_hidden_states else None

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if attention_mask is not None:
                labels = labels.masked_fill(attention_mask == 0, -100)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1), ignore_index=-100)

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
        )
