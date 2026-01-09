import torch
from typing import Any, Dict

from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

from models.transformer_pp import TransformerPPConfig, TransformerPPForCausalLM


def _infer_intermediate_size(hidden_size: int, multiplier: float) -> int:
    return int(hidden_size * multiplier)


def build_tokenizer(config: Dict[str, Any]):
    model_cfg = config.get("model", {})
    arch = model_cfg.get("arch")
    if arch == "transformer++":
        arch = "transformer_pp"
    name = config.get("tokenizer_name") or "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_model(config: Dict[str, Any]) -> torch.nn.Module:
    model_cfg = config.get("model")
    if model_cfg is None:
        model_name = config.get("model_name")
        if model_name is None:
            raise ValueError("Either model or model_name must be specified")
        return GPT2LMHeadModel.from_pretrained(model_name)

    arch = model_cfg.get("arch", "gpt2")
    if arch == "transformer++":
        arch = "transformer_pp"
    vocab_size = model_cfg["vocab_size"]
    hidden_size = model_cfg["hidden_size"]
    num_layers = model_cfg["num_layers"]
    num_heads = model_cfg["num_heads"]
    max_positions = model_cfg.get("max_position_embeddings", 1024)

    if arch == "gpt2":
        intermediate = model_cfg.get("intermediate_size", _infer_intermediate_size(hidden_size, 4.0))
        gpt2_cfg = GPT2Config(
            vocab_size=vocab_size,
            n_positions=max_positions,
            n_ctx=max_positions,
            n_embd=hidden_size,
            n_layer=num_layers,
            n_head=num_heads,
            n_inner=intermediate,
            resid_pdrop=model_cfg.get("resid_dropout", 0.0),
            embd_pdrop=model_cfg.get("embd_dropout", 0.0),
            attn_pdrop=model_cfg.get("attn_dropout", 0.0),
            bos_token_id=model_cfg.get("bos_token_id", 50256),
            eos_token_id=model_cfg.get("eos_token_id", 50256),
            tie_word_embeddings=model_cfg.get("tie_word_embeddings", True),
        )
        return GPT2LMHeadModel(gpt2_cfg)

    if arch == "llama":
        intermediate = model_cfg.get("intermediate_size")
        if intermediate is None:
            intermediate = _infer_intermediate_size(hidden_size, 8.0 / 3.0)
        llama_cfg = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=model_cfg.get("num_key_value_heads", num_heads),
            intermediate_size=intermediate,
            max_position_embeddings=max_positions,
            rms_norm_eps=model_cfg.get("rms_norm_eps", 1e-5),
            rope_theta=model_cfg.get("rope_theta", 10000.0),
            attention_dropout=model_cfg.get("attn_dropout", 0.0),
            hidden_dropout=model_cfg.get("resid_dropout", 0.0),
            tie_word_embeddings=model_cfg.get("tie_word_embeddings", False),
            bos_token_id=model_cfg.get("bos_token_id", 1),
            eos_token_id=model_cfg.get("eos_token_id", 2),
            attention_bias=model_cfg.get("attention_bias", False),
        )
        return LlamaForCausalLM(llama_cfg)

    if arch == "transformer_pp":
        intermediate = model_cfg.get("intermediate_size", _infer_intermediate_size(hidden_size, 4.0))
        cfg = TransformerPPConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_size=intermediate,
            max_position_embeddings=max_positions,
            attn_dropout=model_cfg.get("attn_dropout", 0.0),
            resid_dropout=model_cfg.get("resid_dropout", 0.0),
            embd_dropout=model_cfg.get("embd_dropout", 0.0),
            rms_norm_eps=model_cfg.get("rms_norm_eps", 1e-5),
            use_rope=model_cfg.get("use_rope", False),
            rope_theta=model_cfg.get("rope_theta", 10000.0),
            tie_word_embeddings=model_cfg.get("tie_word_embeddings", True),
            use_bias=model_cfg.get("use_bias", True),
        )
        return TransformerPPForCausalLM(cfg)

    raise ValueError(f"Unknown model arch: {arch}")
