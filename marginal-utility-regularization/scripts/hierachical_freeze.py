import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train.train_clm import train
from utils.config import apply_overrides, load_config


def _ensure_hierarchical_defaults(config):
    hfreeze_cfg = config.setdefault("hierarchical_freeze", {})
    hfreeze_cfg.setdefault("enabled", True)
    hfreeze_cfg.setdefault("mode", "input")
    if "unfreeze_tokens" not in hfreeze_cfg:
        max_train_tokens = config.get("max_train_tokens")
        if max_train_tokens is not None:
            hfreeze_cfg["unfreeze_tokens"] = int(max_train_tokens)
        else:
            steps = int(config.get("num_train_steps", 0))
            batch_size = int(config.get("batch_size", 1))
            seq_len = int(config.get("seq_len", 1))
            hfreeze_cfg["unfreeze_tokens"] = steps * batch_size * seq_len
    hfreeze_cfg.setdefault("seed", config.get("seed", 42))

    freeze_cfg = config.setdefault("freeze", {})
    freeze_cfg["enabled"] = False
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a causal LM with hierarchical unfreezing"
    )
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument("--override", action="append", default=[], help="Override config: key=value")
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_overrides(config, args.override)
    config = _ensure_hierarchical_defaults(config)

    train(config)


if __name__ == "__main__":
    main()
