import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train.train_clm import train
from utils.config import apply_overrides, load_config


def _ensure_freeze_defaults(config):
    freeze_cfg = config.setdefault("freeze", {})
    freeze_cfg.setdefault("enabled", True)
    freeze_cfg.setdefault("mode", "random_near_input")
    freeze_cfg.setdefault("near_input_ratio", 0.25)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a causal LM with a randomly frozen input-near layer"
    )
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument("--override", action="append", default=[], help="Override config: key=value")
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_overrides(config, args.override)
    config = _ensure_freeze_defaults(config)

    train(config)


if __name__ == "__main__":
    main()
