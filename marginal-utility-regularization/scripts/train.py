import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from train.train_clm import train
from utils.config import apply_overrides, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a causal LM with MUR regularization")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument("--override", action="append", default=[], help="Override config: key=value")
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_overrides(config, args.override)

    train(config)


if __name__ == "__main__":
    main()
