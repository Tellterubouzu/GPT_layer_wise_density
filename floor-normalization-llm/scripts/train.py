import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from train.train_clm import train
from utils.config import apply_overrides, load_config, save_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a causal LM with BI-Floor regularization")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument("--override", action="append", default=[], help="Override config: key=value")
    args = parser.parse_args()

    config = load_config(args.config)
    config = apply_overrides(config, args.override)

    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    save_config(config, os.path.join(output_dir, "config.json"))

    train(config)


if __name__ == "__main__":
    main()
