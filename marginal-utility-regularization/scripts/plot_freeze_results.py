import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib.pyplot as plt

from visualization.plot_metrics import load_json, load_jsonl


def _resolve_metrics_dir(path: str) -> Optional[str]:
    if not path:
        return None
    if os.path.isdir(path):
        if os.path.basename(path) == "metrics":
            return path
        candidate = os.path.join(path, "metrics")
        if os.path.isdir(candidate):
            return candidate
        return path
    return None


def _load_train_records(path: str) -> List[Dict]:
    metrics_dir = _resolve_metrics_dir(path)
    if metrics_dir is None:
        return []
    train_path = os.path.join(metrics_dir, "train.jsonl")
    if not os.path.exists(train_path):
        return []
    return load_jsonl(train_path)

def _load_metric_json(path: str, filename: str) -> Optional[Dict]:
    metrics_dir = _resolve_metrics_dir(path)
    if metrics_dir is None:
        return None
    metric_path = os.path.join(metrics_dir, filename)
    if not os.path.exists(metric_path):
        return None
    return load_json(metric_path)


def _extract_series(records: List[Dict], key: str) -> Tuple[List[int], List[float]]:
    steps: List[int] = []
    values: List[float] = []
    for record in records:
        if key in record:
            steps.append(int(record["step"]))
            values.append(float(record[key]))
    return steps, values


def _plot_metric(
    baseline: Optional[List[Dict]],
    freeze: Optional[List[Dict]],
    key: str,
    out_path: str,
    title: str,
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 4))
    if baseline:
        steps, values = _extract_series(baseline, key)
        if steps:
            plt.plot(steps, values, label="baseline", color="#4c72b0")
    if freeze:
        steps, values = _extract_series(freeze, key)
        if steps:
            plt.plot(steps, values, label="freeze", color="#dd8452")
    plt.xlabel("Step")
    plt.ylabel(key)
    plt.title(title)
    if baseline and freeze:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_layer_metric(
    baseline: Optional[Dict],
    freeze: Optional[Dict],
    key: str,
    out_path: str,
    title: str,
) -> None:
    if not freeze or key not in freeze:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    base_layers = baseline.get("layers") if baseline else None
    freeze_layers = freeze.get("layers")
    if freeze_layers is None:
        freeze_layers = list(range(len(freeze[key])))
    plt.figure(figsize=(8, 4))
    if baseline and key in baseline:
        if base_layers is None:
            base_layers = list(range(len(baseline[key])))
        plt.plot(base_layers, baseline[key], label="baseline", color="#4c72b0")
    plt.plot(freeze_layers, freeze[key], label="freeze", color="#dd8452")
    plt.xlabel("Layer")
    plt.ylabel(key)
    plt.title(title)
    if baseline and key in baseline:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves for freeze experiments")
    parser.add_argument("--freeze_dir", required=True, help="Run dir or metrics dir for freeze run")
    parser.add_argument("--baseline_dir", default=None, help="Optional baseline run dir or metrics dir")
    parser.add_argument("--out_dir", required=True, help="Output directory for plots")
    args = parser.parse_args()

    freeze_records = _load_train_records(args.freeze_dir)
    if not freeze_records:
        raise SystemExit("Could not find train.jsonl for freeze_dir")

    baseline_records = _load_train_records(args.baseline_dir) if args.baseline_dir else None

    _plot_metric(baseline_records, freeze_records, "train_loss", os.path.join(args.out_dir, "train_loss.png"), "Train loss")
    _plot_metric(
        baseline_records,
        freeze_records,
        "train_perplexity",
        os.path.join(args.out_dir, "train_perplexity.png"),
        "Train perplexity",
    )

    baseline_bi = _load_metric_json(args.baseline_dir, "bi_metric.json") if args.baseline_dir else None
    freeze_bi = _load_metric_json(args.freeze_dir, "bi_metric.json")
    _plot_layer_metric(
        baseline_bi,
        freeze_bi,
        "bi",
        os.path.join(args.out_dir, "bi_curve.png"),
        "Block influence",
    )

    baseline_drop = _load_metric_json(args.baseline_dir, "layer_drop.json") if args.baseline_dir else None
    freeze_drop = _load_metric_json(args.freeze_dir, "layer_drop.json")
    _plot_layer_metric(
        baseline_drop,
        freeze_drop,
        "delta_ppl",
        os.path.join(args.out_dir, "layer_drop_delta_ppl.png"),
        "Layer drop delta PPL",
    )


if __name__ == "__main__":
    main()
