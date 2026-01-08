import json
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def load_jsonl(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_layer_curve(
    baseline: Dict[str, List[float]],
    delta: Dict[str, List[float]],
    key: str,
    out_path: str,
    title: str,
    baseline_label: str = "baseline",
    delta_label: str = "delta-orth",
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(baseline[key], label=baseline_label)
    plt.plot(delta[key], label=delta_label)
    plt.xlabel("Layer")
    plt.ylabel(key)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_metric_curve(
    records: List[Dict], key: str, out_path: str, title: str, label: Optional[str] = None
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    steps = [r["step"] for r in records]
    values = [r[key] for r in records]
    plt.figure(figsize=(8, 4))
    plt.plot(steps, values, label=label or key)
    plt.xlabel("Step")
    plt.ylabel(key)
    plt.title(title)
    if label:
        plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
