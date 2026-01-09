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


def _resolve_layers(data: Dict, key: str) -> List[int]:
    if "layers" in data:
        return data["layers"]
    return list(range(len(data.get(key, []))))


def plot_layer_curve(
    baseline: Dict[str, List[float]],
    mur: Dict[str, List[float]],
    key: str,
    out_path: str,
    title: str,
    baseline_label: str = "baseline",
    mur_label: str = "mur",
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    base_layers = _resolve_layers(baseline, key)
    mur_layers = _resolve_layers(mur, key)
    plt.figure(figsize=(8, 4))
    plt.plot(base_layers, baseline[key], label=baseline_label)
    plt.plot(mur_layers, mur[key], label=mur_label)
    plt.xlabel("Layer")
    plt.ylabel(key)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_metric_curve(
    records: List[Dict],
    key: str,
    out_path: str,
    title: str,
    label: Optional[str] = None,
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
