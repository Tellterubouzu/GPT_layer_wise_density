import argparse
import json
import math
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_last_eval_ppl(path: str) -> float:
    if not os.path.exists(path):
        return float("nan")
    last = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            last = json.loads(line)
    if last and "eval_ppl" in last:
        return float(last["eval_ppl"])
    return float("nan")


def _collect_runs(root: str) -> List[Dict[str, Any]]:
    runs = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "config.json" not in filenames:
            continue
        metrics_dir = os.path.join(dirpath, "metrics")
        ppl_path = os.path.join(metrics_dir, "ppl.json")
        if not os.path.exists(ppl_path) and not os.path.exists(os.path.join(metrics_dir, "eval.jsonl")):
            continue

        config = _load_json(os.path.join(dirpath, "config.json"))
        model_cfg = config.get("model", {})
        arch = model_cfg.get("arch", "gpt2")
        mur_enabled = bool(config.get("mur", {}).get("enabled", False))
        variant = "mur" if mur_enabled else "baseline"

        params_path = os.path.join(dirpath, "model_params.json")
        if not os.path.exists(params_path):
            continue
        params = _load_json(params_path)
        params_m = float(params.get("param_count_m", 0.0))
        if params_m <= 0:
            continue

        if os.path.exists(ppl_path):
            ppl = float(_load_json(ppl_path).get("ppl", float("nan")))
        else:
            ppl = _load_last_eval_ppl(os.path.join(metrics_dir, "eval.jsonl"))

        runs.append(
            {
                "run_dir": dirpath,
                "arch": arch,
                "variant": variant,
                "params_m": params_m,
                "ppl": ppl,
            }
        )
    return runs


def _plot_scaling(runs: List[Dict[str, Any]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    arches = sorted({run["arch"] for run in runs})
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#66a61e"]
    color_map = {arch: colors[idx % len(colors)] for idx, arch in enumerate(arches)}
    label_map = {"gpt2": "GPT-2", "transformer_pp": "Transformer++", "llama": "Llama"}

    plt.figure(figsize=(8, 5))
    for arch in arches:
        for variant, style in [("baseline", "-"), ("mur", "--")]:
            subset = [run for run in runs if run["arch"] == arch and run["variant"] == variant]
            if not subset:
                continue
            subset = sorted(subset, key=lambda r: r["params_m"])
            xs = [r["params_m"] for r in subset]
            ys = [r["ppl"] for r in subset]
            filtered = [(x, y) for x, y in zip(xs, ys) if not math.isnan(y)]
            if not filtered:
                continue
            xs, ys = zip(*filtered)
            label_arch = label_map.get(arch, arch)
            label = f"{label_arch}-{variant}"
            plt.plot(xs, ys, linestyle=style, marker="o", color=color_map[arch], label=label)

    plt.xlabel("Model size (M params)")
    plt.ylabel("Perplexity")
    plt.title("Scaling: Model Size vs Perplexity")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scaling_plot.png"))
    plt.close()


def _write_table(runs: List[Dict[str, Any]], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    rows = [r for r in runs if not math.isnan(r["ppl"])]
    rows = sorted(rows, key=lambda r: (r["arch"], r["variant"], r["params_m"]))
    csv_path = os.path.join(out_dir, "scaling_table.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("arch,variant,params_m,ppl,run_dir\n")
        for row in rows:
            f.write(f"{row['arch']},{row['variant']},{row['params_m']:.2f},{row['ppl']:.4f},{row['run_dir']}\n")

    table_data = [[r["arch"], r["variant"], f"{r['params_m']:.2f}", f"{r['ppl']:.4f}"] for r in rows]
    plt.figure(figsize=(8, 0.4 * (len(rows) + 2)))
    plt.axis("off")
    table = plt.table(
        cellText=table_data,
        colLabels=["arch", "variant", "params_m", "ppl"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scaling_table.png"))
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot scaling curves and tables from run directories")
    parser.add_argument("--runs_dir", required=True, help="Root directory containing run subfolders")
    parser.add_argument("--out_dir", required=True, help="Output directory for plots")
    args = parser.parse_args()

    runs = _collect_runs(args.runs_dir)
    if not runs:
        raise SystemExit("No runs found with config.json and metrics")

    _plot_scaling(runs, args.out_dir)
    _write_table(runs, args.out_dir)


if __name__ == "__main__":
    main()
