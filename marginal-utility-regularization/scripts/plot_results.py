import argparse
import json
import math
import os
import sys
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import matplotlib.pyplot as plt

from models.hidden_state_utils import resolve_layer_indices
from visualization.plot_metrics import load_json, load_jsonl, plot_layer_curve, plot_metric_curve


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


def _resolve_run_dir(path: str) -> Optional[str]:
    if not path or not os.path.isdir(path):
        return None
    if os.path.basename(path) == "metrics":
        return os.path.dirname(path)
    if os.path.exists(os.path.join(path, "config.json")):
        return path
    candidate = os.path.join(path, "metrics")
    if os.path.isdir(candidate) and os.path.exists(os.path.join(path, "config.json")):
        return path
    return None


def _load_ppl(metrics_dir: str) -> Optional[float]:
    ppl_path = os.path.join(metrics_dir, "ppl.json")
    if os.path.exists(ppl_path):
        return float(load_json(ppl_path).get("ppl", float("nan")))
    eval_path = os.path.join(metrics_dir, "eval.jsonl")
    if os.path.exists(eval_path):
        records = load_jsonl(eval_path)
        if records:
            return float(records[-1].get("eval_ppl", float("nan")))
    return None


def _plot_ppl_bar(baseline: float, mur: float, out_path: str, title: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.bar(["baseline", "mur"], [baseline, mur], color=["#4c72b0", "#dd8452"])
    plt.ylabel("Perplexity")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def _plot_pair(baseline_dir: str, mur_dir: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    base_metrics = _resolve_metrics_dir(baseline_dir)
    mur_metrics = _resolve_metrics_dir(mur_dir)
    if base_metrics is None or mur_metrics is None:
        raise SystemExit("Could not resolve metrics directories for baseline or MUR")

    mur_run_dir = _resolve_run_dir(mur_dir)
    mur_config = None
    if mur_run_dir:
        config_path = os.path.join(mur_run_dir, "config.json")
        if os.path.exists(config_path):
            mur_config = load_json(config_path)

    base_mur_path = os.path.join(base_metrics, "mur_stats.json")
    mur_mur_path = os.path.join(mur_metrics, "mur_stats.json")
    if os.path.exists(base_mur_path) and os.path.exists(mur_mur_path):
        base_mur = load_json(base_mur_path)
        mur_mur = load_json(mur_mur_path)
        plot_layer_curve(base_mur, mur_mur, "mean", os.path.join(out_dir, "mur_mean.png"), "MUR mean")
        plot_layer_curve(
            base_mur,
            mur_mur,
            "frac_below_tau",
            os.path.join(out_dir, "mur_frac_below_tau.png"),
            "MUR frac below tau",
        )
        plot_layer_curve(
            base_mur,
            mur_mur,
            "mean_delta_norm",
            os.path.join(out_dir, "mur_delta_norm.png"),
            "Delta norm",
        )
        plot_layer_curve(
            base_mur,
            mur_mur,
            "mean_grad_norm",
            os.path.join(out_dir, "mur_grad_norm.png"),
            "Grad norm",
        )

    base_bi_path = os.path.join(base_metrics, "bi_metric.json")
    mur_bi_path = os.path.join(mur_metrics, "bi_metric.json")
    if os.path.exists(base_bi_path) and os.path.exists(mur_bi_path):
        base_bi = load_json(base_bi_path)
        mur_bi = load_json(mur_bi_path)
        plot_layer_curve(base_bi, mur_bi, "bi", os.path.join(out_dir, "bi_curve.png"), "Block Influence")

    base_drop_path = os.path.join(base_metrics, "layer_drop.json")
    mur_drop_path = os.path.join(mur_metrics, "layer_drop.json")
    if os.path.exists(base_drop_path) and os.path.exists(mur_drop_path):
        base_drop = load_json(base_drop_path)
        mur_drop = load_json(mur_drop_path)
        plot_layer_curve(
            base_drop,
            mur_drop,
            "delta_ppl",
            os.path.join(out_dir, "layer_drop_delta_ppl.png"),
            "Layer drop delta PPL",
        )
        if mur_config:
            mur_cfg = mur_config.get("mur", {})
            mid_start = mur_cfg.get("mid_start", 0.33)
            mid_end = mur_cfg.get("mid_end", 0.67)
            num_layers = len(mur_drop.get("delta_ppl", []))
            mid_indices = resolve_layer_indices(num_layers, mid_start, mid_end)
            if mid_indices:
                base_mid = {
                    "layers": mid_indices,
                    "delta_ppl": [base_drop["delta_ppl"][i] for i in mid_indices],
                }
                mur_mid = {
                    "layers": mid_indices,
                    "delta_ppl": [mur_drop["delta_ppl"][i] for i in mid_indices],
                }
                plot_layer_curve(
                    base_mid,
                    mur_mid,
                    "delta_ppl",
                    os.path.join(out_dir, "mid_layer_drop_delta_ppl.png"),
                    "Mid-layer drop delta PPL",
                )

    baseline_ppl = _load_ppl(base_metrics)
    mur_ppl = _load_ppl(mur_metrics)
    if baseline_ppl is not None and mur_ppl is not None:
        _plot_ppl_bar(baseline_ppl, mur_ppl, os.path.join(out_dir, "ppl_comparison.png"), "PPL comparison")

    base_eval_path = os.path.join(base_metrics, "eval.jsonl")
    mur_eval_path = os.path.join(mur_metrics, "eval.jsonl")
    if os.path.exists(base_eval_path):
        base_eval = load_jsonl(base_eval_path)
        plot_metric_curve(base_eval, "eval_ppl", os.path.join(out_dir, "baseline_eval_ppl.png"), "Baseline eval PPL")
    if os.path.exists(mur_eval_path):
        mur_eval = load_jsonl(mur_eval_path)
        plot_metric_curve(mur_eval, "eval_ppl", os.path.join(out_dir, "mur_eval_ppl.png"), "MUR eval PPL")


def _parse_run_name(run_name: str) -> Optional[Tuple[str, str, str]]:
    for arch in ("transformer_pp", "gpt2", "llama"):
        prefix = f"{arch}_"
        if run_name.startswith(prefix):
            rest = run_name[len(prefix) :]
            parts = rest.split("_")
            if len(parts) >= 2:
                return arch, parts[0], parts[1]
    return None


def _collect_pairs(runs_dir: str) -> Dict[Tuple[str, str], Dict[str, str]]:
    pairs: Dict[Tuple[str, str], Dict[str, str]] = {}
    for name in os.listdir(runs_dir):
        run_path = os.path.join(runs_dir, name)
        if not os.path.isdir(run_path):
            continue
        parsed = _parse_run_name(name)
        if parsed is None:
            continue
        arch, size_tag, variant = parsed
        key = (arch, size_tag)
        if key not in pairs:
            pairs[key] = {}
        pairs[key][variant] = run_path
    return pairs


def _collect_runs_for_scaling(root: str):
    runs = []
    for dirpath, _, filenames in os.walk(root):
        if "config.json" not in filenames:
            continue
        metrics_dir = os.path.join(dirpath, "metrics")
        ppl_path = os.path.join(metrics_dir, "ppl.json")
        if not os.path.exists(ppl_path) and not os.path.exists(os.path.join(metrics_dir, "eval.jsonl")):
            continue

        config = load_json(os.path.join(dirpath, "config.json"))
        model_cfg = config.get("model", {})
        arch = model_cfg.get("arch", "gpt2")
        mur_enabled = bool(config.get("mur", {}).get("enabled", False))
        variant = "mur" if mur_enabled else "baseline"

        params_path = os.path.join(dirpath, "model_params.json")
        if not os.path.exists(params_path):
            continue
        params = load_json(params_path)
        params_m = float(params.get("param_count_m", 0.0))
        if params_m <= 0:
            continue

        ppl_value = _load_ppl(metrics_dir)
        if ppl_value is None:
            continue

        runs.append(
            {
                "run_dir": dirpath,
                "arch": arch,
                "variant": variant,
                "params_m": params_m,
                "ppl": ppl_value,
            }
        )
    return runs


def _plot_scaling(runs, out_dir: str) -> None:
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


def _write_scaling_table(runs, out_dir: str) -> None:
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
    parser = argparse.ArgumentParser(description="Plot MUR experiment results")
    parser.add_argument("--baseline_dir", default=None, help="Baseline run directory")
    parser.add_argument("--mur_dir", default=None, help="MUR run directory")
    parser.add_argument("--runs_dir", default=None, help="Root runs directory for auto pairing or scaling plots")
    parser.add_argument("--out_dir", required=True, help="Output directory for plots")
    parser.add_argument("--scaling_plot", action="store_true", default=False, help="Plot scaling curves instead of pairwise comparison")
    args = parser.parse_args()

    if args.scaling_plot:
        if not args.runs_dir:
            raise SystemExit("--runs_dir is required when --scaling_plot is set")
        runs = _collect_runs_for_scaling(args.runs_dir)
        if not runs:
            raise SystemExit("No runs found with config.json and metrics")
        _plot_scaling(runs, args.out_dir)
        _write_scaling_table(runs, args.out_dir)
        print("[info] Scaling plots and table saved to", args.out_dir)
        return

    if args.baseline_dir and args.mur_dir:
        _plot_pair(args.baseline_dir, args.mur_dir, args.out_dir)
        print("[info] Plots saved to", args.out_dir)
        return

    if args.runs_dir:
        pairs = _collect_pairs(args.runs_dir)
        if not pairs:
            raise SystemExit("No baseline/mur pairs found under runs_dir")
        for (arch, size_tag), variants in sorted(pairs.items()):
            if "baseline" not in variants or "mur" not in variants:
                continue
            pair_out = os.path.join(args.out_dir, f"{arch}_{size_tag}")
            _plot_pair(variants["baseline"], variants["mur"], pair_out)
            print(f"[info] Plots for {arch} {size_tag} saved to", pair_out)
        return

    raise SystemExit("Provide --baseline_dir/--mur_dir or --runs_dir")


if __name__ == "__main__":
    main()
