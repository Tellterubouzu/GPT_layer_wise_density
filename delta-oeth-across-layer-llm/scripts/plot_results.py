import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from visualization.plot_metrics import load_json, load_jsonl, plot_layer_curve, plot_metric_curve


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot delta-orth experiment results")
    parser.add_argument("--baseline_dir", required=True, help="Baseline run directory")
    parser.add_argument("--delta_dir", required=True, help="Delta-orth run directory")
    parser.add_argument("--out_dir", required=True, help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    base_metrics = os.path.join(args.baseline_dir, "metrics")
    delta_metrics = os.path.join(args.delta_dir, "metrics")

    base_delta_path = os.path.join(base_metrics, "delta_stats.json")
    delta_delta_path = os.path.join(delta_metrics, "delta_stats.json")
    if os.path.exists(base_delta_path) and os.path.exists(delta_delta_path):
        base_delta = load_json(base_delta_path)
        delta_delta = load_json(delta_delta_path)
        plot_layer_curve(base_delta, delta_delta, "mean_cos", os.path.join(args.out_dir, "delta_cos_mean.png"), "Delta cosine mean")
        plot_layer_curve(base_delta, delta_delta, "p90_cos", os.path.join(args.out_dir, "delta_cos_p90.png"), "Delta cosine p90")

    base_bi_path = os.path.join(base_metrics, "bi_metric.json")
    delta_bi_path = os.path.join(delta_metrics, "bi_metric.json")
    if os.path.exists(base_bi_path) and os.path.exists(delta_bi_path):
        base_bi = load_json(base_bi_path)
        delta_bi = load_json(delta_bi_path)
        plot_layer_curve(base_bi, delta_bi, "bi", os.path.join(args.out_dir, "bi_curve.png"), "Block Influence")

    base_drop_path = os.path.join(base_metrics, "layer_drop.json")
    delta_drop_path = os.path.join(delta_metrics, "layer_drop.json")
    if os.path.exists(base_drop_path) and os.path.exists(delta_drop_path):
        base_drop = load_json(base_drop_path)
        delta_drop = load_json(delta_drop_path)
        plot_layer_curve(base_drop, delta_drop, "delta_ppl", os.path.join(args.out_dir, "layer_drop_delta_ppl.png"), "Layer drop delta PPL")

    base_eval_path = os.path.join(base_metrics, "eval.jsonl")
    delta_eval_path = os.path.join(delta_metrics, "eval.jsonl")
    if os.path.exists(base_eval_path):
        base_eval = load_jsonl(base_eval_path)
        plot_metric_curve(base_eval, "eval_ppl", os.path.join(args.out_dir, "baseline_eval_ppl.png"), "Baseline eval PPL", label="baseline")
    if os.path.exists(delta_eval_path):
        delta_eval = load_jsonl(delta_eval_path)
        plot_metric_curve(delta_eval, "eval_ppl", os.path.join(args.out_dir, "delta_eval_ppl.png"), "Delta-orth eval PPL", label="delta-orth")


if __name__ == "__main__":
    main()
