import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from visualization.plot_metrics import load_json, load_jsonl, plot_layer_curve, plot_metric_curve


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MUR experiment results")
    parser.add_argument("--baseline_dir", required=True, help="Baseline run directory")
    parser.add_argument("--mur_dir", required=True, help="MUR run directory")
    parser.add_argument("--out_dir", required=True, help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    base_metrics = os.path.join(args.baseline_dir, "metrics")
    mur_metrics = os.path.join(args.mur_dir, "metrics")

    base_mur_path = os.path.join(base_metrics, "mur_stats.json")
    mur_mur_path = os.path.join(mur_metrics, "mur_stats.json")
    if os.path.exists(base_mur_path) and os.path.exists(mur_mur_path):
        base_mur = load_json(base_mur_path)
        mur_mur = load_json(mur_mur_path)
        plot_layer_curve(base_mur, mur_mur, "mean", os.path.join(args.out_dir, "mur_mean.png"), "MUR mean")
        plot_layer_curve(
            base_mur,
            mur_mur,
            "frac_below_tau",
            os.path.join(args.out_dir, "mur_frac_below_tau.png"),
            "MUR frac below tau",
        )
        plot_layer_curve(
            base_mur,
            mur_mur,
            "mean_delta_norm",
            os.path.join(args.out_dir, "mur_delta_norm.png"),
            "Delta norm",
        )
        plot_layer_curve(
            base_mur,
            mur_mur,
            "mean_grad_norm",
            os.path.join(args.out_dir, "mur_grad_norm.png"),
            "Grad norm",
        )

    base_bi_path = os.path.join(base_metrics, "bi_metric.json")
    mur_bi_path = os.path.join(mur_metrics, "bi_metric.json")
    if os.path.exists(base_bi_path) and os.path.exists(mur_bi_path):
        base_bi = load_json(base_bi_path)
        mur_bi = load_json(mur_bi_path)
        plot_layer_curve(base_bi, mur_bi, "bi", os.path.join(args.out_dir, "bi_curve.png"), "Block Influence")

    base_drop_path = os.path.join(base_metrics, "layer_drop.json")
    mur_drop_path = os.path.join(mur_metrics, "layer_drop.json")
    if os.path.exists(base_drop_path) and os.path.exists(mur_drop_path):
        base_drop = load_json(base_drop_path)
        mur_drop = load_json(mur_drop_path)
        plot_layer_curve(
            base_drop,
            mur_drop,
            "delta_ppl",
            os.path.join(args.out_dir, "layer_drop_delta_ppl.png"),
            "Layer drop delta PPL",
        )

    base_eval_path = os.path.join(base_metrics, "eval.jsonl")
    mur_eval_path = os.path.join(mur_metrics, "eval.jsonl")
    if os.path.exists(base_eval_path):
        base_eval = load_jsonl(base_eval_path)
        plot_metric_curve(base_eval, "eval_ppl", os.path.join(args.out_dir, "baseline_eval_ppl.png"), "Baseline eval PPL")
    if os.path.exists(mur_eval_path):
        mur_eval = load_jsonl(mur_eval_path)
        plot_metric_curve(mur_eval, "eval_ppl", os.path.join(args.out_dir, "mur_eval_ppl.png"), "MUR eval PPL")


if __name__ == "__main__":
    main()
