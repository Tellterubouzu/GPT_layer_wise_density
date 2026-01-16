import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from visualization.plot_metrics import load_json, load_jsonl, plot_layer_curve, plot_metric_curve


def _plot_if_exists(path: str, key: str, out_path: str, title: str, label: str) -> None:
    if not os.path.exists(path):
        return
    records = load_jsonl(path)
    plot_metric_curve(records, key, out_path, title, label=label)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot BI-Floor experiment results")
    parser.add_argument("--baseline_dir", required=True, help="Baseline run directory")
    parser.add_argument("--bifloor_dir", required=True, help="BI-Floor run directory")
    parser.add_argument("--out_dir", required=True, help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    base_metrics = os.path.join(args.baseline_dir, "metrics")
    bifloor_metrics = os.path.join(args.bifloor_dir, "metrics")

    base_bi_path = os.path.join(base_metrics, "bi_metric.json")
    bifloor_bi_path = os.path.join(bifloor_metrics, "bi_metric.json")
    if os.path.exists(base_bi_path) and os.path.exists(bifloor_bi_path):
        base_bi = load_json(base_bi_path)
        bifloor_bi = load_json(bifloor_bi_path)
        plot_layer_curve(
            base_bi,
            bifloor_bi,
            "bi",
            os.path.join(args.out_dir, "bi_curve.png"),
            "Block Influence",
            baseline_label="baseline",
            delta_label="bi-floor",
        )

    base_drop_path = os.path.join(base_metrics, "layer_drop.json")
    bifloor_drop_path = os.path.join(bifloor_metrics, "layer_drop.json")
    if os.path.exists(base_drop_path) and os.path.exists(bifloor_drop_path):
        base_drop = load_json(base_drop_path)
        bifloor_drop = load_json(bifloor_drop_path)
        plot_layer_curve(
            base_drop,
            bifloor_drop,
            "delta_ppl",
            os.path.join(args.out_dir, "layer_drop_delta_ppl.png"),
            "Layer drop delta PPL",
            baseline_label="baseline",
            delta_label="bi-floor",
        )

    base_repr_cos = os.path.join(base_metrics, "repr_cos.json")
    bifloor_repr_cos = os.path.join(bifloor_metrics, "repr_cos.json")
    if os.path.exists(base_repr_cos) and os.path.exists(bifloor_repr_cos):
        base_repr = load_json(base_repr_cos)
        bifloor_repr = load_json(bifloor_repr_cos)
        plot_layer_curve(
            base_repr,
            bifloor_repr,
            "adjacent_cos",
            os.path.join(args.out_dir, "repr_adjacent_cos.png"),
            "Adjacent layer cosine",
            baseline_label="baseline",
            delta_label="bi-floor",
        )

    base_repr_cka = os.path.join(base_metrics, "repr_cka.json")
    bifloor_repr_cka = os.path.join(bifloor_metrics, "repr_cka.json")
    if os.path.exists(base_repr_cka) and os.path.exists(bifloor_repr_cka):
        base_repr = load_json(base_repr_cka)
        bifloor_repr = load_json(bifloor_repr_cka)
        plot_layer_curve(
            base_repr,
            bifloor_repr,
            "cka",
            os.path.join(args.out_dir, "repr_adjacent_cka.png"),
            "Adjacent layer CKA",
            baseline_label="baseline",
            delta_label="bi-floor",
        )

    _plot_if_exists(
        os.path.join(base_metrics, "train.jsonl"),
        "frac_below_tau_mid",
        os.path.join(args.out_dir, "baseline_frac_below_tau.png"),
        "Baseline frac below tau (mid)",
        label="baseline",
    )
    _plot_if_exists(
        os.path.join(bifloor_metrics, "train.jsonl"),
        "frac_below_tau_mid",
        os.path.join(args.out_dir, "bifloor_frac_below_tau.png"),
        "BI-Floor frac below tau (mid)",
        label="bi-floor",
    )

    _plot_if_exists(
        os.path.join(base_metrics, "eval.jsonl"),
        "eval_ppl",
        os.path.join(args.out_dir, "baseline_eval_ppl.png"),
        "Baseline eval PPL",
        label="baseline",
    )
    _plot_if_exists(
        os.path.join(bifloor_metrics, "eval.jsonl"),
        "eval_ppl",
        os.path.join(args.out_dir, "bifloor_eval_ppl.png"),
        "BI-Floor eval PPL",
        label="bi-floor",
    )


if __name__ == "__main__":
    main()
