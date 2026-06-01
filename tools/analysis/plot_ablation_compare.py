import argparse
import csv
import json
import os
import os.path as osp
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from common import (
    SCIENTIFIC_COLORS,
    ensure_dir,
    load_json_lines,
    save_figure,
    set_plot_style,
    split_train_val,
)


BEST_KEYS = ["mIoU", "mAcc", "aAcc", "oIoU", "P@0.5", "P@0.6", "P@0.7", "P@0.8", "P@0.9"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ablation comparison figures.")
    parser.add_argument(
        "--exp-dirs",
        nargs="+",
        required=True,
        help="Experiment directories, each containing *.log.json and best_metrics.json",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Display labels for experiments. Default: basename(exp_dir).",
    )
    parser.add_argument(
        "--out-dir",
        default="work_dirs/figures_compare",
        help="Output directory for comparison figures",
    )
    return parser.parse_args()


def find_log_json(exp_dir: str) -> str:
    candidates = [x for x in sorted(os.listdir(exp_dir)) if x.endswith(".log.json")]
    if not candidates:
        raise FileNotFoundError(f"No *.log.json in {exp_dir}")
    return osp.join(exp_dir, candidates[0])


def load_best_metrics(exp_dir: str) -> Dict:
    path = osp.join(exp_dir, "best_metrics.json")
    if not osp.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_experiment(exp_dir: str, label: str) -> Dict:
    log_json = find_log_json(exp_dir)
    rows = load_json_lines(log_json)
    _, val = split_train_val(rows)
    best = load_best_metrics(exp_dir)
    return {"exp_dir": exp_dir, "label": label, "val": val, "best": best}


def plot_miou_curves(exps: List[Dict], out_dir: str) -> None:
    plt.figure()
    for i, exp in enumerate(exps):
        val = exp["val"]
        iters = [x["iter"] for x in val]
        vals = [x.get("mIoU", np.nan) for x in val]
        plt.plot(
            iters,
            vals,
            marker="o",
            color=SCIENTIFIC_COLORS[i % len(SCIENTIFIC_COLORS)],
            label=exp["label"],
        )
    plt.xlabel("Iteration")
    plt.ylabel("mIoU")
    plt.title("Ablation: Validation mIoU Curves")
    plt.legend()
    save_figure(osp.join(out_dir, "ablation_miou_curves.png"))


def collect_best_scores(exps: List[Dict], keys: List[str]) -> Tuple[List[str], Dict[str, List[float]]]:
    labels = [x["label"] for x in exps]
    score_map: Dict[str, List[float]] = {}
    for key in keys:
        vals = []
        for exp in exps:
            best = exp["best"]
            metrics = best.get("metrics", {})
            vals.append(float(metrics.get(key, np.nan)))
        score_map[key] = vals
    return labels, score_map


def _plot_grouped_bars_by_metric(
    exps: List[Dict], keys: List[str], out_path: str, title: str
) -> None:
    # x-axis = metric, bars = experiments (clearer side-by-side comparison)
    labels = [x["label"] for x in exps]
    _, score_map = collect_best_scores(exps, keys)
    x = np.arange(len(keys))
    n_exp = len(exps)
    width = min(0.80 / max(n_exp, 1), 0.34)
    start = -(n_exp - 1) / 2.0

    plt.figure(figsize=(11, 6))
    for i, exp_label in enumerate(labels):
        values = [score_map[k][i] for k in keys]
        offset = (start + i) * width
        bars = plt.bar(
            x + offset,
            values,
            width=width,
            color=SCIENTIFIC_COLORS[i % len(SCIENTIFIC_COLORS)],
            label=exp_label,
        )
        # Annotate values lightly for clearer comparison.
        for b in bars:
            h = b.get_height()
            if np.isfinite(h):
                plt.text(
                    b.get_x() + b.get_width() / 2,
                    h + 0.003,
                    f"{h:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    plt.xticks(x, keys, rotation=0, ha="center")
    plt.ylabel("Score")
    plt.title(title)
    plt.legend()
    plt.ylim(0, 1.02)
    save_figure(out_path)


def plot_best_metric_bars(exps: List[Dict], out_dir: str) -> None:
    _plot_grouped_bars_by_metric(
        exps=exps,
        keys=["mIoU", "mAcc", "aAcc", "oIoU"],
        out_path=osp.join(out_dir, "ablation_best_main_metrics.png"),
        title="Ablation: Best Main Metrics",
    )


def plot_best_precision_bars(exps: List[Dict], out_dir: str) -> None:
    _plot_grouped_bars_by_metric(
        exps=exps,
        keys=["P@0.5", "P@0.6", "P@0.7", "P@0.8", "P@0.9"],
        out_path=osp.join(out_dir, "ablation_best_precision_thresholds.png"),
        title="Ablation: Best Precision@Threshold",
    )


def write_summary_csv(exps: List[Dict], out_csv: str) -> None:
    keys = ["label", "best_step"] + BEST_KEYS
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for exp in exps:
            best = exp["best"]
            metrics = best.get("metrics", {})
            row = {
                "label": exp["label"],
                "best_step": best.get("best_step", ""),
            }
            for k in BEST_KEYS:
                row[k] = metrics.get(k, "")
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    set_plot_style()
    ensure_dir(args.out_dir)

    if args.labels is not None and len(args.labels) != len(args.exp_dirs):
        raise ValueError("--labels count must equal --exp-dirs count")

    labels = args.labels or [osp.basename(osp.abspath(x)) for x in args.exp_dirs]
    exps = [load_experiment(exp, label) for exp, label in zip(args.exp_dirs, labels)]

    write_summary_csv(exps, osp.join(args.out_dir, "ablation_summary.csv"))
    plot_miou_curves(exps, args.out_dir)
    plot_best_metric_bars(exps, args.out_dir)
    plot_best_precision_bars(exps, args.out_dir)
    print(f"Saved ablation comparison figures to: {osp.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
