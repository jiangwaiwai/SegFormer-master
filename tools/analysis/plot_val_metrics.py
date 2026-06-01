import argparse
import csv
import os.path as osp
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from common import (
    SCIENTIFIC_COLORS,
    ensure_dir,
    infer_default_out_dir,
    load_json_lines,
    save_figure,
    set_plot_style,
    split_train_val,
)


VAL_METRICS = ["mIoU", "mAcc", "aAcc", "oIoU", "P@0.5", "P@0.6", "P@0.7", "P@0.8", "P@0.9"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot validation curves from mmseg log files.")
    parser.add_argument("log_json", help="Path to *.log.json file")
    parser.add_argument(
        "--train-log",
        default=None,
        help="Path to plain text train log for per-class parsing. Default: infer from log_json.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for figures (default: <exp_dir>/figures/val)",
    )
    return parser.parse_args()


def write_val_csv(rows: List[Dict], out_csv: str) -> None:
    if not rows:
        return
    keys = ["iter"] + VAL_METRICS
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in keys})


def plot_main_metrics(val: List[Dict], out_dir: str) -> None:
    iters = [x["iter"] for x in val]
    plt.figure()
    for i, metric in enumerate(["mIoU", "mAcc", "aAcc", "oIoU"]):
        vals = [x.get(metric, np.nan) for x in val]
        plt.plot(iters, vals, color=SCIENTIFIC_COLORS[i], marker="o", label=metric)
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.title("Validation Main Metrics")
    plt.legend()
    save_figure(osp.join(out_dir, "val_main_metrics_curve.png"))


def plot_precision_thresholds(val: List[Dict], out_dir: str) -> None:
    iters = [x["iter"] for x in val]
    keys = ["P@0.5", "P@0.6", "P@0.7", "P@0.8", "P@0.9"]
    plt.figure()
    for i, metric in enumerate(keys):
        vals = [x.get(metric, np.nan) for x in val]
        plt.plot(iters, vals, color=SCIENTIFIC_COLORS[i], marker="o", label=metric)
    plt.xlabel("Iteration")
    plt.ylabel("Score")
    plt.title("Validation Precision@Threshold")
    plt.legend()
    save_figure(osp.join(out_dir, "val_precision_threshold_curve.png"))


def plot_best_so_far(val: List[Dict], out_dir: str) -> None:
    iters = [x["iter"] for x in val]
    miou = [x.get("mIoU", np.nan) for x in val]
    best = np.maximum.accumulate(np.asarray(miou, dtype=float))
    plt.figure()
    plt.plot(iters, miou, color=SCIENTIFIC_COLORS[0], marker="o", alpha=0.6, label="mIoU")
    plt.step(iters, best, where="post", color=SCIENTIFIC_COLORS[3], label="best-so-far mIoU")
    plt.xlabel("Iteration")
    plt.ylabel("mIoU")
    plt.title("Validation mIoU and Best-so-far")
    plt.legend()
    save_figure(osp.join(out_dir, "val_miou_best_so_far.png"))


def plot_precision_profile(val: List[Dict], out_dir: str) -> None:
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    keys = ["P@0.5", "P@0.6", "P@0.7", "P@0.8", "P@0.9"]
    plt.figure()
    n = len(val)
    for idx, row in enumerate(val):
        vals = [row.get(k, np.nan) for k in keys]
        alpha = 0.25 + 0.55 * (idx + 1) / max(n, 1)
        plt.plot(thresholds, vals, color=SCIENTIFIC_COLORS[0], alpha=alpha)
    plt.xlabel("Threshold")
    plt.ylabel("Precision")
    plt.title("Precision-Threshold Profiles (All Validation Points)")
    save_figure(osp.join(out_dir, "val_precision_profile_all_iters.png"))


def parse_per_class_from_train_log(train_log: str) -> List[Dict]:
    if not osp.isfile(train_log):
        return []

    row_re = re.compile(r"^\|\s*(?P<cls>[^|]+?)\s*\|\s*(?P<iou>[\d.]+)\s*\|\s*(?P<acc>[\d.]+)\s*\|$")
    iter_re = re.compile(r"Iter\(val\)\s+\[(?P<iter>\d+)\]")

    records: List[Dict] = []
    pending_rows: List[Tuple[str, float, float]] = []
    capturing = False

    with open(train_log, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if "per class results:" in line:
                capturing = True
                pending_rows = []
                continue

            if capturing:
                m = row_re.match(line)
                if m:
                    cls_name = m.group("cls").strip()
                    if cls_name.lower() != "class":
                        pending_rows.append(
                            (cls_name, float(m.group("iou")), float(m.group("acc")))
                        )

            iter_match = iter_re.search(line)
            if iter_match and pending_rows:
                iter_id = int(iter_match.group("iter"))
                for cls_name, iou, acc in pending_rows:
                    records.append(
                        {
                            "iter": iter_id,
                            "class": cls_name,
                            "IoU": iou,
                            "Acc": acc,
                        }
                    )
                pending_rows = []
                capturing = False
    return records


def write_per_class_csv(records: List[Dict], out_csv: str) -> None:
    if not records:
        return
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["iter", "class", "IoU", "Acc"])
        writer.writeheader()
        writer.writerows(records)


def _build_matrix(records: List[Dict], key: str) -> Tuple[List[int], List[str], np.ndarray]:
    iters = sorted({x["iter"] for x in records})
    classes = sorted({x["class"] for x in records})
    arr = np.full((len(classes), len(iters)), np.nan, dtype=float)
    c2i = {c: i for i, c in enumerate(classes)}
    t2i = {t: i for i, t in enumerate(iters)}
    for r in records:
        arr[c2i[r["class"]], t2i[r["iter"]]] = r[key]
    return iters, classes, arr


def plot_per_class_curves(records: List[Dict], out_dir: str) -> None:
    if not records:
        return
    iters, classes, iou_mat = _build_matrix(records, "IoU")
    _, _, acc_mat = _build_matrix(records, "Acc")

    plt.figure()
    for i, cls_name in enumerate(classes):
        plt.plot(
            iters,
            iou_mat[i],
            marker="o",
            color=SCIENTIFIC_COLORS[i % len(SCIENTIFIC_COLORS)],
            label=cls_name,
        )
    plt.xlabel("Iteration")
    plt.ylabel("IoU (%)")
    plt.title("Per-class IoU Curves")
    plt.legend(ncol=2)
    save_figure(osp.join(out_dir, "val_per_class_iou_curves.png"))

    plt.figure()
    for i, cls_name in enumerate(classes):
        plt.plot(
            iters,
            acc_mat[i],
            marker="o",
            color=SCIENTIFIC_COLORS[i % len(SCIENTIFIC_COLORS)],
            label=cls_name,
        )
    plt.xlabel("Iteration")
    plt.ylabel("Acc (%)")
    plt.title("Per-class Acc Curves")
    plt.legend(ncol=2)
    save_figure(osp.join(out_dir, "val_per_class_acc_curves.png"))


def plot_per_class_heatmaps(records: List[Dict], out_dir: str) -> None:
    if not records:
        return
    iters, classes, iou_mat = _build_matrix(records, "IoU")
    _, _, acc_mat = _build_matrix(records, "Acc")

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(iou_mat, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xticks(range(len(iters)))
    ax.set_xticklabels(iters, rotation=45)
    ax.set_title("Per-class IoU Heatmap")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Class")
    fig.colorbar(im, ax=ax, label="IoU (%)")
    save_figure(osp.join(out_dir, "val_per_class_iou_heatmap.png"), tight=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(acc_mat, aspect="auto", cmap="magma")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xticks(range(len(iters)))
    ax.set_xticklabels(iters, rotation=45)
    ax.set_title("Per-class Acc Heatmap")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Class")
    fig.colorbar(im, ax=ax, label="Acc (%)")
    save_figure(osp.join(out_dir, "val_per_class_acc_heatmap.png"), tight=False)


def plot_best_vs_last_bar(records: List[Dict], val_rows: List[Dict], out_dir: str) -> None:
    if not records or not val_rows:
        return
    best_row = max(val_rows, key=lambda x: x.get("mIoU", -1))
    best_iter = best_row["iter"]
    last_iter = val_rows[-1]["iter"]

    best_items = [x for x in records if x["iter"] == best_iter]
    last_items = [x for x in records if x["iter"] == last_iter]
    if not best_items or not last_items:
        return

    classes = [x["class"] for x in best_items]
    best_iou = [x["IoU"] for x in best_items]
    last_map = {x["class"]: x["IoU"] for x in last_items}
    last_iou = [last_map.get(c, np.nan) for c in classes]

    x = np.arange(len(classes))
    width = 0.38
    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, best_iou, width, color=SCIENTIFIC_COLORS[0], label=f"Best@{best_iter}")
    plt.bar(x + width / 2, last_iou, width, color=SCIENTIFIC_COLORS[1], label=f"Last@{last_iter}")
    plt.xticks(x, classes, rotation=0, ha="center")
    plt.ylabel("IoU (%)")
    plt.title("Per-class IoU: Best vs Last")
    plt.legend()
    save_figure(osp.join(out_dir, "val_per_class_iou_best_vs_last.png"))


def main() -> None:
    args = parse_args()
    set_plot_style()
    rows = load_json_lines(args.log_json)
    _, val = split_train_val(rows)
    if not val:
        raise RuntimeError(f"No val rows found in {args.log_json}")

    out_root = args.out_dir or infer_default_out_dir(args.log_json)
    out_dir = osp.join(out_root, "val")
    ensure_dir(out_dir)

    write_val_csv(val, osp.join(out_dir, "val_metrics.csv"))
    plot_main_metrics(val, out_dir)
    plot_precision_thresholds(val, out_dir)
    plot_best_so_far(val, out_dir)
    plot_precision_profile(val, out_dir)

    train_log = args.train_log or args.log_json.replace(".log.json", ".log")
    per_class = parse_per_class_from_train_log(train_log)
    if per_class:
        write_per_class_csv(per_class, osp.join(out_dir, "val_per_class_metrics.csv"))
        plot_per_class_curves(per_class, out_dir)
        plot_per_class_heatmaps(per_class, out_dir)
        plot_best_vs_last_bar(per_class, val, out_dir)
    else:
        print(f"Skipped per-class plots: train log not found or unparsable ({train_log})")

    print(f"Saved validation figures to: {out_dir}")


if __name__ == "__main__":
    main()
