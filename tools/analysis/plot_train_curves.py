import argparse
import csv
import os.path as osp
from typing import Dict, List

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training curves from mmseg log.json.")
    parser.add_argument("log_json", help="Path to *.log.json file")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for figures (default: <exp_dir>/figures/train)",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=1,
        help="Moving average window for train curves (default: 1, no smoothing).",
    )
    return parser.parse_args()


def moving_average(values: List[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if window <= 1 or window > len(arr):
        return arr
    kernel = np.ones(window) / window
    valid = np.convolve(arr, kernel, mode="valid")
    head = np.full(window - 1, valid[0])
    return np.concatenate([head, valid])


def write_train_csv(rows: List[Dict], out_csv: str) -> None:
    if not rows:
        return
    keys = [
        "iter",
        "lr",
        "loss",
        "decode.loss_seg",
        "decode.acc_seg",
        "time",
        "data_time",
        "memory",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in keys})


def plot_loss(train: List[Dict], out_dir: str, smooth_window: int) -> None:
    iters = [x["iter"] for x in train]
    loss = [x.get("loss") for x in train]
    seg_loss = [x.get("decode.loss_seg") for x in train]

    plt.figure()
    plt.plot(iters, moving_average(loss, smooth_window), color=SCIENTIFIC_COLORS[0], label="loss")
    if any(v is not None for v in seg_loss):
        seg_vals = [float(v) if v is not None else np.nan for v in seg_loss]
        plt.plot(
            iters,
            moving_average(seg_vals, smooth_window),
            color=SCIENTIFIC_COLORS[1],
            linestyle="--",
            label="decode.loss_seg",
        )
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.legend()
    save_figure(osp.join(out_dir, "train_loss_curve.png"))


def plot_accuracy(train: List[Dict], out_dir: str, smooth_window: int) -> None:
    iters = [x["iter"] for x in train]
    acc = [x.get("decode.acc_seg") for x in train]
    acc = [float(v) if v is not None else np.nan for v in acc]
    plt.figure()
    # Single-metric figure uses blue for a cleaner scientific style.
    plt.plot(iters, moving_average(acc, smooth_window), color=SCIENTIFIC_COLORS[0], label="decode.acc_seg")
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy (%)")
    plt.title("Training Pixel Accuracy Curve")
    plt.legend()
    save_figure(osp.join(out_dir, "train_acc_curve.png"))


def plot_lr(train: List[Dict], out_dir: str) -> None:
    iters = [x["iter"] for x in train]
    lr = [x.get("lr") for x in train]
    plt.figure()
    # Single-metric figure uses blue for a cleaner scientific style.
    plt.plot(iters, lr, color=SCIENTIFIC_COLORS[0], label="lr")
    plt.xlabel("Iteration")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.legend()
    save_figure(osp.join(out_dir, "train_lr_curve.png"))


def plot_timing(train: List[Dict], out_dir: str, smooth_window: int) -> None:
    iters = [x["iter"] for x in train]
    total = np.asarray([x.get("time", np.nan) for x in train], dtype=float)
    data = np.asarray([x.get("data_time", np.nan) for x in train], dtype=float)
    compute = np.maximum(total - data, 0.0)

    plt.figure()
    plt.plot(iters, moving_average(total.tolist(), smooth_window), color=SCIENTIFIC_COLORS[0], label="time")
    plt.plot(
        iters,
        moving_average(data.tolist(), smooth_window),
        color=SCIENTIFIC_COLORS[1],
        label="data_time",
    )
    plt.plot(
        iters,
        moving_average(compute.tolist(), smooth_window),
        color=SCIENTIFIC_COLORS[2],
        label="compute_time≈time-data_time",
    )
    plt.xlabel("Iteration")
    plt.ylabel("Seconds")
    plt.title("Training Time Breakdown")
    plt.legend()
    save_figure(osp.join(out_dir, "train_time_breakdown_curve.png"))


def plot_memory(train: List[Dict], out_dir: str) -> None:
    iters = [x["iter"] for x in train]
    mem = [x.get("memory") for x in train]
    plt.figure()
    # Single-metric figure uses blue for a cleaner scientific style.
    plt.plot(iters, mem, color=SCIENTIFIC_COLORS[0], label="memory")
    plt.xlabel("Iteration")
    plt.ylabel("Memory (MB)")
    plt.title("GPU Memory Curve")
    plt.legend()
    save_figure(osp.join(out_dir, "train_memory_curve.png"))


def plot_joint(train: List[Dict], out_dir: str, smooth_window: int) -> None:
    iters = [x["iter"] for x in train]
    loss = moving_average([float(x.get("loss", np.nan)) for x in train], smooth_window)
    acc = moving_average([float(x.get("decode.acc_seg", np.nan)) for x in train], smooth_window)
    lr = np.asarray([float(x.get("lr", np.nan)) for x in train], dtype=float)

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(iters, loss, color=SCIENTIFIC_COLORS[0], label="loss")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss", color=SCIENTIFIC_COLORS[0])
    ax1.tick_params(axis="y", labelcolor=SCIENTIFIC_COLORS[0])

    ax2 = ax1.twinx()
    line2 = ax2.plot(iters, acc, color=SCIENTIFIC_COLORS[2], label="decode.acc_seg")
    ax2.set_ylabel("Accuracy (%)", color=SCIENTIFIC_COLORS[2])
    ax2.tick_params(axis="y", labelcolor=SCIENTIFIC_COLORS[2])

    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("axes", 1.12))
    line3 = ax3.plot(iters, lr, color=SCIENTIFIC_COLORS[3], linestyle="--", label="lr")
    ax3.set_ylabel("LR", color=SCIENTIFIC_COLORS[3])
    ax3.tick_params(axis="y", labelcolor=SCIENTIFIC_COLORS[3])

    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")
    plt.title("Training Loss / Acc / LR")
    save_figure(osp.join(out_dir, "train_joint_loss_acc_lr.png"))


def main() -> None:
    args = parse_args()
    set_plot_style()
    rows = load_json_lines(args.log_json)
    train, _ = split_train_val(rows)
    if not train:
        raise RuntimeError(f"No train rows found in {args.log_json}")

    out_root = args.out_dir or infer_default_out_dir(args.log_json)
    out_dir = osp.join(out_root, "train")
    ensure_dir(out_dir)

    write_train_csv(train, osp.join(out_dir, "train_metrics.csv"))
    plot_loss(train, out_dir, args.smooth_window)
    plot_accuracy(train, out_dir, args.smooth_window)
    plot_lr(train, out_dir)
    plot_timing(train, out_dir, args.smooth_window)
    plot_memory(train, out_dir)
    plot_joint(train, out_dir, args.smooth_window)
    print(f"Saved training figures to: {out_dir}")


if __name__ == "__main__":
    main()
