import json
import os
import os.path as osp
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


SCIENTIFIC_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # gray
    "#bcbd22",  # olive
    "#17becf",  # cyan
]


def set_plot_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "axes.unicode_minus": False,
            "figure.figsize": (10, 6),
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "grid.alpha": 0.35,
            "lines.linewidth": 2.0,
        }
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def infer_default_out_dir(log_json: str) -> str:
    return osp.join(osp.dirname(osp.abspath(log_json)), "figures")


def load_json_lines(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(obj)
    return rows


def split_train_val(rows: Iterable[Dict]) -> Tuple[List[Dict], List[Dict]]:
    train = [r for r in rows if r.get("mode") == "train" and "iter" in r]
    val = [r for r in rows if r.get("mode") == "val" and "iter" in r]
    train.sort(key=lambda x: x["iter"])
    val.sort(key=lambda x: x["iter"])
    return train, val


def save_figure(path: str, tight: bool = True) -> None:
    if tight:
        plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

