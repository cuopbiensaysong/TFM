import json
import os
from datetime import datetime
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np


def summarize_array(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def compute_pair_variables(
    x0_values: np.ndarray,
    x1_values: np.ndarray,
    times_x0: np.ndarray,
    times_x1: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    x0 = np.asarray(x0_values, dtype=np.float32)
    x1 = np.asarray(x1_values, dtype=np.float32)
    t0 = np.asarray(times_x0, dtype=np.float32).reshape(-1)
    t1 = np.asarray(times_x1, dtype=np.float32).reshape(-1)

    delta_time = t1 - t0
    delta_x = x1 - x0
    zero_dt_mask = np.abs(delta_time) < eps
    safe_dt = np.where(zero_dt_mask, eps, delta_time)
    velocity = delta_x / safe_dt[:, None]

    return delta_time, delta_x, velocity, int(np.sum(zero_dt_mask))


def _maybe_subsample(values: np.ndarray, max_points: int = 500000) -> np.ndarray:
    if values.shape[0] <= max_points:
        return values
    idx = np.linspace(0, values.shape[0] - 1, num=max_points, dtype=np.int64)
    return values[idx]


def _plot_hist_1d(values: np.ndarray, title: str, out_path: str, bins: int = 80) -> None:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    finite = _maybe_subsample(finite)
    plt.figure(figsize=(8, 5))
    plt.hist(finite, bins=bins)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _plot_hist_per_dim(values: np.ndarray, title_prefix: str, out_path: str, bins: int = 80) -> None:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    n_dim = arr.shape[1]
    fig, axes = plt.subplots(n_dim, 1, figsize=(8, max(4, 3 * n_dim)), squeeze=False)
    for i in range(n_dim):
        finite = arr[:, i][np.isfinite(arr[:, i])]
        finite = _maybe_subsample(finite)
        axes[i, 0].hist(finite, bins=bins)
        axes[i, 0].set_title(f"{title_prefix} (dim {i})")
        axes[i, 0].set_xlabel("Value")
        axes[i, 0].set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_audit_outputs(
    dataset_name: str,
    out_dir: str,
    delta_time: np.ndarray,
    delta_x: np.ndarray,
    velocity: np.ndarray,
    zero_dt_count: int,
) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"{dataset_name}_{ts}"

    stats = {
        "dataset_name": dataset_name,
        "num_pairs": int(np.asarray(delta_time).shape[0]),
        "zero_delta_time_count": int(zero_dt_count),
        "delta_time": summarize_array(delta_time),
        "delta_x_overall": summarize_array(delta_x),
        "velocity_overall": summarize_array(velocity),
        "delta_x_per_dim": [summarize_array(delta_x[:, d]) for d in range(delta_x.shape[1])],
        "velocity_per_dim": [summarize_array(velocity[:, d]) for d in range(velocity.shape[1])],
    }

    json_path = os.path.join(out_dir, f"{prefix}_stats.json")
    txt_path = os.path.join(out_dir, f"{prefix}_summary.txt")
    dt_hist_path = os.path.join(out_dir, f"{prefix}_delta_time_hist.png")
    dx_hist_path = os.path.join(out_dir, f"{prefix}_delta_x_hist_per_dim.png")
    vel_hist_path = os.path.join(out_dir, f"{prefix}_velocity_hist_per_dim.png")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {stats['dataset_name']}\n")
        f.write(f"Num pairs: {stats['num_pairs']}\n")
        f.write(f"Zero delta_time count: {stats['zero_delta_time_count']}\n\n")
        f.write(f"delta_time: {stats['delta_time']}\n")
        f.write(f"delta_x_overall: {stats['delta_x_overall']}\n")
        f.write(f"velocity_overall: {stats['velocity_overall']}\n")
        f.write(f"delta_x_per_dim: {stats['delta_x_per_dim']}\n")
        f.write(f"velocity_per_dim: {stats['velocity_per_dim']}\n")

    _plot_hist_1d(delta_time, f"{dataset_name}: delta_time", dt_hist_path)
    _plot_hist_per_dim(delta_x, f"{dataset_name}: delta_x", dx_hist_path)
    _plot_hist_per_dim(velocity, f"{dataset_name}: velocity", vel_hist_path)

    return {
        "stats_json": json_path,
        "summary_txt": txt_path,
        "delta_time_hist": dt_hist_path,
        "delta_x_hist_per_dim": dx_hist_path,
        "velocity_hist_per_dim": vel_hist_path,
    }
