"""Sample random test patients, load a Lightning checkpoint, plot GT vs predicted trajectories."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import get_class, instantiate
from omegaconf import OmegaConf


_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data.datamodule import PatientDataset

def _heading_label(headings, index: int, default: str) -> str:
    if headings is None:
        return default
    if isinstance(headings, str):
        return headings if index == 0 else default
    try:
        seq = list(headings)
        return str(seq[index])
    except (IndexError, TypeError):
        return default


def plot_trajectory_pred_vs_gt(
    pred_traj: np.ndarray,
    groundtruth: np.ndarray,
    t_span: np.ndarray,
    *,
    time_axis_name: str = "Time",
    state_axis_names: tuple[str, str] = ("dim_0", "dim_1"),
    title: str = "",
):
    """3D path: time vs two state dims, with legend and axis labels."""
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    ax.plot(t_span, pred_traj[:, 0], pred_traj[:, 1], color="olive", linewidth=2, label="Prediction")
    ax.plot(
        t_span, groundtruth[:, 0], groundtruth[:, 1], color="hotpink", linewidth=2, label="Ground truth"
    )

    ax.scatter(t_span, pred_traj[:, 0], pred_traj[:, 1], s=36, alpha=0.65, color="olive", edgecolors="k", linewidths=0.3)
    ax.scatter(
        t_span, groundtruth[:, 0], groundtruth[:, 1], s=36, alpha=0.65, color="purple", edgecolors="k", linewidths=0.3
    )

    # Single start marker at the actual first timestamp (aligned with plotted trajectories).
    ax.scatter(
        [t_span[0]],
        [pred_traj[0, 0]],
        [pred_traj[0, 1]],
        s=120,
        c="red",
        marker="*",
        zorder=5,
        label="Start (t₀, shared x₀)",
    )

    ax.set_xlabel(time_axis_name)
    ax.set_ylabel(state_axis_names[0])
    ax.set_zlabel(state_axis_names[1])
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))

    return fig


def plot_trajectory_per_dimension_2d(
    pred_traj: np.ndarray,
    groundtruth: np.ndarray,
    t_span: np.ndarray,
    *,
    time_axis_name: str = "Time",
    feature_names: list[str] | None = None,
    title: str = "",
) -> plt.Figure:
    """One 2D panel per state dimension: time on x-axis, feature value on y-axis."""
    if pred_traj.shape != groundtruth.shape:
        raise ValueError(f"pred_traj {pred_traj.shape} vs groundtruth {groundtruth.shape}")
    n_steps, d = pred_traj.shape
    if len(t_span) != n_steps:
        raise ValueError(f"t_span length {len(t_span)} != trajectory steps {n_steps}")

    if feature_names is None:
        feature_names = [f"dim_{i}" for i in range(d)]
    elif len(feature_names) != d:
        raise ValueError(f"feature_names length {len(feature_names)} != dim {d}")

    fig_h = max(4.0, 3.2 * d)
    fig, axes = plt.subplots(d, 1, figsize=(12, fig_h), sharex=True, squeeze=False)
    ax_list = axes.flatten().tolist()

    for i in range(d):
        ax = ax_list[i]
        ax.plot(
            t_span,
            pred_traj[:, i],
            color="olive",
            linewidth=2,
            marker="o",
            markersize=5,
            label="Prediction",
        )
        ax.plot(
            t_span,
            groundtruth[:, i],
            color="hotpink",
            linewidth=2,
            marker="s",
            markersize=5,
            alpha=0.85,
            label="Ground truth",
        )
        ax.axvline(t_span[0], color="lightgray", linestyle="--", linewidth=1, zorder=0)
        ax.set_ylabel(feature_names[i])
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.35)

    ax_list[-1].set_xlabel(time_axis_name)
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def plot_uncertainty_per_dimension_2d(
    uncertainty_traj: np.ndarray,
    t_span: np.ndarray,
    *,
    time_axis_name: str = "Time",
    feature_names: list[str] | None = None,
    predicted_uncertainty: np.ndarray | None = None,
    title: str = "",
) -> plt.Figure:
    """One 2D panel per state dimension for uncertainty trajectories."""
    n_steps, d = uncertainty_traj.shape
    if len(t_span) != n_steps:
        raise ValueError(f"t_span length {len(t_span)} != uncertainty steps {n_steps}")
    if predicted_uncertainty is not None and predicted_uncertainty.shape != uncertainty_traj.shape:
        raise ValueError(
            f"predicted_uncertainty {predicted_uncertainty.shape} != uncertainty_traj {uncertainty_traj.shape}"
        )

    if feature_names is None:
        feature_names = [f"dim_{i}" for i in range(d)]
    elif len(feature_names) != d:
        raise ValueError(f"feature_names length {len(feature_names)} != dim {d}")

    fig_h = max(4.0, 3.2 * d)
    fig, axes = plt.subplots(d, 1, figsize=(12, fig_h), sharex=True, squeeze=False)
    ax_list = axes.flatten().tolist()

    for i in range(d):
        ax = ax_list[i]
        ax.plot(
            t_span,
            uncertainty_traj[:, i],
            color="tab:blue",
            linewidth=2,
            marker="o",
            markersize=4,
            label="Uncertainty |GT - Pred|",
        )
        if predicted_uncertainty is not None:
            ax.plot(
                t_span,
                predicted_uncertainty[:, i],
                color="tab:orange",
                linewidth=2,
                marker="^",
                markersize=4,
                alpha=0.85,
                label="Model predicted uncertainty",
            )
        ax.axvline(t_span[0], color="lightgray", linestyle="--", linewidth=1, zorder=0)
        ax.set_ylabel(feature_names[i])
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.35)

    ax_list[-1].set_xlabel(time_axis_name)
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def _collate_patient_batch(sample_tuple: tuple, device: torch.device) -> tuple:
    x0, xc, x1, t0, t1 = sample_tuple
    return (
        torch.from_numpy(np.asarray(x0)).unsqueeze(0).float().to(device),
        torch.from_numpy(np.asarray(xc)).unsqueeze(0).float().to(device),
        torch.from_numpy(np.asarray(x1)).unsqueeze(0).float().to(device),
        torch.from_numpy(np.asarray(t0)).unsqueeze(0).float().to(device),
        torch.from_numpy(np.asarray(t1)).unsqueeze(0).float().to(device),
    )


def build_datamodule(cfg):
    """Mirror main.py datamodule construction (memory sync, dim, treatment_cond, Cond train_consecutive)."""
    if "memory" in cfg.model_module.keys():
        cfg.data_module.memory = cfg.model_module.memory

    data_module = instantiate(cfg.data_module)

    x_dim = data_module.dims[0]
    if "dim" in cfg.model_module.keys():
        cfg.model_module.dim = x_dim
    elif "input_dim" in cfg.model_module.keys():
        cfg.model_module.input_dim = x_dim
        cfg.model_module.output_dim = x_dim

    if "treatment_cond" in cfg.model_module.keys():
        cfg.model_module.treatment_cond = len(data_module.cond_headings)

    model_meta = instantiate(cfg.model_module)

    if "Cond" not in model_meta.naming:
        cfg.data_module.train_consecutive = True
        data_module = instantiate(cfg.data_module)
    else:
        cfg.data_module.train_consecutive = False
        data_module = instantiate(cfg.data_module)

    return data_module


def _full_traj_and_time(batch: tuple, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    x0_values, _x0_classes, x1_values, times_x0, times_x1 = batch
    times_x0 = times_x0.squeeze()
    times_x1 = times_x1.squeeze()
    full_traj = torch.cat(
        [x0_values[0, 0, :dim].unsqueeze(0), x1_values[0, :, :dim]],
        dim=0,
    )
    full_time = torch.cat([times_x0[0].unsqueeze(0), times_x1], dim=0)
    return full_traj, full_time


def _dataset_for_split_draw(data_module, split: str) -> PatientDataset:
    """Patient-level dataset for inference (same structure val/test use).

    When ``train_consecutive`` is False (conditional models), ``train_dataloader`` uses
    pair-wise ``TrainingDataset``; for trajectory plotting we instead build full patients
    from ``data_module.train`` like validation does.
    """
    if split == "train":
        if data_module.train_consecutive:
            return data_module.train_dataloader(shuffle=False).dataset
        patient_rows = data_module.create_patient_data(data_module.train)
        return PatientDataset(patient_rows)
    if split == "val":
        return data_module.val_dataloader().dataset
    if split == "test":
        return data_module.test_dataloader().dataset
    raise ValueError(f"Unknown split: {split}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot GT vs predicted trajectories on random samples from a split.")
    parser.add_argument("--config", type=Path, required=True, help="Path to Hydra config.yaml (e.g. outputs/.../.hydra/config.yaml)")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to Lightning checkpoint (.ckpt)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory to save PNG figures")
    parser.add_argument(
        "--split",
        type=str,
        choices=("train", "val", "test"),
        default="test",
        help="Which split to sample patients from (train, val, or test). Output filenames are prefixed with this name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda | cpu (default: cuda if available)",
    )
    parser.add_argument(
        "--ode-t-span-points",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Override integration grid size (torch.linspace(..., N) per interval in test_trajectory). "
            "Must be >= 2. Default: value from checkpoint / model hyperparameter."
        ),
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    data_module = build_datamodule(cfg)
    data_module.prepare_data()
    # Load train/val/test so any --split works (setup('test') alone skips train/val).
    data_module.setup(None)

    split = args.split
    split_prefix = f"{split}_"

    device_str = args.device
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    model_cls = get_class(cfg.model_module._target_)
    model = model_cls.load_from_checkpoint(str(args.ckpt), map_location=device)
    model.eval()
    model.to(device)

    if args.ode_t_span_points is not None:
        if args.ode_t_span_points < 2:
            raise SystemExit("--ode-t-span-points must be >= 2.")
        if not hasattr(model, "ode_t_span_points"):
            raise SystemExit(
                "This checkpoint's model has no attribute ode_t_span_points "
                "(not supported for this architecture)."
            )
        model.ode_t_span_points = int(args.ode_t_span_points)

    ode_ts = int(getattr(model, "ode_t_span_points", 10))
    print(f"Using ode_t_span_points={ode_ts} (torch.linspace grid length per interval)")

    time_lbl = _heading_label(data_module.t_headings, 0, "Time")
    feature_names = [_heading_label(data_module.x_headings, i, f"dim_{i}") for i in range(model.dim)]

    dataset = _dataset_for_split_draw(data_module, split)
    n = len(dataset)
    k = min(3, n)
    indices = random.Random(args.seed).sample(range(n), k=k)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for rank, idx in enumerate(indices):
        batch = _collate_patient_batch(dataset[idx], device)
        with torch.no_grad():
            out = model.test_trajectory(batch)
        pred_traj = out[1]

        full_traj, full_time = _full_traj_and_time(batch, model.dim)

        pred_np = pred_traj.detach().cpu().numpy()
        full_np = full_traj.detach().cpu().numpy()
        time_np = full_time.detach().cpu().numpy()
        uncertainty_np = np.abs(full_np - pred_np)
        predicted_uncertainty_np = None
        if len(out) >= 4:
            noise_traj = out[3]
            noise_np = noise_traj.detach().cpu().numpy()
            if noise_np.shape[1] == model.dim and noise_np.shape[0] == pred_np.shape[0] - 1:
                predicted_uncertainty_np = np.vstack([np.zeros((1, model.dim), dtype=noise_np.dtype), noise_np])

        if pred_np.shape[0] != full_np.shape[0] or pred_np.shape[0] != len(time_np):
            raise RuntimeError(
                f"Shape mismatch: pred {pred_np.shape}, gt {full_np.shape}, time {time_np.shape}"
            )

        base = f"{split_prefix}trajectory_sample_{rank}_idx_{idx}_odeTs{ode_ts}"

        if model.dim == 2:
            fig_3d = plot_trajectory_pred_vs_gt(
                pred_np,
                full_np,
                time_np,
                time_axis_name=time_lbl,
                state_axis_names=(feature_names[0], feature_names[1]),
                title=f"3D [{split}] trajectory dataset idx {idx}",
            )
            path_3d = args.out_dir / f"{base}.png"
            fig_3d.savefig(path_3d, dpi=150, bbox_inches="tight")
            plt.close(fig_3d)
            print(f"Saved {path_3d}")

        fig_2d = plot_trajectory_per_dimension_2d(
            pred_np,
            full_np,
            time_np,
            time_axis_name=time_lbl,
            feature_names=feature_names,
            title=f"Per-dimension trajectories [{split}] (dataset idx {idx})",
        )
        path_2d = args.out_dir / f"{base}_per_dim.png"
        fig_2d.savefig(path_2d, dpi=150, bbox_inches="tight")
        plt.close(fig_2d)
        print(f"Saved {path_2d}")

        fig_unc = plot_uncertainty_per_dimension_2d(
            uncertainty_np,
            time_np,
            time_axis_name=time_lbl,
            feature_names=feature_names,
            predicted_uncertainty=predicted_uncertainty_np,
            title=f"Per-dimension uncertainty trajectories [{split}] (dataset idx {idx})",
        )
        path_unc = args.out_dir / f"{base}_uncertainty_per_dim.png"
        fig_unc.savefig(path_unc, dpi=150, bbox_inches="tight")
        plt.close(fig_unc)
        print(f"Saved {path_unc}")


if __name__ == "__main__":
    main()
