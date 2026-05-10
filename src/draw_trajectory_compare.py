"""Plot GT vs rollout vs teacher-forced trajectories from a checkpoint."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import get_class
from omegaconf import OmegaConf
from torchdyn.core import NeuralODE

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from draw_trajectory import (  # noqa: E402
    _collate_patient_batch,
    _dataset_for_split_draw,
    _full_traj_and_time,
    _heading_label,
    build_datamodule,
)
from model.components.grad_util import torch_wrapper_tv  # noqa: E402


def _core_ode_module(model: torch.nn.Module) -> torch.nn.Module:
    if hasattr(model, "flow_model"):
        return model.flow_model
    if hasattr(model, "model"):
        return model.model
    raise TypeError(
        "Unsupported model structure. Expected one of: "
        "`flow_model` (Noise_MLP_Cond_Memory_Module) or "
        "`model` (MLP_Cond_Memory_Module)."
    )


def _extract_rollout_pred(test_output):
    if isinstance(test_output, (tuple, list)) and len(test_output) >= 2:
        return test_output[1]
    raise TypeError(
        "Unexpected output from model.test_trajectory(batch). "
        "Expected tuple/list with predicted trajectory at index 1."
    )


def _teacher_forced_trajectory(
    model: torch.nn.Module,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    ode_t_span_points: int,
) -> torch.Tensor:
    if getattr(model, "implementation", "ODE") != "ODE":
        raise ValueError("Teacher-forced ODE comparison currently supports ODE implementation only.")

    ode_core = _core_ode_module(model)
    node = NeuralODE(torch_wrapper_tv(ode_core), solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

    x0_values, x0_classes, x1_values, times_x0, times_x1 = batch
    x0_values = x0_values.squeeze(0)
    x1_values = x1_values.squeeze(0)
    times_x0 = times_x0.squeeze()
    times_x1 = times_x1.squeeze()
    x0_classes = x0_classes.squeeze()

    if len(x0_classes.shape) == 1:
        x0_classes = x0_classes.unsqueeze(1)

    total_pred = [x0_values[0, : model.dim].unsqueeze(0)]
    len_path = x0_values.shape[0]
    if len_path != x1_values.shape[0]:
        raise RuntimeError("Unexpected trajectory lengths in batch.")

    for i in range(len_path):
        time_span = torch.linspace(times_x0[i], times_x1[i], ode_t_span_points).to(x0_values.device).float()
        testpt = torch.cat([x0_values[i, : model.dim].unsqueeze(0), x0_classes[i].unsqueeze(0)], dim=1)
        with torch.no_grad():
            traj = node.trajectory(testpt, t_span=time_span)
        pred_traj = traj[-1, :, : model.dim]
        total_pred.append(pred_traj)

    return torch.cat(total_pred, dim=0)


def _plot_per_dim_three(
    gt: np.ndarray,
    rollout: np.ndarray,
    teacher: np.ndarray,
    t_span: np.ndarray,
    *,
    time_axis_name: str,
    feature_names: list[str],
    title: str,
) -> plt.Figure:
    if gt.shape != rollout.shape or gt.shape != teacher.shape:
        raise ValueError(f"Shape mismatch: gt={gt.shape}, rollout={rollout.shape}, teacher={teacher.shape}")
    n_steps, d = gt.shape
    if len(t_span) != n_steps:
        raise ValueError(f"t_span length {len(t_span)} != trajectory steps {n_steps}")

    fig_h = max(4.0, 3.2 * d)
    fig, axes = plt.subplots(d, 1, figsize=(12, fig_h), sharex=True, squeeze=False)
    ax_list = axes.flatten().tolist()

    for i in range(d):
        ax = ax_list[i]
        ax.plot(t_span, gt[:, i], color="hotpink", linewidth=2, marker="s", markersize=4, alpha=0.9, label="Ground truth")
        ax.plot(
            t_span,
            rollout[:, i],
            color="olive",
            linewidth=2,
            marker="o",
            markersize=4,
            alpha=0.9,
            label="Rollout prediction",
        )
        ax.plot(
            t_span,
            teacher[:, i],
            color="tab:blue",
            linewidth=2,
            marker="^",
            markersize=4,
            alpha=0.9,
            label="Teacher-forced prediction",
        )
        ax.axvline(t_span[0], color="lightgray", linestyle="--", linewidth=1, zorder=0)
        ax.set_ylabel(feature_names[i] if i < len(feature_names) else f"dim_{i}")
        ax.legend(loc="best", fontsize=9)
        ax.grid(True, alpha=0.35)

    ax_list[-1].set_xlabel(time_axis_name)
    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def _plot_3d_three(
    gt: np.ndarray,
    rollout: np.ndarray,
    teacher: np.ndarray,
    t_span: np.ndarray,
    *,
    time_axis_name: str,
    state_axis_names: tuple[str, str],
    title: str,
) -> plt.Figure:
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.plot(t_span, gt[:, 0], gt[:, 1], color="hotpink", linewidth=2, label="Ground truth")
    ax.plot(t_span, rollout[:, 0], rollout[:, 1], color="olive", linewidth=2, label="Rollout prediction")
    ax.plot(t_span, teacher[:, 0], teacher[:, 1], color="tab:blue", linewidth=2, label="Teacher-forced prediction")

    ax.scatter([t_span[0]], [gt[0, 0]], [gt[0, 1]], s=120, c="red", marker="*", zorder=5, label="Start")
    ax.set_xlabel(time_axis_name)
    ax.set_ylabel(state_axis_names[0])
    ax.set_zlabel(state_axis_names[1])
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot GT vs rollout prediction vs teacher-forced prediction on sampled trajectories."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to Hydra config.yaml.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to Lightning checkpoint (.ckpt).")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for PNG figures.")
    parser.add_argument("--split", type=str, choices=("train", "val", "test"), default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=3, help="Number of random patients to plot.")
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu (default: cuda if available).")
    parser.add_argument(
        "--ode-t-span-points",
        type=int,
        default=None,
        metavar="N",
        help="Override ODE integration grid size per interval. Must be >= 2.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    data_module = build_datamodule(cfg)
    data_module.prepare_data()
    data_module.setup(None)

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    model_cls = get_class(cfg.model_module["_target_"])
    model = model_cls.load_from_checkpoint(str(args.ckpt), map_location=device)
    model.eval().to(device)

    if args.ode_t_span_points is not None:
        if args.ode_t_span_points < 2:
            raise SystemExit("--ode-t-span-points must be >= 2.")
        model.ode_t_span_points = int(args.ode_t_span_points)
    ode_ts = int(getattr(model, "ode_t_span_points", 10) or 10)
    print(f"Using ode_t_span_points={ode_ts}")

    time_lbl = _heading_label(data_module.t_headings, 0, "Time")
    feature_names = [_heading_label(data_module.x_headings, i, f"dim_{i}") for i in range(model.dim)]

    dataset = _dataset_for_split_draw(data_module, args.split)
    n = len(dataset)
    k = min(args.num_samples, n)
    if k <= 0:
        raise SystemExit("No samples found in dataset.")
    indices = random.Random(args.seed).sample(range(n), k=k)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_prefix = f"{args.split}_"

    for rank, idx in enumerate(indices):
        batch = _collate_patient_batch(dataset[idx], device)
        with torch.no_grad():
            rollout_out = model.test_trajectory(batch)
        rollout_pred = _extract_rollout_pred(rollout_out)
        teacher_pred = _teacher_forced_trajectory(model, batch, ode_t_span_points=ode_ts)
        full_traj, full_time = _full_traj_and_time(batch, model.dim)

        gt_np = full_traj.detach().cpu().numpy()
        rollout_np = rollout_pred.detach().cpu().numpy()
        teacher_np = teacher_pred.detach().cpu().numpy()
        time_np = full_time.detach().cpu().numpy()

        if not (gt_np.shape == rollout_np.shape == teacher_np.shape) or gt_np.shape[0] != len(time_np):
            raise RuntimeError(
                f"Shape mismatch: gt {gt_np.shape}, rollout {rollout_np.shape}, "
                f"teacher {teacher_np.shape}, time {time_np.shape}"
            )

        base = f"{split_prefix}trajectory_sample_{rank}_idx_{idx}_odeTs{ode_ts}"

        fig_2d = _plot_per_dim_three(
            gt_np,
            rollout_np,
            teacher_np,
            time_np,
            time_axis_name=time_lbl,
            feature_names=feature_names,
            title=f"GT vs rollout vs teacher-forced [{args.split}] (dataset idx {idx})",
        )
        path_2d = args.out_dir / f"{base}_gt_rollout_teacher_per_dim.png"
        fig_2d.savefig(path_2d, dpi=150, bbox_inches="tight")
        plt.close(fig_2d)
        print(f"Saved {path_2d}")

        if model.dim == 2:
            fig_3d = _plot_3d_three(
                gt_np,
                rollout_np,
                teacher_np,
                time_np,
                time_axis_name=time_lbl,
                state_axis_names=(feature_names[0], feature_names[1]),
                title=f"3D GT vs rollout vs teacher-forced [{args.split}] (dataset idx {idx})",
            )
            path_3d = args.out_dir / f"{base}_gt_rollout_teacher_3d.png"
            fig_3d.savefig(path_3d, dpi=150, bbox_inches="tight")
            plt.close(fig_3d)
            print(f"Saved {path_3d}")


if __name__ == "__main__":
    main()

