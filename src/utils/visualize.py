import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def _to_numpy(values):
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def plot_3d_path_ind(traj, groundtruth, t_span=torch.linspace(0, 4 * np.pi, 100), title=""):
    traj = _to_numpy(traj)
    groundtruth = _to_numpy(groundtruth)
    t_span = _to_numpy(t_span)

    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    ax1.plot(t_span, traj[:, 0], traj[:, 1], alpha=1, c="olive", label="Prediction")
    ax1.plot(t_span, groundtruth[:, 0], groundtruth[:, 1], alpha=1, c="pink", label="Ground Truth")
    ax1.scatter([t_span[0]], [traj[0, 0]], [traj[0, 1]], alpha=0.8, c="red", label="Prediction Start")
    ax1.scatter(t_span, traj[:, 0], traj[:, 1], alpha=0.5, c="blue", label="Prediction Points")
    ax1.scatter(t_span, groundtruth[:, 0], groundtruth[:, 1], alpha=0.5, c="purple", label="Ground Truth Points")
    ax1.set_title(title)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Dim 0')
    ax1.set_zlabel('Dim 1')
    ax1.legend()

    return fig


def plot_3d_path_ind_noise(traj, groundtruth, noise, t_span=torch.linspace(0, 4 * np.pi, 100), title=""):
    traj = _to_numpy(traj)
    groundtruth = _to_numpy(groundtruth)
    noise = _to_numpy(noise)
    t_span = _to_numpy(t_span)

    n = len(t_span)
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    
    ax1.scatter([t_span[0]], [traj[0, 0]], [traj[0, 1]], alpha=0.8, c="red", label="Prediction Start")

    # Plot trajectory and ground truth
    ax1.plot(t_span, traj[:, 0], traj[:, 1], label='Prediction', c='olive')
    ax1.plot(t_span, groundtruth[:, 0], groundtruth[:, 1], label='Ground Truth', c='pink')
    ax1.scatter(t_span, traj[:, 0], traj[:, 1], alpha=0.5, c="blue", label="Prediction Points")
    ax1.scatter(t_span, groundtruth[:, 0], groundtruth[:, 1], alpha=0.5, c="purple", label="Ground Truth Points")
    # Plot uncertainty as scatter points around each trajectory point
    # Plus and minus noise values for visualization
    for i in range(n-1):
        if i == 0:
            continue
        x_noise_pos = traj[i+1, 0] + noise[i, 0]
        y_noise_pos = traj[i+1, 1] + noise[i, 1]
        x_noise_neg = traj[i+1, 0] - noise[i, 0]
        y_noise_neg = traj[i+1, 1] - noise[i, 1]
        label = "Prediction +/- Noise" if i == 1 else None
        ax1.scatter([t_span[i]]*2, [x_noise_pos, x_noise_neg], [y_noise_pos, y_noise_neg], color='gray', alpha=0.5, label=label)

    ax1.set_title(title)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Dim 0')
    ax1.set_zlabel('Dim 1')
    ax1.legend()

    return fig


def plot_trajectory_dimensions(traj, groundtruth, t_span=torch.linspace(0, 4 * np.pi, 100), title=""):
    traj = _to_numpy(traj)
    groundtruth = _to_numpy(groundtruth)
    t_span = _to_numpy(t_span)

    if traj.shape != groundtruth.shape:
        raise ValueError(f"traj and groundtruth must have the same shape, got {traj.shape} and {groundtruth.shape}")
    if traj.ndim != 2:
        raise ValueError(f"traj and groundtruth must be 2D arrays, got ndim={traj.ndim}")
    if len(t_span) != traj.shape[0]:
        raise ValueError(f"t_span length must match trajectory length, got {len(t_span)} and {traj.shape[0]}")

    dim = traj.shape[1]
    fig, axes = plt.subplots(dim, 1, figsize=(12, 3.5 * dim), sharex=True, squeeze=False)
    for idx, ax in enumerate(axes[:, 0]):
        ax.plot(t_span, traj[:, idx], color="olive", label="Prediction")
        ax.plot(t_span, groundtruth[:, idx], color="pink", label="Ground Truth")
        ax.scatter(t_span, traj[:, idx], color="blue", alpha=0.5, s=20)
        ax.scatter(t_span, groundtruth[:, idx], color="purple", alpha=0.5, s=20)
        ax.set_title(f"Dimension {idx}")
        ax.set_xlabel("Time")
        ax.set_ylabel(f"Dim {idx}")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    return fig



def join_3d_plots(figs, rows, cols):
    new_fig = plt.figure(figsize=(15 * cols, 10 * rows))
    for i, fig in enumerate(figs):
        ax = new_fig.add_subplot(rows, cols, i + 1, projection='3d')
        original_ax = fig.get_children()[1]
        for line in original_ax.get_lines():
            ax.add_line(line)
        for patch in original_ax.get_patches():
            ax.add_patch(patch)
    return new_fig