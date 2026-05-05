# %%
import os
import pandas as pd
import numpy as np

import torch

import matplotlib.pyplot as plt
import pickle


def _load_training_symbols_from_example():
    """
    Load reusable training symbols from example.py without executing
    its final top-level training call.
    """
    example_path = os.path.join(os.path.dirname(__file__), "example.py")
    with open(example_path, "r", encoding="utf-8") as f:
        source = f.read()
    cutoff = source.find("\ntrain_model(model, False, train_loader_1d_3m")
    safe_source = source[:cutoff] if cutoff != -1 else source
    namespace = {}
    exec(safe_source, namespace)
    return (
        namespace["eICUDataLoader"],
        namespace["MLP_Cond_Memory_Module"],
        namespace["train_model"],
        namespace["test_model"],
        namespace["test_func_step"],
    )


eICUDataLoader, MLP_Cond_Memory_Module, _, _, test_func_step = _load_training_symbols_from_example()


def test_model_irregular(model, noise_prediction, test_loader, device, save_path=None):
    """Irregular-time evaluation using per-trajectory time axis."""
    model.eval()
    dict_full_trajs = {}
    dict_pred_trajs = {}
    dict_times = {}
    loss_sum = 0.0

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch = [x.to(device) for x in batch]
            loss, pairs, _, _, _, full_time = test_func_step(batch, batch_idx, model, noise_prediction)
            full_traj = pairs[0][0].detach().cpu().numpy()
            pred_traj = pairs[0][1].detach().cpu().numpy()
            dict_full_trajs[batch_idx] = np.squeeze(full_traj)
            dict_pred_trajs[batch_idx] = np.squeeze(pred_traj)
            dict_times[batch_idx] = full_time
            loss_sum += loss

        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111)
        colors = ["red", "green", "blue"]
        for count, idx in enumerate(dict_pred_trajs.keys()):
            full_time = dict_times[idx]
            time_axis = full_time[model.memory:]
            y_pred = np.atleast_1d(dict_pred_trajs[idx])
            n = min(len(time_axis), len(y_pred))
            ax.plot(time_axis[:n], y_pred[:n], label="predicted" if count == 0 else "", color=colors[count % len(colors)])
        for count, idx in enumerate(dict_full_trajs.keys()):
            full_time = dict_times[idx]
            time_axis = full_time[model.memory:]
            y_true = np.atleast_1d(dict_full_trajs[idx])
            n = min(len(time_axis), len(y_true))
            ax.plot(time_axis[:n], y_true[:n], label="ground truth" if count == 0 else "", color=colors[count % len(colors)], linestyle="--")
        ax.set_xlabel("scaled t", fontsize="10")
        ax.set_ylabel("x", fontsize="10")
        ax.legend(fontsize="10")
        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close()

    print(f"validation loss: {loss_sum / len(test_loader)}")


def train_model_irregular(
    model,
    noise_prediction,
    train_loader,
    val_loader=None,
    num_epochs=10,
    device="cpu",
    plot_dir=".",
    val_every=100,
):
    """
    Train model and run irregular validation plotting periodically.
    """
    model.to(device)
    optimizer = model.configure_optimizers()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x0, x0_class, x1, x0_time, x1_time = [b.to(device) for b in batch]

            x, ut, t_model, futuretime, t = model.__x_processing__(x0, x1, x0_time, x1_time)
            in_tensor = torch.cat([x, x0_class, t_model], dim=-1)
            xt = model.model.forward_train(in_tensor)
            loss = model.loss_fn(xt[:, :model.dim], x1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        if (epoch + 1) % val_every == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss/len(train_loader)}")
            if val_loader is not None:
                save_path = os.path.join(plot_dir, f"validation_step{epoch+1}.png")
                test_model_irregular(
                    model,
                    noise_prediction,
                    val_loader,
                    device,
                    save_path=save_path,
                )


def damped_harmonic_oscillator(x0, v0, m, k, c, t_max=10, dt=0.1):
    """
    Simulate the trajectory of a damped harmonic oscillator
    simple newton method solver

    Args:
        x0 (float): Initial position
        v0 (float): Initial velocity
        m (float): Mass
        k (float): Spring constant
        c (float): Viscous force coefficient
        t_max (float, optional): Maximum time to simulate (default: 10)
        dt (float, optional): Time step (default: 0.01)

    Returns:
        numpy.ndarray: Array of position values
        numpy.ndarray: Array of time values
    """
    t = np.arange(0, t_max, dt)
    x = np.zeros_like(t)
    v = np.zeros_like(t)

    x[0] = x0
    v[0] = v0

    for i in range(1, len(t)):
        a = -c * v[i-1] / m - k * x[i-1] / m  # Acceleration
        v[i] = v[i-1] + a * dt  # Update velocity
        x[i] = x[i-1] + v[i-1] * dt  # Update position

    return x, t

def get_damping_zone(m,k,c):
    zeta = c / (2 * np.sqrt(k * m))
    if zeta > 1:
        return "over-damped"
    if zeta == 1:
        return "critically damped"
    if zeta < 1:
        return "under-damped"


# for simplicity just have it produce trajectories across a range of viscosities
# this is the only conditioned augmented dimension for the proof of concept

x0 = 1.0  # Initial position
v0 = 0.0  # Initial velocity
m = 1.0   # Mass
k = 1.0   # Spring constant
c = np.array([0.25, 2, 3.75]) # np.arange(0.25,5.25,0.25)   # Viscous force coefficient
t_max = 10
dt = 0.1
_plot_dir = os.path.dirname(__file__)

X = []
T = [] # same for all unnecessary
labels = []


for ci in c:
    xi,ti = damped_harmonic_oscillator(x0, v0, m, k, ci, t_max, dt=dt)
    labeli = get_damping_zone(m,k,ci)
    X.append(xi)
    T.append(ti)
    labels.append(labeli)


colors = ['red', 'green', 'blue']
for i, x in enumerate(X):
    labeli = labels[i]
    plt.plot(T[i], x, label=labeli, color=colors[i])

# for i, x in enumerate(X):
#     labeli = labels[i]
#     plt.plot(T[i], x, label=labeli)
plt.xlabel('Time')
plt.ylabel('Position')
plt.title(f'Harmonic Oscillator')
plt.legend()
plt.show()
save_path = os.path.join(_plot_dir, 'full_trajectory_plot.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()



red_x = X[0]
red_t = T[0]
green_x = X[1]
green_t = T[1]
blue_x = X[2]
blue_t = T[2]

_rng = np.random.default_rng(10)
plt.figure()
irregular_records = []
for traj_idx, (xi, ti, labeli, color) in enumerate(zip(X, T, labels, colors)):
    n = len(ti)
    k = min(99, n)
    idx = np.sort(_rng.choice(n, size=k, replace=False))
    plt.scatter(ti[idx], xi[idx], label=labeli, color=color, s=35, alpha=0.85)
    hadm_id = traj_idx
    for t_i, x_i in zip(ti[idx], xi[idx]):
        irregular_records.append({
            "HADM_ID": hadm_id,
            "x": float(x_i),
            "c": float(c[hadm_id]),
            "t": float(t_i / t_max),  # normalize time to [0, 1] like example.py
        })
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Irregular sampling: 10 random time indices per trajectory')
plt.legend()
plt.savefig(os.path.join(_plot_dir, 'irregular_data_plot.png'), dpi=150, bbox_inches='tight')
plt.show()


df_irregular = pd.DataFrame(irregular_records)
df_all = {"train": df_irregular, "val": df_irregular, "test": df_irregular}

file_path = os.path.join(_plot_dir, "3Oscillation_irregular_data.pkl")
with open(file_path, "wb") as f:
    pickle.dump(df_all, f)

# Reuse the same dataloader + model training flow from example.py
memory = 3
data_loader = eICUDataLoader(
    file_path=file_path,
    t_headings="t",
    x_headings=["x"],
    cond_headings=["c"],
    memory=memory,
    batch_size=64,
    groupby="HADM_ID",
    train_consecutive=False,
)

train_loader = data_loader.get_train_loader()
val_loader = data_loader.get_val_loader()
test_loader = data_loader.get_test_loader()

# example.py positional encoding builds CPU frequencies, so keep this run on CPU
device = "cpu"
model = MLP_Cond_Memory_Module(
    treatment_cond=1,
    memory=memory,
    dim=data_loader.output_dim,
    w=64,
    time_varying=True,
    conditional=True,
    lr=1e-3,
    sigma=0.05,
    implementation="ODE",
)

train_model_irregular(
    model,
    noise_prediction=False,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=1000,
    device=device,
    plot_dir=_plot_dir,
    val_every=100,
)
test_model_irregular(
    model,
    noise_prediction=False,
    test_loader=test_loader,
    device=device,
    save_path=os.path.join(_plot_dir, "validation_steplast.png"),
)