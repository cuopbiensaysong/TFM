
import pandas as pd
import numpy as np
import random
import os
import shutil
import torch

def log_print(message, log_dir=None):
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "log.txt"), "a") as f:
            f.write(message + "\n")
    print(message)


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def _unpack_eval_batch(batch):
    x0_values, x0_classes, x1_values, times_x0, times_x1 = batch
    return (
        x0_values.squeeze(0),
        x0_classes.squeeze(0),
        x1_values.squeeze(0),
        times_x0.squeeze(),
        times_x1.squeeze(),
    )

def _build_condition_with_memory(x0_classes, idx, model, time_history):
    if model.memory > 0:
        static_cond = x0_classes[idx][:-(model.memory * model.dim)].unsqueeze(0)
        return torch.cat([static_cond, time_history.unsqueeze(0)], dim=1)
    cond_row = x0_classes[idx]
    return cond_row.unsqueeze(0) if cond_row.dim() == 1 else cond_row


def save_figure(fig, log_dir, filename, dpi=150):
    os.makedirs(log_dir, exist_ok=True)
    output_path = os.path.join(log_dir, filename)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return output_path


def save_script_copy(script_path, log_dir, filename=None):
    os.makedirs(log_dir, exist_ok=True)
    target_name = filename if filename is not None else os.path.basename(script_path)
    output_path = os.path.join(log_dir, target_name)
    shutil.copy2(script_path, output_path)
    return output_path
