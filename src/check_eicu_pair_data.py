import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
from omegaconf import OmegaConf

from data.datamodule import clinical_DataModule
from utils.data_audit import compute_pair_variables, save_audit_outputs


def _load_datamodule_cfg(config_path: str) -> Dict:
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if "data_module" not in cfg:
        raise KeyError(f"'data_module' not found in config: {config_path}")
    dm_cfg = dict(cfg["data_module"])
    dm_cfg.pop("_target_", None)
    return dm_cfg


def _collect_pairs_from_splits(dm: clinical_DataModule) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pair_lists: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for split_name in ("train", "val", "test"):
        split_df = getattr(dm, split_name, None)
        if split_df is None:
            continue
        x0, _x0_class, x1, t0, t1 = dm.create_pairs(split_df)
        pair_lists.append((x0, x1, t0, t1))

    if not pair_lists:
        raise RuntimeError("No pairs found in train/val/test splits.")

    x0_all = np.concatenate([p[0] for p in pair_lists], axis=0)
    x1_all = np.concatenate([p[1] for p in pair_lists], axis=0)
    t0_all = np.concatenate([p[2] for p in pair_lists], axis=0)
    t1_all = np.concatenate([p[3] for p in pair_lists], axis=0)
    return x0_all, x1_all, t0_all, t1_all


def run_audit(config_path: str, out_dir: str) -> Dict[str, str]:
    dm_cfg = _load_datamodule_cfg(config_path)
    dm_cfg["train_consecutive"] = False  # match conditional-model setting in main.py
    dm = clinical_DataModule(**dm_cfg)
    dm.setup(stage=None)

    x0_all, x1_all, t0_all, t1_all = _collect_pairs_from_splits(dm)
    delta_time, delta_x, velocity, zero_dt_count = compute_pair_variables(x0_all, x1_all, t0_all, t1_all)

    dataset_name = dm.naming or os.path.splitext(os.path.basename(config_path))[0]
    result_paths = save_audit_outputs(
        dataset_name=dataset_name,
        out_dir=out_dir,
        delta_time=delta_time,
        delta_x=delta_x,
        velocity=velocity,
        zero_dt_count=zero_dt_count,
    )
    return result_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit pair-level deltas/velocity for eICU datasets.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "src/conf/data/eICU.yaml",
            "src/conf/data/eICU_multdim.yaml",
        ],
        help="YAML config paths containing a data_module section.",
    )
    parser.add_argument(
        "--out-dir",
        default="/home/nvidia-lab/ai4life/thaind2/time_series/TFM/data",
        help="Directory where logs and figures are saved.",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for cfg_path in args.configs:
        abs_cfg_path = cfg_path if os.path.isabs(cfg_path) else os.path.abspath(cfg_path)
        print(f"\n=== Auditing config: {abs_cfg_path} ===")
        outputs = run_audit(abs_cfg_path, args.out_dir)
        for key, path in outputs.items():
            print(f"{key}: {path}")


if __name__ == "__main__":
    sys.exit(main())
