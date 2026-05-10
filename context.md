# Framework Context (Compact)

## Environment

- Python env: `conda activate ti-env`
- Main run command (from `src/`): `python main.py`

## What this framework does

- Trains a conditional trajectory model for irregular clinical time series (patient-wise sequences).
- Uses PyTorch Lightning + Hydra for config-driven training.
- Supports ODE (`torchdyn.NeuralODE`, `dopri5`) and SDE rollout variants.
- Learns next-state coordinates plus time-to-next-event, then integrates forward over each interval.

## Core execution flow

1. `src/main.py` loads Hydra config (`src/conf/config.yaml` + selected data/model yaml).
2. Instantiates `clinical_DataModule` and model module.
3. Runs `trainer.fit(...)`, selects best checkpoint via `val_loss`.
4. Runs `trainer.test(..., ckpt_path="best")`.

## Main files

### `src/main.py`

- Entry point and training orchestration.
- Seeds, instantiates datamodule/model, builds W&B config, sets callbacks (checkpoint + early stopping), builds Lightning `Trainer`.
- Important caveat from current code: `check_val_every_n_epoch=50` is hardcoded in trainer, which can override config intent (config has `10`).

### `src/data/datamodule.py`

- `clinical_DataModule` handles reading, filtering, pairing, and dataloaders.
- Loads train/val/test from pickle and includes NumPy 1.x/2.x compatibility fallback for pickle imports.
- Filters out short patient trajectories (`len <= min_timept`).
- Creates:
  - pairwise samples for training (`create_pairs`)
  - per-patient trajectories for val/test (`create_patient_data`, `create_patient_data_t0`)
- Supports memory augmentation by appending previous `x` values to condition features.

### `src/model/components/mlp.py`

- Defines core MLP backbones and conditional/time-encoded variants.
- `MLP_conditional_liver_pe_memory`:
  - positional encoding for time
  - memory-aware input construction
  - multiple output heads for distributional training:
    - Gaussian NLL
    - MDN
    - Quantile
    - L1+variance (or default point estimate)
- In ODE forward, converts predicted destination/time into velocity field `v_t`.

### `src/model/mlp_memory.py`

- Lightning module wrapper around memory-aware MLP model.
- `training_step` samples interpolated noisy points between `(x0, x1)` and learns endpoint + remaining time.
- `_compute_loss` switches by distributional mode (`gaussian_nll`, `mdn`, `quantile`, `l1_variance`, default point loss).
- `validation_step` / `test_step` call trajectory rollout and compute metrics.
- `test_trajectory_ode` performs per-interval ODE integration with `torch.linspace(..., ode_t_span_points)` and recursive rollout using previous prediction.

## Config anchors

- `src/conf/config.yaml`: global defaults (data/model selection, epochs, validation cadence, logging).
- `src/conf/data/eICU.yaml`: eICU datamodule setup (features, condition vars, file path, memory).

## Practical notes from prior debugging context

- Oversmoothing/sticky trajectories are mainly linked to training objective + train/inference mismatch (exposure bias), not only solver choice.
- `ode_t_span_points` controls the number of solver evaluation points per interval; increasing it raises compute cost.

