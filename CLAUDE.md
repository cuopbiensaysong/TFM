# Framework Context (Current Branch)

## Environment
- Python env: `conda activate ti-env`
- Typical run (from `src/`): `python main.py`

## Purpose
- Clinical irregular time-series trajectory forecasting with conditional neural dynamics.
- Learns next state `x1` and remaining time to next event from interpolated/noisy intermediate states.
- Performs rollout per interval with either ODE (`dopri5`) or SDE solver.

## Pipeline (high level)
1. `src/main.py` (Hydra + Lightning) instantiates datamodule/model from YAML.
2. `trainer.fit(...)` trains and monitors `val_loss` (checkpoint + early stopping).
3. `trainer.test(..., ckpt_path="best")` evaluates best checkpoint.

## Key files

### `src/model/mlp_memory.py`
- Main Lightning module: `MLP_Cond_Memory_Module`.
- Uses `MLP_conditional_liver_pe_memory` backbone from `mlp.py`.
- `training_step`:
  - samples random interpolation time `t`
  - builds noisy state `x_t = (1-t)x0 + tx1 + noise`
  - predicts next coordinates + time-to-next-event
  - optimizes coordinate loss + time loss (`mse_loss`/`l1_loss` configurable via `loss_fn`)
- Supports `implementation="ODE"` or `"SDE"`:
  - ODE: rollout via `NeuralODE(..., solver="dopri5", sensitivity="adjoint")`
  - SDE: Euler-Maruyama-like custom loop in `_sde_solver`
- Memory-aware autoregressive rollout:
  - keeps history in conditioning vector
  - updates history with each predicted point
- `ode_t_span_points` controls integration grid size per interval (`>=2`).

### `src/model/components/mlp.py`
- Defines base `MLP`, `MLP_conditional_liver_pe`, and memory variant `MLP_conditional_liver_pe_memory`.
- `MLP_conditional_liver_pe_memory`:
  - input = current coords + conditioning + memory + positional time encoding
  - `forward_train` predicts `x1` coords and remaining time
  - `forward` converts prediction to velocity field:
    - `v_t = (x1_pred - x_t) / time_remaining` (with optional `clip` min denominator)
    - appends zeros for non-coordinate conditional channels
- This velocity field is what ODE/SDE solvers integrate.

## Related framework components
- `src/data/datamodule.py`: creates pair data for training and patient trajectory batches for val/test; supports memory-augmented conditioning.
- `src/conf/config.yaml`: global run config.
- `src/conf/data/eICU.yaml`: eICU dataset + feature/condition headings + memory setting.

## Current branch caveats relevant to oversmoothing
- Training target is endpoint regression (`x_t -> x1`), which can bias toward conservative/mean predictions in low-displacement regimes.
- Inference is recursive rollout (model conditions on its own previous predictions), creating train/infer mismatch risk (exposure bias).
- Larger `ode_t_span_points` increases compute; it does not by itself solve oversmoothing.

