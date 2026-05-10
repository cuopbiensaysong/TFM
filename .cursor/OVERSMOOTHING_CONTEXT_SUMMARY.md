# Oversmoothing Investigation Context (TFM)

This file summarizes the oversmoothing investigation from the prior chat so a new chat can continue without re-discovering context.

## 1) Original problem report

- Severe oversmoothing in trajectory prediction plots from:
  - ODE/SDE models trained on `data/eicu_sepsis_eicu.pkl`
  - Overfit runs on `data/overfit_data.pkl`
- Symptom: predicted trajectories collapse toward smooth/low-variance behavior, failing to follow volatile GT dynamics.

## 2) Main diagnostic outcomes

### A. Overfit checkpoint (`outputs/2026-05-05/10-13-35/...best_model.ckpt`)

- This run used `data/overfit_data.pkl` and all splits were identical (train/val/test all 7 patients).
- Changing `ode_t_span_points` did not materially change smoothing (2..100 points gave near-identical metrics in diagnostics).
- Teacher-forced per-interval evaluation had substantially better variance/error than strict autoregressive rollout.
- Initial conclusion: major train/infer mismatch + autoregressive drift was a dominant contributor.

### B. Larger-data checkpoint (`outputs/2026-05-03/05-43-13/...best_model.ckpt`)

- Same pattern remained on broader test set:
  - Rollout variance was much lower than teacher-forced variance for most patients.
  - Rollout error was worse than teacher-forced error in most patients.
- Aggregate diagnostic over 60 test patients (computed in chat):
  - mean teacher_mse: ~0.484
  - mean rollout_mse: ~0.758
  - mean GT var: ~0.510
  - mean teacher var: ~0.454
  - mean rollout var: ~0.058
  - rollout worse MSE fraction: 0.85
  - rollout smoother-than-teacher fraction: 0.967

### C. User-proposed velocity check (confirmed)

- User requested exact teacher-forced step displacement check:
  - `||x0_values[i,:dim] - p_i||` where `p_i` is the predicted end-of-interval point.
- Result (60-patient aggregate in chat):
  - global mean: ~0.063
  - global median: ~0.024
  - fraction `< 0.10`: ~81%
- This confirmed a key point: even teacher-forced steps are often small/sticky under that criterion.

### D. Current assessment of immediate oversmoothing cause

- Current working assessment from the chat:
  - A major **proximal cause** of oversmoothing is that per-step prediction often stays too close to the current input state (`x0_values[i,:dim]`).
  - This produces very small step displacement; repeated in rollout, it collapses trajectory variability.
- Important distinction:
  - This explains the immediate behavior ("sticky dynamics"), while deeper root causes can still include objective design, optimization scaling, and autoregressive compounding.

## 3) Files/scripts added during investigation

### A. Trajectory comparison script

- Added: `src/draw_trajectory_compare.py`
- Purpose: plot GT vs rollout vs teacher-forced trajectories in one figure set.
- Outputs include:
  - `*_gt_rollout_teacher_per_dim.png`
  - `*_gt_rollout_teacher_3d.png` (for `dim=2`)

## 4) Most useful reproducibility commands used in chat

- Compare trajectories:
  - `python src/draw_trajectory_compare.py --config ... --ckpt ... --out-dir ... --split test --num-samples 3 --ode-t-span-points 10`
- Check overfit split identity:
  - pandas read of `data/overfit_data.pkl` and compare HADM_ID sets across train/val/test.
- Rollout vs teacher diagnostics:
  - computed by constructing per-patient trajectories under both modes and comparing MSE/variance.

