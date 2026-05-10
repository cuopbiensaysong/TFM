# Oversmoothing Investigation Context (Backup)

This is a clean backup context file for handoff to another chat.

## 1) Original problem report

- Severe oversmoothing in trajectory prediction plots from ODE/SDE models.
- Observed on:
  - full-data runs (`data/eicu_sepsis_eicu.pkl`)
  - overfit runs (`data/overfit_data.pkl`)
- Symptom: predicted trajectories collapse toward smooth/low-variance dynamics and miss sharp fluctuations.

## 2) Main diagnostic outcomes

### A. Overfit checkpoint (`outputs/2026-05-05/10-13-35/...best_model.ckpt`)

- Overfit dataset splits were identical (train/val/test all same 7 patients).
- Changing `ode_t_span_points` did not materially change smoothing in diagnostics.
- Teacher-forced per-interval predictions were better than strict autoregressive rollout.
- Initial interpretation: autoregressive compounding + train/infer mismatch contributes strongly.

### B. Larger-data checkpoint (`outputs/2026-05-03/05-43-13/...best_model.ckpt`)

- Same qualitative pattern on larger test set:
  - rollout variance << teacher-forced variance
  - rollout error > teacher-forced error for most patients
- Aggregate diagnostic over 60 test patients (from chat):
  - mean teacher_mse: ~0.484
  - mean rollout_mse: ~0.758
  - mean GT variance: ~0.510
  - mean teacher variance: ~0.454
  - mean rollout variance: ~0.058
  - rollout worse MSE fraction: 0.85
  - rollout smoother-than-teacher fraction: 0.967

### C. User-requested displacement check (teacher-forced)

- Exact metric requested:
  - `||x0_values[i,:dim] - p_i||` in teacher-forced loop
- 60-patient aggregate (from chat):
  - global mean: ~0.063
  - global median: ~0.024
  - fraction `< 0.10`: ~81%
- Confirms many predicted endpoints remain close to current input state.

### D. Current assessment of immediate cause

- Working assessment:
  - A major proximal cause of oversmoothing is that per-step predictions are often too close to current state (`x0_values[i,:dim]`).
  - Repeated small steps in rollout produce low-variance/smoothed trajectories.
- Distinction:
  - This explains immediate behavior, while deeper causes can include objective design, scaling, and autoregressive error accumulation.

## 3) Notebook/main pipeline comparison note

- Detailed notebook-vs-main logic differences were split out to:
  - `notebook/NOTEBOOK_PIPELINE_LOGIC_DIFFERENCES.md`

## 4) Files/scripts added in this investigation

### A. Plot comparison script

- `src/draw_trajectory_compare.py`
- Purpose: visualize GT vs rollout vs teacher-forced side-by-side.
- Outputs:
  - `*_gt_rollout_teacher_per_dim.png`
  - `*_gt_rollout_teacher_3d.png` (for `dim=2`)

### B. Fix-model files introduced

- `src/model/mlp_memory_fix.py`
- `src/model/mlp_noise_fix.py`
- `src/model/components/mlp4velocity.py`

Notes:
- Earlier fix versions were criticized for:
  1. objective mismatch vs original semantics
  2. redundant forwards
  3. unstable velocity scaling when dividing by tiny normalized `dt`
- Later refactor moved toward direct velocity/timegap objective design.

## 5) Important caveats for continuation

- Current refactored fix path changed training objective semantics (next-point -> velocity/timegap).
- Rollout sampling in main model codepaths still commonly uses dataset interval bounds (`times_x0 -> times_x1`) at test time.
- Any continuation should validate:
  - training stability of velocity/timegap objectives
  - relative magnitude of velocity/timegap/noise losses
  - whether predicted timegap should also drive rollout interval construction
  - A/B comparison against original models under same seed/split.

## 6) Useful reproducibility tasks used in chat

- Compare GT/rollout/teacher plots with:
  - `python src/draw_trajectory_compare.py --config ... --ckpt ... --out-dir ... --split test --num-samples 3 --ode-t-span-points 10`
- Confirm overfit split identity:
  - inspect `data/overfit_data.pkl` and compare HADM_ID sets across splits.
- Quantify oversmoothing:
  - rollout vs teacher MSE/variance
  - teacher-forced displacement `||x0 - p||` statistics.

## 7) Recommended next step

- Run a controlled A/B:
  1. baseline original (`mlp_memory.py` or `mlp_noise.py`)
  2. refactored velocity fix versions
- Evaluate same seed/split with:
  - rollout MSE
  - rollout variance
  - teacher-forced variance
  - teacher-forced displacement stats
- Decide whether to keep velocity-only objective, hybrid objective, or rollback.

