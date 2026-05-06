# Critical logic differences vs notebook pipeline

This file extracts the "critical logic differences" section from oversmoothing investigation notes for quick reuse in other chats.

## 1) Main difference previously observed

- In earlier `notebook/example.py` ODE test logic, rollout context handling could leak teacher-forced context behavior by directly using per-step `x0_classes[i]` instead of fully recursive context update.

## 2) Rollout context currently in notebook after revision

- The notebook ODE rollout now uses complete rollout context:
  - `new_x_classes` is built from exogenous class channels + recursively updated `time_history`.
  - `testpt` uses:
    - `i == 0`: `[x0_values[i], new_x_classes]`
    - `i > 0`: `[pred_traj, new_x_classes]`

Reference:

```869:874:notebook/example.py
with torch.no_grad():
    if i == 0:
        testpt = torch.cat([x0_values[i].unsqueeze(0), new_x_classes], dim=1)
    else:
        testpt = torch.cat([pred_traj, new_x_classes], dim=1)
```

## 3) Main-pipeline counterpart

- Main model rollout in `src/model/mlp_memory.py` also uses recursive memory context (`new_x_classes`) and dataset interval bounds (`times_x0[i] -> times_x1[i]`) for each interval.

Reference:

```243:248:src/model/mlp_memory.py
for i in range(len_path): 
    time_span = self.__convert_tensor__(
        torch.linspace(times_x0[i], times_x1[i], self.ode_t_span_points)
    ).to(x0_values.device)
```

## 4) Why this matters for oversmoothing interpretation

- If notebook rollout injects true per-step context, trajectories can look less oversmoothed than strict autoregressive rollout.
- Aligning notebook and main rollout logic is necessary before comparing oversmoothing severity across scripts.
