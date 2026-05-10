# Displacement Prediction + Distributional Loss: Implementation Plan

## Overview

This plan combines two complementary strategies:
1. **Displacement prediction** — network predicts Δ = (x1 - x0) instead of raw x1
2. **Distributional loss** — network outputs distribution parameters, not a point estimate

Together they address:
- Mean regression (distributional loss allows bold predictions)
- Architectural bias (displacement centers output around 0, focuses on learning changes)
- Inference sampling (draw from predicted distribution to inject variance)

---

## 1) Architecture Changes

### Current Network Output

```
forward_train(input) → [predicted_x1 (dim), conditioning_passthrough, predicted_time_remaining (1)]
```

### New Network Output (Displacement + Distributional)

```
forward_train(input) → [predicted_Δ_mu (dim), predicted_Δ_log_var (dim), conditioning_passthrough, predicted_time_remaining (1)]
```

Where:
- `predicted_Δ_mu`: mean of the displacement distribution (dim values)
- `predicted_Δ_log_var`: log-variance of displacement distribution (dim values)
- At inference: `predicted_x1 = x_current + sample(N(Δ_mu, exp(Δ_log_var)))`

### Network Architecture Modification

```python
class MLP_conditional_liver_pe_memory_displacement(torch.nn.Module):
    def __init__(self, dim, treatment_cond, memory, w=64, time_varying=False,
                 conditional=False, time_dim=NUM_FREQS*2, clip=None):
        super().__init__()
        self.dim = dim
        self.out_dim = dim * 2 + 1  # mu(dim) + log_var(dim) + time_remaining(1)
        self.treatment_cond = treatment_cond
        self.memory = memory
        self.clip = clip

        self.indim = dim + (time_dim if time_varying else 0) + \
                     (treatment_cond if conditional else 0) + (dim * memory)

        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.indim, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, w),
            torch.nn.SELU(),
            torch.nn.Linear(w, self.out_dim),
        )

    def forward_train(self, x):
        time_tensor = x[:, -1]
        encoded_time = positional_encoding_tensor(time_tensor).reshape(-1, NUM_FREQS * 2)
        new_x = torch.cat([x[:, :-1], encoded_time], dim=1)
        result = self.net(new_x)

        delta_mu = result[:, :self.dim]
        delta_log_var = result[:, self.dim:2*self.dim]
        time_remaining = result[:, -1:]

        # Clamp log_var for stability
        delta_log_var = torch.clamp(delta_log_var, min=-6.0, max=4.0)

        return delta_mu, delta_log_var, time_remaining

    def forward(self, x):
        """ODE-compatible forward: returns velocity field."""
        delta_mu, delta_log_var, time_remaining = self.forward_train(x)

        # Sample displacement
        std = torch.exp(0.5 * delta_log_var)
        eps = torch.randn_like(std)
        delta_sample = delta_mu + std * eps  # reparameterization trick

        # Velocity = displacement / time_remaining
        x_coord = x[:, :self.dim]
        if self.clip is None:
            vt = delta_sample / time_remaining
        else:
            vt = delta_sample / torch.clip(time_remaining, min=self.clip)

        # Zero velocity for conditioning dimensions
        final_vt = torch.cat([vt, torch.zeros_like(x[:, self.dim:-1])], dim=1)
        return final_vt
```

---

## 2) Training Loss

### Gaussian NLL on Displacement

```python
def displacement_gaussian_nll(delta_mu, delta_log_var, true_displacement):
    """
    Negative log-likelihood of true displacement under predicted Gaussian.

    Args:
        delta_mu: [B, dim] predicted mean of displacement
        delta_log_var: [B, dim] predicted log-variance of displacement
        true_displacement: [B, dim] actual (x1 - x0)

    Returns:
        scalar loss
    """
    # NLL = 0.5 * (log_var + (x - mu)^2 / var)
    var = torch.exp(delta_log_var)
    nll = 0.5 * (delta_log_var + (true_displacement - delta_mu) ** 2 / var)
    return nll.mean()
```

### Modified Training Step

```python
def training_step(self, batch, batch_idx):
    x0, x0_class, x1, x0_time, x1_time = batch
    # ... standard preprocessing ...

    # Compute true displacement
    true_delta = x1[:, :self.dim] - x0[:, :self.dim]

    # Interpolation (same as before but for displacement framing)
    t = torch.rand(B, 1).to(device)
    data_t_diff = (x1_time - x0_time).unsqueeze(1)
    t_model = t * data_t_diff + x0_time.unsqueeze(1)
    futuretime = x1_time.unsqueeze(1) - t_model

    # Interpolated state (still needed for the flow matching framework)
    x_t = x0 * (1 - t) + x1 * t + sigma * noise

    in_tensor = torch.cat([x_t, x0_class, t_model], dim=-1)
    delta_mu, delta_log_var, pred_time = self.model.forward_train(in_tensor)

    # The displacement target depends on where we are on the path:
    # At time t along interpolation, the "remaining displacement" is (x1 - x_t)
    # But we reframe: the network predicts total displacement from x0 to x1
    # regardless of current interpolation position (simpler, more stable)
    loss_coord = displacement_gaussian_nll(delta_mu, delta_log_var, true_delta)
    loss_time = mse_loss(pred_time.squeeze(), futuretime.squeeze())

    loss = loss_coord + loss_time
    return loss
```

---

## 3) Inference Changes

### ODE Inference with Sampling

```python
def test_trajectory_ode(self, pt_tensor):
    node = NeuralODE(
        torch_wrapper_tv(self.model), solver="dopri5", ...
    )
    # The forward() method now samples from the displacement distribution
    # Each ODE solve produces a STOCHASTIC trajectory even with ODE
    # (stochasticity comes from sampling the displacement, not from the solver)
    ...
```

### Deterministic vs Stochastic Mode

At inference, you can choose:
- **Stochastic** (default): sample from N(mu, var) → diverse trajectories
- **Deterministic**: use mu directly → reproducible but potentially still conservative

```python
def forward(self, x, deterministic=False):
    delta_mu, delta_log_var, time_remaining = self.forward_train(x)

    if deterministic:
        delta = delta_mu
    else:
        std = torch.exp(0.5 * delta_log_var)
        delta = delta_mu + std * torch.randn_like(std)

    vt = delta / torch.clip(time_remaining, min=self.clip)
    final_vt = torch.cat([vt, torch.zeros_like(x[:, self.dim:-1])], dim=1)
    return final_vt
```

---

## 4) Important Design Decisions

### A. What is the displacement target?

Two options:

**Option A (recommended): Total displacement from x0 to x1**
- Target: `Δ = x1 - x0` (regardless of current interpolation position t)
- Simpler, more stable
- The velocity is then `Δ / time_remaining_to_t1`

**Option B: Remaining displacement from x_t to x1**
- Target: `Δ_remaining = x1 - x_t` (depends on t)
- More aligned with flow matching theory
- But scale varies with t (large when t≈0, small when t≈1)

Recommendation: Start with Option A for stability.

### B. Variance initialization

Initialize the last layer bias for log_var outputs to a negative value (e.g., -2.0). This starts the model with small variance (conservative) and lets it learn to increase variance where needed.

```python
# After creating self.net:
with torch.no_grad():
    self.net[-1].bias[dim:2*dim].fill_(-2.0)  # init log_var ≈ small
```

### C. KL regularization (optional)

Add a KL divergence term to prevent variance from collapsing to 0 or exploding:

```python
# KL(predicted || N(0, 1))
kl = 0.5 * (delta_mu**2 + delta_log_var.exp() - delta_log_var - 1).mean()
loss = nll + beta * kl  # beta = 0.001 to 0.01
```

This acts as a soft constraint keeping predictions "reasonable" but is optional.

### D. Temperature scaling at inference

At inference, you can scale the predicted variance:

```python
# temperature > 1.0: more diverse trajectories
# temperature < 1.0: more conservative
# temperature = 1.0: use as-is
delta = delta_mu + temperature * std * torch.randn_like(std)
```

This gives you a knob to control trajectory diversity without retraining.

---

## 5) Files to Create/Modify

| File | Action | Description |
|------|--------|-------------|
| `src/model/components/mlp_displacement.py` | **Create** | New network class with displacement + distributional output |
| `src/utils/loss.py` | **Modify** | Add `displacement_gaussian_nll`, `displacement_mdn_loss`, etc. |
| `src/model/mlp_memory_displacement.py` | **Create** | Lightning module using displacement prediction |
| `src/conf/model/tfm_displacement.yaml` | **Create** | Config for displacement model |
| `src/model/components/grad_util.py` | **Modify** | Add wrapper that handles stochastic forward |

---

## 6) Hyperparameters

| Parameter | Recommended | Notes |
|-----------|------------|-------|
| `log_var_min` | -6.0 | Clamp floor for log-variance (prevents near-zero variance) |
| `log_var_max` | 4.0 | Clamp ceiling (prevents explosion) |
| `log_var_init` | -2.0 | Initial bias for log-var output neurons |
| `temperature` | 1.0 | Inference sampling temperature |
| `kl_beta` | 0.001 | KL regularization weight (0 to disable) |
| `clip` | 1e-2 | Min value for time_remaining denominator |

---

## 7) Expected Benefits vs Current Model

| Aspect | Current Model | Displacement + Distributional |
|--------|--------------|------------------------------|
| Output meaning | absolute x1 position | displacement distribution N(Δ_mu, Δ_var) |
| MSE-optimal behavior | predicts E[x1] ≈ x0 | predicts E[Δ] but with learned uncertainty |
| Inference | deterministic velocity ≈ 0 | sampled velocity with variance |
| Rollout variance | very low (0.058) | expected higher (tunable via temperature) |
| Architectural efficiency | must learn identity + change | only learns change |

---

## 8) Migration Path (Incremental)

1. **Phase 1**: Keep existing architecture, just change loss from MSE to Gaussian NLL (double output dim). No displacement yet.
2. **Phase 2**: Switch to displacement target. Compare with Phase 1.
3. **Phase 3**: Add temperature tuning, KL regularization, scheduled sampling.

This lets you A/B test each change independently.
