# Clear context: training flow and oversmoothing

This note summarizes how training is wired in this framework (without tying the narrative to implementation-specific names) and why predictions tend to **stick near the current input**—the core “oversmoothing” symptom seen in trajectories.

---

## 1. Pseudocode: how training runs (conditional consecutive-interval setup)

batched training pairs, and an **ODE** rollout at test time.

### Notation (used below)

| Symbol | Meaning |
|--------|---------|
| **k** | Index along one patient’s time-ordered visits (integer step on the discrete grid). |
| **x_k** | Feature vector at visit **k** (the “anchor” observation for one pair). |
| **x_next** | Feature vector at the **next** visit after k (visit k+1); this is the supervised target for that segment. |
| **T_k** | Time stamp associated with **x_k** (start of the segment in clock time). |
| **T_next** | Time stamp associated with **x_next** (end of the segment in clock time). **Not** “end” in an abstract sense—always “the next measured time” after **T_k**. |
| **Δt** | Segment length: **Δt = T_next − T_k** (scalar, positive). |
| **u** | Random interpolation coefficient in **(0, 1)** along the segment from **x_k** toward **x_next**. |
| **t(u)** | Clock time of the synthetic training point: **t(u) = T_k + u · Δt** (lies strictly between **T_k** and **T_next** when **u ∈ (0,1)**). |
| **τ_rem** | **Remaining** clock time until **T_next** from the current synthetic point: **τ_rem = T_next − t(u)**. |

### 1.1 Data preparation (high level)

```
FOR each patient trajectory (sorted by time):
    FOR each valid index k along the trajectory:
        x_k           ← features at time T_k          # anchor / “current” visit
        x_next        ← features at time T_next      # following visit; supervised label for this pair
        context_k     ← static or slow-moving conditioning (e.g. treatment / admission info)
        optionally append a flat window of past feature vectors (memory): … , x_{k−2}, x_{k−1}
        record (T_k, T_next) for this pair
    emit many training pairs (one segment per valid k) from that patient
```

Training batches are built from these pairs (shuffled, multi-example batches). Validation and test use **one patient per batch** and full trajectories for rollout.

### 1.2 One training step (core algorithm)

For each pair anchored at **k**:

1. **Interpolation along the segment [T_k, T_next].** Draw **u ~ Uniform(0, 1)**. Build a **mixed state** **x̃(u) = (1 − u)·x_k + u·x_next**, then add **Gaussian noise** scaled by a hyperparameter. Intuition: the network sees states **between** **x_k** and **x_next**, not only those endpoints.

2. **Clock inputs.** From **T_k**, **T_next**, and **u**, compute:
   - **t(u) = T_k + u·(T_next − T_k)** — absolute time at the synthetic point, and  
   - **τ_rem = T_next − t(u)** — remaining duration until **T_next**.

3. **Network inputs.** Concatenate:
   - the mixed noisy state (feature dimensions only),
   - the conditioning vector (and memory strip if enabled),
   - information derived from **t(u)** (often positional encoding of time inside the network).

4. **Network outputs (train mode).** The network predicts:
   - an estimate of **x_next** (same dimensionality as trajectory features), and  
   - an estimate of **τ_rem** (remaining time until **T_next**).

5. **Loss (default point-wise regression).** Typically **mean squared error** (or similar) on:
   - predicted **x_next** vs **true x_next**, plus  
   - predicted **τ_rem** vs **true τ_rem** (from the same **u** that built the synthetic point).

Gradients update weights so the model minimizes this objective **averaged over batches**.

### 1.3 Inference / rollout (why this ties to “velocity”)

At test time, the same network is used inside a **neural ODE** wrapper. The ODE right-hand side is **not** “train a velocity directly”; it is **derived** from the trained “predict **x_next** for this segment” head:

- At a state **z** along the interval **[T_k, T_next]**, the model outputs a predicted **segment endpoint** in feature space (call it **x̂_next**) and a predicted **remaining time** **τ̂_rem**.
- The **instantaneous change rate** in feature space is set proportional to  
  **(x̂_next − current feature part of z) / τ̂_rem**  
  (with a small numerical floor on the denominator in code).

The ODE solver **integrates** this field from clock time **T_k** to clock time **T_next**. The codebase uses an adaptive explicit solver (**dopri5**) with tolerances; changing the number of time samples along the segment did **not** materially fix smoothing in diagnostics—the field being integrated matters more than solver granularity.

### 1.4 Outer loop over the trajectory (autoregressive effect)

For each consecutive segment **[T_k, T_next]** on a test patient:

1. Start from the **predicted** state at **T_next** of the previous segment (after the first step, which may anchor on ground-truth **x_k**).
2. Update any **memory buffer** by shifting in the new predicted coordinates.
3. Solve the ODE for the next segment (next **k**) and append the prediction.

So errors and “stickiness” **compound** along the sequence: small displacements per segment accumulate into **collapsed variance** over long horizons.

---

## 2. Root of oversmoothing (why behavior stays close to the current input)

### 2.1 Primary: regression loss favors staying near the current state

The training target is **predict x_next** under squared loss (or variants). For clinical series where successive measurements are often **highly correlated**, the minimum-risk predictor under mean squared error is close to the **conditional expectation** of the next state given context—often **near the current measured state**. The model therefore learns **predictions of x_next that hug the current interpolated position**, especially when inputs lie on or near the interpolation path.

That is a **training-objective and data-statistics** effect, not an artifact of the integrator.

### 2.2 Amplifier: velocity built from “predicted endpoint minus current point”

Inference constructs dynamics so that **motion in feature space scales with how far the predicted next-state estimate is from where you are now** (that estimate targets **x_next**). If that estimate stays near the current state, the **implied velocity is small** regardless of which ODE integrator is used. The solver then faithfully produces **small displacements**—dopri5 is integrating a **near-zero** field, not “over-smoothing” on its own.

### 2.3 Compounding: autoregressive rollout

Teacher-forced evaluation (feeding ground-truth segment anchors) typically looks better than strict rollout because rollout **re-injects the model’s own small movements** as the next segment’s start. Repeated **small steps** yield trajectories with **much lower variance** than ground truth—matching observed diagnostics (rollout variance collapse vs teacher-forced).

### 2.4 Secondary: train vs test state distribution

Training mixes states **along noisy interpolants** between **x_k** and **x_next**. Testing repeatedly hits **states produced by the model** after integration. That mismatch can worsen stability but is secondary to **mean-regression under MSE** and the **velocity construction** above.

### 2.5 What is *not* the main cause

- **The dopri5 solver** is not the root cause: it approximates the continuous dynamics defined by the network. With the same learned field, another solver would exhibit the same stickiness.
- **Solver resolution** (number of time points in the span) was checked and did not fundamentally fix the smooth collapse—the bottleneck is the **learned dynamics**, not integration precision alone.

---

## 3. One-line summary

**Oversmoothing here is driven mainly by (i) regression-to-the-mean under the training loss on correlated sequential data, (ii) expressing dynamics as displacement toward the predicted next observation (`x_next`) divided by predicted remaining time—so “sticky” endpoint estimates imply tiny motion—and (iii) autoregressive error compounding; the ODE solver is largely a faithful integrator of that field.**

For prior numerical context (eICU / overfit runs, teacher vs rollout, displacement magnitudes), see `OVERSMOOTHING_CONTEXT_SUMMARY.md` in this folder.
