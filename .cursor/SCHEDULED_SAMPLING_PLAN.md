# Scheduled Sampling / Exposure Bias Correction for TFM

## 1) Problem Statement

During training, the model always receives **ground-truth** intermediate states as input:
- `x_t = (1-t)*x0_true + t*x1_true + noise`
- The model never sees its own imperfect predictions during training.

During inference (autoregressive rollout):
- Step i's prediction becomes step i+1's input.
- If the model makes even small errors (e.g., predicts x1 too close to x0), that error becomes the starting point for the next step.
- The model was never trained on such "drifted" inputs → it doesn't know how to recover.

**Scheduled sampling** closes this gap by gradually replacing ground-truth inputs with model-generated inputs during training.

---

## 2) Adaptation to the Flow Matching Framework

### Current Training Loop (single pair)

```python
# In training_step():
x0, x0_class, x1, x0_time, x1_time = batch  # ground-truth pair
t = rand(0, 1)
x_t = (1-t)*x0 + t*x1 + sigma*noise  # interpolated input
in_tensor = [x_t, x0_class, t_model]
pred = model.forward_train(in_tensor)
loss = MSE(pred[:,:dim], x1) + MSE(pred[:,-1], futuretime)
```

### Scheduled Sampling Version

The key change: with probability `p_sample` (that increases during training), replace the starting point `x0` in a training pair with the model's own prediction from a "previous step."

Since the datamodule already creates consecutive pairs per patient, we can exploit the sequential structure.

---

## 3) Implementation Plan

### Step A: Modify DataModule to Return Sequential Triplets

Instead of independent (x0, x1) pairs, the training dataloader should optionally return **sequences** of K consecutive observations per patient:

```
(x_0, x_1, x_2, ..., x_{K-1}), (cond_0, cond_1, ...), (t_0, t_1, ..., t_{K-1})
```

This allows the training loop to process multi-step sequences where earlier predictions feed later steps.

**Concrete change in `datamodule.py`:**

```python
def create_sequences(self, df, seq_len=4):
    """Create overlapping sequences of length seq_len for scheduled sampling."""
    sequences = []
    for _, group in df.groupby('HADM_ID'):
        sorted_group = group.sort_values(by=self.t_headings)
        n = len(sorted_group)
        for start_idx in range(self.memory, n - seq_len):
            seq_x = sorted_group.iloc[start_idx:start_idx+seq_len][self.x_headings].values
            seq_cond = []
            for j in range(seq_len):
                idx = start_idx + j
                cond_j = sorted_group.iloc[idx][self.cond_headings].values
                if self.memory > 0:
                    mem = sorted_group.iloc[idx-self.memory:idx][self.x_headings].values.flatten()
                    cond_j = np.append(cond_j, mem)
                seq_cond.append(cond_j)
            seq_t = sorted_group.iloc[start_idx:start_idx+seq_len][self.t_headings].values
            sequences.append((
                seq_x.astype(np.float32),
                np.array(seq_cond).astype(np.float32),
                seq_t.astype(np.float32),
            ))
    return sequences
```

### Step B: Add a Sampling Schedule

Define a schedule that controls `p_sample` (probability of using model's own prediction):

```python
class SamplingSchedule:
    """Controls the probability of using model predictions vs ground truth."""

    def __init__(self, strategy="linear", warmup_epochs=50, final_p=0.5):
        self.strategy = strategy
        self.warmup_epochs = warmup_epochs
        self.final_p = final_p

    def get_p(self, current_epoch):
        if self.strategy == "linear":
            return min(self.final_p, self.final_p * current_epoch / self.warmup_epochs)
        elif self.strategy == "inverse_sigmoid":
            k = 5.0
            x = k * (2.0 * current_epoch / self.warmup_epochs - 1.0)
            return self.final_p / (1.0 + math.exp(-x))
        elif self.strategy == "constant":
            return self.final_p
        else:
            return 0.0
```

### Step C: Modified Training Step

```python
def training_step_scheduled(self, batch, batch_idx):
    """
    batch: (seq_x, seq_cond, seq_t) where each is [batch, seq_len, ...]
    """
    seq_x, seq_cond, seq_t = batch
    # seq_x: [B, K, dim]   (K consecutive observations)
    # seq_cond: [B, K, cond_dim]
    # seq_t: [B, K]

    B, K, dim = seq_x.shape
    p_sample = self.sampling_schedule.get_p(self.current_epoch)

    total_loss = 0.0
    prev_pred = None  # model's prediction from previous step

    for step in range(K - 1):
        x0_true = seq_x[:, step, :]          # [B, dim]
        x1_true = seq_x[:, step + 1, :]      # [B, dim]
        cond = seq_cond[:, step, :]           # [B, cond_dim]
        t0 = seq_t[:, step]                   # [B]
        t1 = seq_t[:, step + 1]              # [B]

        # --- Scheduled Sampling Decision ---
        if step == 0 or prev_pred is None:
            x0_input = x0_true
        else:
            # With probability p_sample, use model's own prediction
            use_model = (torch.rand(B, 1, device=x0_true.device) < p_sample).float()
            x0_input = use_model * prev_pred.detach() + (1 - use_model) * x0_true

        # --- Standard flow matching computation ---
        t = torch.rand(B, 1, device=x0_input.device)
        mu_t = x0_input * (1 - t) + x1_true * t
        x_noisy = mu_t + self.sigma * torch.randn(B, self.dim, device=x0_input.device)

        data_t_diff = (t1 - t0).unsqueeze(1)
        t_model = t * data_t_diff + t0.unsqueeze(1)
        futuretime = t1.unsqueeze(1) - t_model

        # --- Forward pass ---
        in_tensor = torch.cat([x_noisy, cond, t_model], dim=-1)
        xt = self.model.forward_train(in_tensor)

        # --- Loss ---
        step_loss = self.loss_fn(xt[:, :self.dim], x1_true)
        step_loss += self.loss_fn(xt[:, -1:], futuretime)
        total_loss += step_loss

        # --- Store prediction for next step's scheduled sampling ---
        # Do a "full inference" of this step to get predicted x1
        with torch.no_grad():
            # Use x0_input at t=t0 to predict x1
            t0_input = torch.cat([x0_input, cond, t0.unsqueeze(1)], dim=-1)
            pred_out = self.model.forward_train(t0_input)
            prev_pred = pred_out[:, :self.dim]

    avg_loss = total_loss / (K - 1)
    self.log('train_loss', avg_loss)
    return avg_loss
```

### Step D: Memory/History Update During Scheduled Sampling

The memory slots (past coordinates) also need updating when using model predictions:

```python
# Inside the step loop, after computing prev_pred:
if self.memory > 0:
    # Update the memory portion of conditioning
    # Shift memory window: drop oldest, append prev_pred
    mem_portion = cond[:, -(self.memory * self.dim):]  # [B, memory*dim]
    updated_mem = torch.cat([
        mem_portion[:, self.dim:],  # drop oldest
        prev_pred.detach()          # append latest prediction
    ], dim=1)
    # Replace memory in conditioning for next step
    seq_cond[:, step + 1, -(self.memory * self.dim):] = updated_mem
```

---

## 4) Integration with Existing Codebase

### Files to Modify

| File | Change |
|------|--------|
| `src/data/datamodule.py` | Add `create_sequences()` method; add `SequenceDataset` class; modify `train_dataloader()` to optionally return sequences |
| `src/model/mlp_memory.py` | Add `training_step_scheduled()` method; add `SamplingSchedule` as an attribute; modify `__init__` to accept schedule params |
| `src/conf/model/tfm_ode.yaml` | Add config keys: `scheduled_sampling: true`, `ss_warmup_epochs: 50`, `ss_final_p: 0.5`, `ss_seq_len: 4` |

### Sequence Dataset Class

```python
class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_x, seq_cond, seq_t = self.sequences[idx]
        return seq_x, seq_cond, seq_t
```

### DataModule Changes

```python
def train_dataloader(self, shuffle=True):
    if self.scheduled_sampling:
        sequences = self.create_sequences(self.train, seq_len=self.ss_seq_len)
        dataset = SequenceDataset(sequences)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=1)
    else:
        # existing logic
        ...
```

---

## 5) Hyperparameter Recommendations

| Parameter | Recommended Start | Notes |
|-----------|------------------|-------|
| `ss_seq_len` | 4 | Number of consecutive steps per training sequence. Longer = more exposure but slower training |
| `ss_warmup_epochs` | 50-100 | Epochs before reaching `ss_final_p`. Too fast → training instability early on |
| `ss_final_p` | 0.3-0.5 | Final probability of using model predictions. Too high → model trains on garbage early |
| `strategy` | "linear" | Start simple. "inverse_sigmoid" gives slower start, faster middle |

---

## 6) Key Design Decisions

### Why detach `prev_pred`?

```python
prev_pred.detach()
```

If you don't detach, gradients flow back through all previous steps (like BPTT in RNNs). This:
- Explodes memory usage for long sequences
- Can cause gradient instability
- Is usually unnecessary — the learning signal from each step is sufficient

Detaching means each step is trained independently, but with a more realistic input distribution. This is the standard approach in scheduled sampling literature.

### Why not always use p=1.0?

Starting with p=0 (ground truth) ensures the model first learns reasonable single-step predictions. Jumping to p=1.0 immediately means the model trains on random/garbage predictions → diverges.

### Interaction with existing validation/test

Validation and test remain unchanged — they already do full autoregressive rollout. The scheduled sampling only affects training. Over time, the model should produce better rollouts because it has seen its own errors during training.

---

## 7) Expected Effects

| Metric | Expected Change | Why |
|--------|----------------|-----|
| Rollout variance | Increase | Model learns to make bolder steps when input is slightly off |
| Rollout MSE | Decrease | Model recovers from own errors instead of drifting |
| Teacher-forced MSE | Slight increase | Model optimizes less perfectly for the ground-truth-input case |
| Training time | ~2-3x slower | Sequential processing within each batch |

---

## 8) Combination with Other Fixes

Scheduled sampling is **complementary** to other approaches:

- **+ Distributional loss (option 2):** Scheduled sampling fixes the train/infer gap; distributional loss fixes the mean-regression tendency. Together they address both root causes.
- **+ Displacement prediction (option 4):** Can predict displacement Δ with scheduled sampling — the model learns to predict bold Δ even from imperfect starting states.

**Recommended combination:** Heteroscedastic Gaussian NLL loss + Scheduled Sampling + keep the original (x1 prediction) formulation. This is the smallest architectural change with the largest expected benefit.

---

## 9) Potential Pitfalls

1. **Training instability in early epochs**: If `p_sample` ramps too fast, the model gets garbage inputs before it has learned anything. Solution: generous warmup (50+ epochs with p=0).

2. **Sequence boundary effects**: The first step in each sequence always uses ground truth. If `seq_len` is too short (e.g., 2), you barely get any scheduled sampling benefit. Use seq_len >= 4.

3. **Memory/history corruption**: When using model predictions, the memory slots become predictions-of-predictions. This can drift badly. Consider only replacing the most recent memory slot (not all of them) with the model prediction.

4. **Batch size reduction**: Sequences take more memory than single pairs. You may need to reduce batch size. Monitor GPU memory.

5. **Evaluation frequency**: Since training is slower, you may want to reduce `check_val_every_n_epoch` to catch divergence early.
