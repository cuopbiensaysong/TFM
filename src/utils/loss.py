import torch
import math


def mse_loss(pred, true):
    return torch.mean((pred - true) ** 2)


def l1_loss(pred, true):
    return torch.mean(torch.abs(pred - true))


# ---------------------------------------------------------------------------
# Distributional losses (anti-mean-regression)
# ---------------------------------------------------------------------------

def gaussian_nll_loss(mu, log_var, target):
    """Heteroscedastic Gaussian negative log-likelihood.

    The network outputs both mean and log-variance per dimension.
    Loss = 0.5 * mean(log_var + (target - mu)^2 / exp(log_var))

    Args:
        mu: [B, dim] predicted mean
        log_var: [B, dim] predicted log-variance (clamped externally)
        target: [B, dim] ground truth

    Returns:
        scalar loss
    """
    var = torch.exp(log_var)
    nll = 0.5 * (log_var + (target - mu) ** 2 / (var + 1e-8))
    return nll.mean()


def mdn_loss(pi, mu, sigma, target):
    """Mixture Density Network loss (negative log-likelihood of GMM).

    Args:
        pi: [B, K] mixture weights (log-softmax already applied)
        mu: [B, K, dim] component means
        sigma: [B, K, dim] component standard deviations (positive)
        target: [B, dim] ground truth

    Returns:
        scalar loss
    """
    K = mu.shape[1]
    target_expanded = target.unsqueeze(1).expand_as(mu)  # [B, K, dim]

    # Log probability under each Gaussian component
    log_norm = -0.5 * (math.log(2 * math.pi) + 2 * torch.log(sigma + 1e-8) +
                       ((target_expanded - mu) / (sigma + 1e-8)) ** 2)
    log_norm = log_norm.sum(dim=-1)  # [B, K] sum over dimensions

    # Log-sum-exp with mixture weights
    log_prob = torch.logsumexp(pi + log_norm, dim=1)  # [B]
    return -log_prob.mean()


def quantile_loss(pred, target, quantiles=(0.1, 0.5, 0.9)):
    """Pinball/quantile loss for multiple quantiles.

    The network outputs one prediction per quantile per dimension.

    Args:
        pred: [B, num_quantiles, dim] predicted quantile values
        target: [B, dim] ground truth
        quantiles: tuple of quantile levels

    Returns:
        scalar loss
    """
    target_expanded = target.unsqueeze(1).expand_as(pred)  # [B, Q, dim]
    total_loss = torch.tensor(0.0, device=pred.device)
    for i, q in enumerate(quantiles):
        errors = target_expanded[:, i, :] - pred[:, i, :]
        total_loss = total_loss + torch.mean(torch.max(q * errors, (q - 1) * errors))
    return total_loss / len(quantiles)


def l1_variance_loss(pred, target, lambda_var=0.1):
    """L1 loss with variance encouragement penalty.

    Combines L1 reconstruction with a term that penalizes low variance
    across predictions in the batch, encouraging bolder predictions.

    Args:
        pred: [B, dim] predictions
        target: [B, dim] ground truth
        lambda_var: weight for variance penalty (negative = encourage variance)

    Returns:
        scalar loss
    """
    reconstruction = torch.mean(torch.abs(pred - target))
    pred_var = pred.var(dim=0).mean()
    return reconstruction - lambda_var * pred_var


# ---------------------------------------------------------------------------
# Registry for config-based resolution
# ---------------------------------------------------------------------------

LOSS_REGISTRY = {
    "mse_loss": mse_loss,
    "l1_loss": l1_loss,
    "gaussian_nll": "gaussian_nll",
    "mdn": "mdn",
    "quantile": "quantile",
    "l1_variance": "l1_variance",
}


def resolve_loss_fn(loss_fn):
    """Resolve a loss function from config string or callable."""
    if callable(loss_fn):
        return loss_fn
    if isinstance(loss_fn, str):
        if loss_fn not in LOSS_REGISTRY:
            raise ValueError(
                f"Unsupported loss_fn '{loss_fn}'. Choose from: {list(LOSS_REGISTRY.keys())}"
            )
        val = LOSS_REGISTRY[loss_fn]
        if callable(val):
            return val
        return val  # string sentinel for distributional losses handled in the module
    raise TypeError(f"loss_fn must be callable or str, got {type(loss_fn)}")
