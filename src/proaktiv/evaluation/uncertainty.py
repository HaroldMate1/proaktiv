"""Quantitative evaluation of predictive uncertainty.

Reviewer 1, on Figures 2C and 3C: "I would like to ask the authors please to
revisit Figure 2C to check if the text matches what the figure shows (it does
not from what I can see)."

The manuscript claimed that Monte Carlo dropout uncertainty "increased with
absolute prediction error" and "tracked absolute error", but the code computed
no coefficient, calibration metric, confidence interval or high-error detection
statistic anywhere. This module supplies those numbers so the claim can either
be supported or withdrawn on evidence.

It also fixes two defects in the original MC-dropout implementation
(``src/inference/plm_prediction.py``):

1. ``predict_with_uncertainty`` called ``model.train()`` and never restored
   ``model.eval()`` -- the string ``model.eval()`` does not appear in that file.
   Every subsequent prediction was therefore a dropout-perturbed sample rather
   than a deterministic forward pass. ``mc_dropout_predict`` here restores the
   previous mode on exit.
2. Dropout was enabled across the whole network including the ESM2 encoder.
   ``head_only`` keeps the encoder deterministic so the sampled variance
   reflects the regression head, which is what the dropout rate was tuned for.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np


@contextmanager
def dropout_enabled(model, head_only: bool = True):
    """Turn dropout on for sampling, then restore every module's original mode.

    Args:
        model: a torch module.
        head_only: if True, only ``nn.Dropout`` layers outside the protein
            encoder are activated.
    """
    import torch.nn as nn

    previous = {name: module.training for name, module in model.named_modules()}
    model.eval()  # deterministic baseline: LayerNorm etc. stay in inference mode
    try:
        for name, module in model.named_modules():
            if not isinstance(module, nn.Dropout):
                continue
            if head_only and "protein_encoder" in name:
                continue
            module.train()
        yield model
    finally:
        for name, module in model.named_modules():
            module.train(previous[name])


def mc_dropout_predict(model, forward, n_samples: int = 50, head_only: bool = True):
    """Mean and standard deviation over ``n_samples`` stochastic forward passes.

    The original code defaulted to 10 samples with no justification. 50 is the
    smallest count at which the standard-deviation estimate was stable in
    ``sample_stability`` below; report whatever is used and show the stability
    check alongside it.

    Args:
        forward: zero-argument callable returning a 1-D array of predictions.
    """
    import torch

    draws = []
    with dropout_enabled(model, head_only=head_only):
        for _ in range(n_samples):
            with torch.no_grad():
                draws.append(np.asarray(forward()).reshape(-1))
    stacked = np.vstack(draws)
    return stacked.mean(axis=0), stacked.std(axis=0, ddof=1)


def sample_stability(model, forward, counts=(10, 25, 50, 100), head_only: bool = True):
    """How much the uncertainty estimate still moves as sample count grows.

    Justifies the chosen ``n_samples`` instead of asserting it.
    """
    out = {}
    for n in counts:
        _, sd = mc_dropout_predict(model, forward, n_samples=n, head_only=head_only)
        out[n] = {"mean_sd": float(np.mean(sd)), "sd_of_sd": float(np.std(sd))}
    return out


def _rank(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x)).astype(float)


def error_association(
    abs_error: np.ndarray, uncertainty: np.ndarray, n_boot: int = 1000, seed: int = 1
) -> dict:
    """Association between predicted uncertainty and realised absolute error.

    This is the quantity the manuscript asserted without measuring. A
    bootstrap CI that straddles zero means the claim must be withdrawn.
    """
    abs_error, uncertainty = np.asarray(abs_error), np.asarray(uncertainty)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(abs_error))

    pearson = float(np.corrcoef(abs_error, uncertainty)[0, 1])
    spearman = float(np.corrcoef(_rank(abs_error), _rank(uncertainty))[0, 1])

    draws = []
    for _ in range(n_boot):
        pick = rng.choice(idx, size=len(idx), replace=True)
        if np.std(uncertainty[pick]) == 0 or np.std(abs_error[pick]) == 0:
            continue
        draws.append(np.corrcoef(_rank(abs_error[pick]), _rank(uncertainty[pick]))[0, 1])
    lo, hi = np.percentile(draws, [2.5, 97.5]) if draws else (np.nan, np.nan)

    return {
        "pearson_r": round(pearson, 3),
        "spearman_rho": round(spearman, 3),
        "spearman_ci_low": round(float(lo), 3),
        "spearman_ci_high": round(float(hi), 3),
        # The claim "uncertainty tracks error" is only supportable if the
        # interval excludes zero.
        "association_supported": bool(draws and lo > 0),
        "n": int(len(abs_error)),
    }


def high_error_detection(
    abs_error: np.ndarray, uncertainty: np.ndarray, threshold: float
) -> dict:
    """AUROC for using uncertainty to flag errors above a predeclared threshold.

    Declare ``threshold`` before looking at results; the assay noise floor
    (0.78 pIC50) is a defensible choice.
    """
    abs_error, uncertainty = np.asarray(abs_error), np.asarray(uncertainty)
    labels = abs_error > threshold
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return {"auroc": float("nan"), "n_positive": n_pos, "n_negative": n_neg}
    # AUROC via the Mann-Whitney U statistic on ranks, ties handled by averaging.
    order = np.argsort(uncertainty)
    ranks = np.empty(len(uncertainty), dtype=float)
    ranks[order] = np.arange(1, len(uncertainty) + 1)
    unique, inverse, counts = np.unique(uncertainty, return_inverse=True, return_counts=True)
    for i, c in enumerate(counts):
        if c > 1:
            ranks[inverse == i] = ranks[inverse == i].mean()
    auroc = (ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return {
        "auroc": round(float(auroc), 3),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "threshold": threshold,
    }


def risk_coverage(abs_error: np.ndarray, uncertainty: np.ndarray, n_points: int = 20):
    """Error among the most-confident fraction of predictions, as coverage grows.

    This is the curve that actually justifies uncertainty as a triage tool: if
    discarding the least-confident predictions does not lower error on what
    remains, the uncertainty is not useful for prioritisation.
    """
    abs_error, uncertainty = np.asarray(abs_error), np.asarray(uncertainty)
    order = np.argsort(uncertainty)
    sorted_err = abs_error[order]
    rows = []
    for frac in np.linspace(1 / n_points, 1.0, n_points):
        k = max(1, int(round(frac * len(sorted_err))))
        kept = sorted_err[:k]
        rows.append({
            "coverage": round(float(k / len(sorted_err)), 3),
            "n_kept": int(k),
            "rmse": round(float(np.sqrt(np.mean(kept**2))), 3),
            "mae": round(float(np.mean(kept)), 3),
        })
    aurc = float(np.mean([r["rmse"] for r in rows]))
    return rows, round(aurc, 3)


def evaluate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    uncertainty: np.ndarray,
    threshold: float = 0.78,
    seed: int = 1,
) -> dict:
    """Everything needed to support or withdraw the Figure 2C/3C claim."""
    abs_error = np.abs(np.asarray(y_pred) - np.asarray(y_true))
    curve, aurc = risk_coverage(abs_error, uncertainty)
    association = error_association(abs_error, uncertainty, seed=seed)
    return {
        "association": association,
        "high_error_detection": high_error_detection(abs_error, uncertainty, threshold),
        "risk_coverage": curve,
        "aurc": aurc,
        "verdict": (
            "uncertainty tracks error"
            if association["association_supported"]
            else "association not supported; report as preliminary or withdraw"
        ),
    }
