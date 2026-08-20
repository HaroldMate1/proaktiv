"""Performance metrics, reported error-first.

Reviewer 1: "Performance should generally be compared using RMSE/MSE, not
correlation (Figure 1 etc)."

The submitted code reported two different quantities under the name ``MSE``: the
per-epoch table header in ``fingerprints_plm.py`` said ``MSE`` but was passed a
root-mean-square error, while the figure titles used a genuine
``mean_squared_error``. Everything here is named for what it actually computes,
and correlation is available but explicitly marked supplementary.
"""

from __future__ import annotations

import numpy as np

# Any test RMSE far below this is memorisation of duplicated measurements rather
# than accuracy: it is the dispersion of the labels themselves. Measured from
# replicate groups in the curated data (results/curation/noise_floor.json).
ASSAY_RMSE_FLOOR = 0.78


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error over all samples.

    Computed on pooled residuals, not by averaging per-batch losses and taking
    the root afterwards -- that is what the training loop did, and it is biased
    whenever the final batch is smaller than the rest.
    """
    return float(np.sqrt(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_pred) - np.asarray(y_true)) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0:
        return float("nan")
    ss_res = float(np.sum((np.asarray(y_pred) - y_true) ** 2))
    return 1.0 - ss_res / ss_tot


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    statistic=rmse,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 1,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a metric."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    draws = [
        statistic(y_true[pick], y_pred[pick])
        for pick in (rng.choice(idx, size=len(idx), replace=True) for _ in range(n_boot))
    ]
    lo, hi = np.percentile(draws, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    seed: int = 1,
) -> dict:
    """Full error-first metric block for one evaluation set."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    point = rmse(y_true, y_pred)
    lo, hi = bootstrap_ci(y_true, y_pred, rmse, n_boot, seed=seed)
    out = {
        "n": int(len(y_true)),
        "rmse": round(point, 3),
        "rmse_ci_low": round(lo, 3),
        "rmse_ci_high": round(hi, 3),
        "mse": round(mse(y_true, y_pred), 3),
        "mae": round(mae(y_true, y_pred), 3),
        "r2": round(r2(y_true, y_pred), 3),
        # Below the assay noise floor is a warning sign, not an achievement.
        "below_assay_noise_floor": bool(point < ASSAY_RMSE_FLOOR),
    }
    if len(y_true) > 1 and np.std(y_pred) > 0:
        out["pearson_r_supplementary"] = round(
            float(np.corrcoef(y_true, y_pred)[0, 1]), 3
        )
        order_t = np.argsort(np.argsort(y_true))
        order_p = np.argsort(np.argsort(y_pred))
        out["spearman_rho_supplementary"] = round(
            float(np.corrcoef(order_t, order_p)[0, 1]), 3
        )
    return out


def stratified_report(
    y_true: np.ndarray, y_pred: np.ndarray, groups, min_n: int = 20, seed: int = 1
) -> list[dict]:
    """Per-group metrics, for the kinase and variant-frequency strata."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    groups = np.asarray(groups)
    rows = []
    for name in sorted(set(groups.tolist())):
        mask = groups == name
        if mask.sum() < min_n:
            rows.append({"group": name, "n": int(mask.sum()), "note": "below min_n"})
            continue
        rows.append({"group": name, **report(y_true[mask], y_pred[mask], seed=seed)})
    return rows
