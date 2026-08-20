"""Tests for error-first metrics and uncertainty evaluation."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proaktiv.evaluation import metrics, uncertainty  # noqa: E402


def test_rmse_mse_and_mae_are_distinct_and_correct():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    p = np.array([1.0, 2.0, 3.0, 8.0])  # single residual of 4
    assert metrics.mse(y, p) == pytest.approx(4.0)
    assert metrics.rmse(y, p) == pytest.approx(2.0)
    assert metrics.mae(y, p) == pytest.approx(1.0)
    # The submitted code reported RMSE under the label "MSE"; these must differ.
    assert metrics.rmse(y, p) != metrics.mse(y, p)


def test_rmse_is_pooled_not_batch_averaged():
    """Averaging per-batch MSE then rooting is biased for unequal batch sizes."""
    y = np.zeros(5)
    p = np.array([1.0, 1.0, 1.0, 1.0, 3.0])
    pooled = metrics.rmse(y, p)
    batch_a, batch_b = slice(0, 4), slice(4, 5)
    batched = np.sqrt(np.mean([metrics.mse(y[batch_a], p[batch_a]),
                               metrics.mse(y[batch_b], p[batch_b])]))
    assert pooled == pytest.approx(np.sqrt(13 / 5))
    assert batched != pytest.approx(pooled)


def test_perfect_prediction_gives_zero_error_and_unit_r2():
    y = np.array([5.0, 6.0, 7.0, 8.0])
    assert metrics.rmse(y, y) == pytest.approx(0.0)
    assert metrics.r2(y, y) == pytest.approx(1.0)


def test_report_flags_results_below_the_assay_noise_floor():
    y = np.linspace(5, 9, 200)
    rng = np.random.default_rng(0)
    tight = y + rng.normal(0, 0.05, y.size)
    loose = y + rng.normal(0, 1.5, y.size)
    assert metrics.report(y, tight)["below_assay_noise_floor"] is True
    assert metrics.report(y, loose)["below_assay_noise_floor"] is False


def test_report_marks_correlation_as_supplementary():
    y = np.linspace(5, 9, 50)
    out = metrics.report(y, y + 0.1)
    assert "pearson_r_supplementary" in out
    assert "spearman_rho_supplementary" in out
    assert "pearson_r" not in out  # must not be presentable as a primary metric


def test_correlation_can_be_perfect_while_error_is_large():
    """Reviewer 1's objection to correlation, encoded as a test."""
    y = np.linspace(5, 9, 50)
    biased = y + 3.0  # perfectly correlated, badly wrong
    out = metrics.report(y, biased)
    assert out["pearson_r_supplementary"] == pytest.approx(1.0)
    assert out["rmse"] == pytest.approx(3.0, abs=1e-6)


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.normal(7, 1, 300)
    p = y + rng.normal(0, 0.5, 300)
    lo, hi = metrics.bootstrap_ci(y, p, n_boot=200)
    assert lo <= metrics.rmse(y, p) <= hi


def test_stratified_report_marks_small_groups():
    y = np.linspace(5, 9, 60)
    groups = np.array(["big"] * 55 + ["small"] * 5)
    rows = {r["group"]: r for r in metrics.stratified_report(y, y + 0.2, groups)}
    assert rows["small"]["note"] == "below min_n"
    assert "rmse" in rows["big"]


# --- uncertainty ----------------------------------------------------------


def test_error_association_detects_a_real_relationship():
    rng = np.random.default_rng(0)
    unc = rng.uniform(0.1, 2.0, 400)
    err = unc * 1.5 + rng.normal(0, 0.1, 400)  # error genuinely grows with unc
    out = uncertainty.error_association(np.abs(err), unc)
    assert out["spearman_rho"] > 0.8
    assert out["association_supported"] is True


def test_error_association_rejects_an_absent_relationship():
    """Guards against restating the manuscript's unsupported Figure 2C claim."""
    rng = np.random.default_rng(0)
    unc = rng.uniform(0.1, 2.0, 400)
    err = np.abs(rng.normal(0, 1, 400))  # independent of uncertainty
    out = uncertainty.error_association(err, unc)
    assert out["association_supported"] is False
    assert out["spearman_ci_low"] <= 0 <= out["spearman_ci_high"]


def test_high_error_detection_auroc_is_chance_for_random_uncertainty():
    rng = np.random.default_rng(3)
    err = np.abs(rng.normal(0, 1, 500))
    unc = rng.uniform(0, 1, 500)
    out = uncertainty.high_error_detection(err, unc, threshold=0.78)
    assert 0.4 < out["auroc"] < 0.6


def test_high_error_detection_auroc_is_high_for_informative_uncertainty():
    rng = np.random.default_rng(3)
    unc = rng.uniform(0, 2, 500)
    err = unc + rng.normal(0, 0.05, 500)
    out = uncertainty.high_error_detection(np.abs(err), unc, threshold=0.78)
    assert out["auroc"] > 0.9


def test_risk_coverage_error_rises_with_coverage_when_uncertainty_is_useful():
    rng = np.random.default_rng(5)
    unc = rng.uniform(0, 2, 400)
    err = unc + rng.normal(0, 0.05, 400)
    curve, aurc = uncertainty.risk_coverage(np.abs(err), unc)
    assert curve[0]["rmse"] < curve[-1]["rmse"]
    assert curve[-1]["coverage"] == pytest.approx(1.0)
    assert aurc > 0


def test_dropout_enabled_restores_every_module_mode():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    model = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5), nn.Linear(4, 1))
    model.eval()
    with uncertainty.dropout_enabled(model, head_only=False):
        assert any(m.training for m in model.modules() if isinstance(m, nn.Dropout))
    # The bug this guards: plm_prediction.py never called model.eval() again.
    assert not any(m.training for m in model.modules())


def test_dropout_enabled_can_leave_the_protein_encoder_deterministic():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    model = nn.Module()
    model.protein_encoder = nn.Sequential(nn.Dropout(0.5))
    model.head = nn.Sequential(nn.Dropout(0.5))
    model.eval()
    with uncertainty.dropout_enabled(model, head_only=True):
        assert model.head[0].training is True
        assert model.protein_encoder[0].training is False
