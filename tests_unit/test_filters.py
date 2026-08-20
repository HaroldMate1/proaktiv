"""Tests for eligibility filtering and replicate aggregation."""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proaktiv.curation import filters  # noqa: E402


@pytest.fixture(scope="module")
def config():
    return filters.load_config()


def _row(**overrides):
    base = {
        "molecule_chembl_id": "CHEMBL1",
        "target_pref_name": "Epidermal growth factor receptor erbB1",
        "assay_variant_mutation": "T790M",
        "canonical_smiles": "c1ccccc1",
        "variant_mutation_sequence": "MABC",
        "standard_relation": "=",
        "standard_units": "nM",
        "standard_value": 100.0,
        "bao_label": "single protein format",
        "document_chembl_id": "CHEMBL_DOC1",
    }
    base.update(overrides)
    return base


def test_exact_nanomolar_record_is_eligible(config):
    eligible, excluded = filters.apply_eligibility(pd.DataFrame([_row()]), config)
    assert len(eligible) == 1 and excluded.empty
    assert eligible["pic50"].iloc[0] == pytest.approx(7.0)


@pytest.mark.parametrize("relation", [">", "<", ">=", "<=", ">>", "~"])
def test_censored_records_are_quarantined(config, relation):
    # A ">10000 nM" record is a non-binder, not an exact pIC50 of 5.
    df = pd.DataFrame([_row(standard_relation=relation, standard_value=10000.0)])
    eligible, excluded = filters.apply_eligibility(df, config)
    assert eligible.empty
    assert excluded["exclusion_reason"].iloc[0] == "censored_relation"


@pytest.mark.parametrize("unit", ["ug.mL-1", "/uM"])
def test_non_nanomolar_units_are_quarantined(config, unit):
    df = pd.DataFrame([_row(standard_units=unit)])
    eligible, excluded = filters.apply_eligibility(df, config)
    assert eligible.empty
    assert excluded["exclusion_reason"].iloc[0] == "non_nanomolar_unit"


def test_unresolved_mutation_label_is_quarantined(config):
    df = pd.DataFrame([_row(assay_variant_mutation="Other Mutation")])
    eligible, excluded = filters.apply_eligibility(df, config)
    assert eligible.empty
    assert excluded["exclusion_reason"].iloc[0] == "unresolved_mutation_label"


def test_every_input_row_is_accounted_for(config):
    df = pd.DataFrame([
        _row(),
        _row(standard_relation=">"),
        _row(standard_units="/uM"),
        _row(standard_value=0.0),
        _row(assay_variant_mutation="Other Mutation"),
    ])
    eligible, excluded = filters.apply_eligibility(df, config)
    assert len(eligible) + len(excluded) == len(df)


def test_assay_classes_are_not_pooled(config):
    df = pd.DataFrame([
        _row(bao_label="single protein format"),
        _row(bao_label="cell-based format"),
    ])
    eligible, _ = filters.apply_eligibility(df, config)
    assert set(eligible["assay_class"]) == {"biochemical", "cell_based"}
    # The replicate key includes assay_class, so the two must not merge.
    assert len(filters.aggregate_replicates(eligible, config)) == 2


def test_replicates_aggregate_to_median_and_retain_dispersion(config):
    df = pd.DataFrame([
        _row(standard_value=100.0),
        _row(standard_value=1000.0),
        _row(standard_value=10.0),
    ])
    eligible, _ = filters.apply_eligibility(df, config)
    out = filters.aggregate_replicates(eligible, config)
    assert len(out) == 1
    assert out["pic50"].iloc[0] == pytest.approx(7.0)  # median of 6, 7, 8
    assert out["n_replicates"].iloc[0] == 3
    assert out["pic50_sd"].iloc[0] == pytest.approx(1.0)
    # The flag threshold is exclusive, so an SD of exactly 1.0 is not flagged.
    assert bool(out["high_dispersion"].iloc[0]) is False


def test_dispersion_flag_fires_above_the_threshold(config):
    df = pd.DataFrame([
        _row(standard_value=10.0),      # pIC50 8
        _row(standard_value=10000.0),   # pIC50 5
        _row(standard_value=100.0),     # pIC50 7
    ])
    eligible, _ = filters.apply_eligibility(df, config)
    out = filters.aggregate_replicates(eligible, config)
    assert out["pic50_sd"].iloc[0] > 1.0
    assert bool(out["high_dispersion"].iloc[0]) is True


def test_noise_floor_ignores_singleton_groups():
    out = pd.DataFrame({"n_replicates": [1, 1], "pic50_sd": [0.0, 0.0]})
    assert filters.noise_floor(out)["n_groups"] == 0
