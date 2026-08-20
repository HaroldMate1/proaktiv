"""Tests for split strategies and the leakage audit.

These encode the reviewers' core objection: a split is only acceptable if the
grouping it claims to isolate genuinely does not cross partitions.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proaktiv.splits import audit, strategies  # noqa: E402


@pytest.fixture
def frame():
    rng = np.random.default_rng(0)
    n = 600
    return pd.DataFrame({
        "molecule_chembl_id": [f"CHEMBL{i % 120}" for i in range(n)],
        "scaffold": [f"S{i % 40}" for i in range(n)],
        "assay_variant_mutation": rng.choice(
            ["Wild Type", "T790M", "L858R", "C797S", "G1202R"], n
        ),
        "variant_mutation_sequence": ["SEQ"] * n,
        "document_year": rng.choice([2015, 2018, 2021, 2023], n),
        "pic50": rng.normal(7, 1, n),
    })


def test_scaffold_split_isolates_scaffolds(frame):
    manifest = strategies.scaffold_split(frame, seed=1)
    report = audit.overlap_report(frame, manifest.assignment)
    passed, problems = audit.verdict(report, audit.REQUIREMENTS["scaffold"])
    assert passed, problems


def test_variant_split_isolates_variants(frame):
    manifest = strategies.variant_split(frame, seed=1)
    report = audit.overlap_report(frame, manifest.assignment)
    passed, problems = audit.verdict(report, ["variant"])
    assert passed, problems


def test_variant_split_pins_wild_type_to_training(frame):
    manifest = strategies.variant_split(frame, seed=1)
    wild = frame["assay_variant_mutation"] == "Wild Type"
    assert set(manifest.assignment[wild]) == {"train"}


def test_combined_split_isolates_both_groupings(frame):
    manifest = strategies.combined_split(frame, seed=1)
    report = audit.overlap_report(frame, manifest.assignment)
    passed, problems = audit.verdict(report, ["scaffold", "variant"])
    assert passed, problems


def test_random_split_leaks_and_the_audit_detects_it(frame):
    # Guards the claim we make to the reviewers about the submitted scheme.
    manifest = strategies.random_split(frame, seed=1)
    report = audit.overlap_report(frame, manifest.assignment)
    passed, problems = audit.verdict(report, ["scaffold", "variant"])
    assert not passed and problems


def test_splits_are_deterministic(frame):
    a = strategies.scaffold_split(frame, seed=7).assignment
    b = strategies.scaffold_split(frame, seed=7).assignment
    pd.testing.assert_series_equal(a, b)


def test_different_seeds_give_different_partitions(frame):
    a = strategies.scaffold_split(frame, seed=1).assignment
    b = strategies.scaffold_split(frame, seed=2).assignment
    assert not a.equals(b)


def test_leave_one_mutation_out_holds_out_exactly_that_variant(frame):
    manifest = strategies.leave_one_mutation_out(frame, "T790M")
    held = frame.loc[manifest.assignment == "test", "assay_variant_mutation"]
    assert set(held) == {"T790M"}
    assert "T790M" not in set(
        frame.loc[manifest.assignment == "train", "assay_variant_mutation"]
    )


def test_temporal_split_puts_later_work_in_test(frame):
    manifest = strategies.temporal_split(frame, cutoff_year=2021)
    years = frame.loc[manifest.assignment == "test", "document_year"]
    assert years.min() >= 2021
    assert frame.loc[manifest.assignment == "train", "document_year"].max() < 2021


def test_every_row_is_assigned(frame):
    for name in ("random", "scaffold", "variant"):
        manifest = getattr(strategies, f"{name}_split")(frame, seed=1)
        assert manifest.assignment.notna().all()
        assert "unassigned" not in set(manifest.assignment)
