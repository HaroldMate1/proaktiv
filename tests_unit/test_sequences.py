"""Tests for kinase-domain windowing.

The regression these guard against: ALK variants collapsing onto wild type when
the encoder truncated full-length sequences at 1024 tokens.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proaktiv.curation.sequences import (  # noqa: E402
    KinaseWindow,
    SequenceWindowError,
    apply_window,
    load_windows,
    mutation_is_encoded,
)


@pytest.fixture(scope="module")
def windows():
    return load_windows()


def test_all_windows_fit_encoder_budget(windows):
    for window in windows.values():
        assert window.size <= 1022, f"{window.name} exceeds the ESM2 residue budget"


def test_alk_window_covers_clinical_resistance_mutations(windows):
    alk = windows["ALK"]
    # Every ALK mutation in the curated data. All lie past residue 1024 and were
    # therefore lost to the previous truncation.
    positions = [1151, 1152, 1156, 1174, 1196, 1202, 1206, 1269, 1275]
    assert all(p > 1024 for p in positions), "premise of the regression"
    assert mutation_is_encoded(alk, positions)


def test_egfr_window_covers_observed_mutations(windows):
    egfr = windows["EGFR"]
    assert mutation_is_encoded(egfr, [709, 710, 746, 750, 763, 770, 790, 797, 858, 861])


def test_braf_window_covers_observed_mutations(windows):
    # P731T sits outside the annotated kinase domain (457-717); the flank is what
    # brings it inside the window.
    assert mutation_is_encoded(windows["BRAF"], [589, 600, 601, 731])


def _toy_window(length=100, start=21, end=80):
    return KinaseWindow("TOY", "P00000", "toy", length, start, end)


def test_apply_window_substitution_preserves_length():
    window = _toy_window()
    wild = "A" * 100
    variant = "A" * 49 + "M" + "A" * 50  # position 50
    cut = apply_window(window, variant, wild)
    assert len(cut) == window.size
    assert cut[50 - window.start] == "M"


def test_apply_window_keeps_indel_variants_in_register():
    """A deletion inside the window shortens the cut but keeps flanks aligned."""
    window = _toy_window()
    wild = "A" * 100
    variant = "A" * 95  # five residues deleted inside the window
    cut = apply_window(window, variant, wild)
    assert len(cut) == window.size - 5


def test_apply_window_rejects_variation_outside_the_window():
    window = _toy_window()
    wild = "A" * 100
    outside = "C" + "A" * 99  # position 1, before the window
    with pytest.raises(SequenceWindowError, match="before residue"):
        apply_window(window, outside, wild)


def test_apply_window_rejects_wrong_reference_length():
    with pytest.raises(SequenceWindowError, match="expected"):
        apply_window(_toy_window(), "A" * 100, "A" * 99)
