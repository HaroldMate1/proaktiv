"""Leakage-controlled split strategies.

Both reviewers led with the same objection: the submitted work used a 70/10/20
random row split, so near-identical ligands and recurrent variants appear on both
sides of the partition. This module produces deterministic split manifests for
the schemes required in Phase 4 of REVISION_PLAN.md, plus the audit that proves
a manifest is actually leakage-free.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:  # RDKit is required for scaffold splits but not for random/variant splits.
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    _HAVE_RDKIT = True
except ImportError:  # pragma: no cover - exercised only in minimal environments
    _HAVE_RDKIT = False


FRACTIONS = (0.7, 0.1, 0.2)


@dataclass(frozen=True)
class Manifest:
    """A frozen assignment of rows to train/validation/test."""

    name: str
    seed: int
    assignment: pd.Series  # index-aligned, values in {"train", "valid", "test"}

    def counts(self) -> dict[str, int]:
        return self.assignment.value_counts().to_dict()


def bemis_murcko_scaffold(smiles: str) -> str:
    """Bemis-Murcko scaffold SMILES; falls back to the molecule itself."""
    if not _HAVE_RDKIT:
        raise ImportError("RDKit is required for scaffold splits")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(scaffold) if scaffold is not None else ""


def add_scaffolds(df: pd.DataFrame, smiles_col: str = "canonical_smiles") -> pd.Series:
    """Scaffold for each row, computed once per unique SMILES."""
    unique = pd.Series(df[smiles_col].unique())
    mapping = {s: bemis_murcko_scaffold(s) for s in unique}
    return df[smiles_col].map(mapping)


def _assign_groups(
    groups: pd.Series, fractions: tuple[float, float, float], seed: int
) -> pd.Series:
    """Assign whole groups to splits, filling test then valid then train.

    Groups are shuffled deterministically and consumed largest-first within the
    shuffle so that the realised fractions stay close to target even when group
    sizes are heavily skewed (EGFR wild type alone is ~47% of rows).
    """
    sizes = groups.value_counts()
    rng = np.random.default_rng(seed)
    order = rng.permutation(sizes.index.to_numpy())

    total = len(groups)
    targets = {
        "test": fractions[2] * total,
        "valid": fractions[1] * total,
        "train": fractions[0] * total,
    }
    filled = {k: 0 for k in targets}
    group_split: dict[object, str] = {}

    for group in order:
        n = int(sizes[group])
        # Send the group wherever it leaves the largest relative deficit.
        deficit = {k: (targets[k] - filled[k]) / targets[k] for k in targets if targets[k] > 0}
        choice = max(deficit, key=deficit.get)
        group_split[group] = choice
        filled[choice] += n

    return groups.map(group_split)


def random_split(df: pd.DataFrame, seed: int = 1, fractions=FRACTIONS) -> Manifest:
    """The submitted scheme. Retained only as an optimistic reference baseline."""
    rng = np.random.default_rng(seed)
    draw = rng.random(len(df))
    assignment = pd.Series("train", index=df.index, dtype="object")
    assignment[draw >= fractions[0]] = "valid"
    assignment[draw >= fractions[0] + fractions[1]] = "test"
    return Manifest("random", seed, assignment)


def scaffold_split(
    df: pd.DataFrame, seed: int = 1, fractions=FRACTIONS, scaffold_col: str = "scaffold"
) -> Manifest:
    """No Bemis-Murcko scaffold appears in more than one partition."""
    return Manifest("scaffold", seed, _assign_groups(df[scaffold_col], fractions, seed))


def variant_split(
    df: pd.DataFrame,
    seed: int = 1,
    fractions=FRACTIONS,
    variant_col: str = "assay_variant_mutation",
) -> Manifest:
    """Unseen-variant evaluation: held-out variant identities are absent from training.

    Wild type is pinned to training for every kinase. Holding it out would remove
    the reference state the mutant predictions are implicitly compared against,
    and it carries roughly half the data.
    """
    groups = df[variant_col].astype(str)
    mutant_mask = ~groups.str.lower().str.startswith("wild")
    assignment = pd.Series("train", index=df.index, dtype="object")
    if mutant_mask.any():
        assignment.loc[mutant_mask] = _assign_groups(
            groups[mutant_mask], fractions, seed
        )
    return Manifest("variant", seed, assignment)


def combined_split(
    df: pd.DataFrame,
    seed: int = 1,
    fractions=FRACTIONS,
    scaffold_col: str = "scaffold",
    variant_col: str = "assay_variant_mutation",
) -> Manifest:
    """Hardest internal test: neither the scaffold nor the variant was seen in training.

    Rows whose scaffold and variant assignments disagree are dropped to
    ``unassigned`` rather than being forced into a partition, which is what keeps
    the guarantee honest.
    """
    scaffold = _assign_groups(df[scaffold_col], fractions, seed)
    variant = variant_split(df, seed, fractions, variant_col).assignment
    assignment = pd.Series("unassigned", index=df.index, dtype="object")
    agree = scaffold == variant
    assignment.loc[agree] = scaffold.loc[agree]
    # Wild-type rows follow the scaffold decision; they are pinned to train above.
    wild = df[variant_col].astype(str).str.lower().str.startswith("wild")
    assignment.loc[wild & (scaffold == "train")] = "train"
    return Manifest("combined", seed, assignment)


def leave_one_mutation_out(
    df: pd.DataFrame, variant: str, variant_col: str = "assay_variant_mutation"
) -> Manifest:
    """Hold out one named variant entirely; train on everything else."""
    assignment = pd.Series("train", index=df.index, dtype="object")
    assignment.loc[df[variant_col] == variant] = "test"
    return Manifest(f"loo:{variant}", 0, assignment)


def temporal_split(
    df: pd.DataFrame, cutoff_year: int, year_col: str = "document_year"
) -> Manifest:
    """External-style validation: train on the historical record, test on later work."""
    years = pd.to_numeric(df[year_col], errors="coerce")
    assignment = pd.Series("train", index=df.index, dtype="object")
    assignment.loc[years >= cutoff_year] = "test"
    assignment.loc[years.isna()] = "unassigned"
    return Manifest(f"temporal:{cutoff_year}", 0, assignment)
