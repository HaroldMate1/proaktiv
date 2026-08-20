"""Eligibility filtering and replicate aggregation for the primary endpoint.

The submitted pipeline read ``standard_value`` straight into
``-log10(value * 1e-9)`` with no relation or unit check, so censored records and
non-nanomolar records became exact pIC50 labels. This module applies the rules
declared in ``configs/curation.yml`` and records a reason for every excluded row.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "curation.yml"


def load_config(path: Path | str = CONFIG_PATH) -> dict:
    return yaml.safe_load(Path(path).read_text())


def assign_assay_class(df: pd.DataFrame, config: dict) -> pd.Series:
    """Map ``bao_label`` onto comparable assay classes."""
    lookup = {
        label: klass
        for klass, labels in config["assay_classes"].items()
        for label in labels
    }
    return df["bao_label"].map(lookup).fillna("unspecified")


def to_pic50(values: pd.Series) -> pd.Series:
    """Convert IC50 in nM to pIC50, leaving non-positive values as NaN."""
    safe = values.where(values > 0)
    return -np.log10(safe * 1e-9)


def apply_eligibility(
    df: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ``df`` into eligible records and an exclusion log.

    Returns:
        (eligible, excluded) where ``excluded`` carries an ``exclusion_reason``
        column. Every input row appears in exactly one of the two frames.
    """
    endpoint = config["endpoint"]
    work = df.copy()
    work["assay_class"] = assign_assay_class(work, config)
    work["pic50"] = to_pic50(work["standard_value"])

    reason = pd.Series(pd.NA, index=work.index, dtype="object")

    def mark(mask: pd.Series, text: str) -> None:
        reason.loc[mask & reason.isna()] = text

    censored = set(endpoint["censored_relations"])
    accepted = set(endpoint["accepted_relations"])
    relation = work["standard_relation"].astype("string")

    mark(relation.isin(censored), "censored_relation")
    mark(~relation.isin(accepted), "unrecognised_relation")
    mark(~work["standard_units"].astype("string").isin(endpoint["accepted_units"]),
         "non_nanomolar_unit")
    mark(work["assay_variant_mutation"].astype("string").isin(
        config["mutation_labels"]["quarantine"]), "unresolved_mutation_label")
    mark(work["variant_mutation_sequence"].isna(), "missing_sequence")
    mark(work["canonical_smiles"].isna(), "missing_structure")
    mark(work["pic50"].isna(), "non_positive_or_missing_ic50")

    low, high = endpoint["pic50_bounds"]
    mark(~work["pic50"].between(low, high), "pic50_out_of_bounds")

    excluded_mask = reason.notna()
    excluded = work.loc[excluded_mask].assign(exclusion_reason=reason.loc[excluded_mask])
    eligible = work.loc[~excluded_mask].copy()
    return eligible, excluded


def aggregate_replicates(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Collapse biological replicates to one row per replicate group.

    Retains the replicate count and within-group dispersion so downstream
    analysis can report the assay noise floor rather than hiding it.
    """
    spec = config["replicates"]
    keys = spec["group_keys"]
    grouped = df.groupby(keys, dropna=False)

    out = grouped.agg(
        pic50=("pic50", spec["aggregate"]),
        pic50_sd=("pic50", "std"),
        pic50_min=("pic50", "min"),
        pic50_max=("pic50", "max"),
        n_replicates=("pic50", "size"),
        canonical_smiles=("canonical_smiles", "first"),
        variant_mutation_sequence=("variant_mutation_sequence", "first"),
        n_documents=("document_chembl_id", "nunique"),
    ).reset_index()

    out["pic50_sd"] = out["pic50_sd"].fillna(0.0)
    out["high_dispersion"] = out["pic50_sd"] > spec["dispersion_flag_sd"]
    return out


def noise_floor(df: pd.DataFrame) -> dict:
    """Summarise within-replicate-group dispersion: the irreducible error."""
    replicated = df.loc[df["n_replicates"] > 1]
    if replicated.empty:
        return {"n_groups": 0}
    return {
        "n_groups": int(len(replicated)),
        "median_sd": float(replicated["pic50_sd"].median()),
        "mean_sd": float(replicated["pic50_sd"].mean()),
        "p90_sd": float(replicated["pic50_sd"].quantile(0.90)),
        "n_sd_over_1": int((replicated["pic50_sd"] > 1.0).sum()),
        # A model cannot beat the measurement it is trained on: RMSE at or below
        # this value on replicated pairs indicates leakage, not accuracy.
        "implied_rmse_floor": float(
            np.sqrt((replicated["pic50_sd"] ** 2).mean())
        ),
    }
