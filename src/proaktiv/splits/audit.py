"""Leakage audit for split manifests.

A manifest is only worth reporting if it demonstrably keeps the forbidden groups
apart. This module measures overlap across partitions for every grouping the
reviewers care about and returns a pass/fail verdict per scheme.
"""

from __future__ import annotations

import pandas as pd

# Grouping -> whether crossing partitions disqualifies the split.
AUDIT_COLUMNS = {
    "molecule_chembl_id": "compound",
    "scaffold": "scaffold",
    "assay_variant_mutation": "variant",
    "variant_mutation_sequence": "sequence",
    "document_chembl_id": "document",
}


def overlap_report(
    df: pd.DataFrame, assignment: pd.Series, columns=AUDIT_COLUMNS
) -> pd.DataFrame:
    """Count group identities shared between train and each held-out partition."""
    rows = []
    for column, label in columns.items():
        if column not in df.columns:
            continue
        train = set(df.loc[assignment == "train", column].dropna())
        for partition in ("valid", "test"):
            held = set(df.loc[assignment == partition, column].dropna())
            shared = train & held
            rows.append(
                {
                    "grouping": label,
                    "partition": partition,
                    "n_held_out_groups": len(held),
                    "n_shared_with_train": len(shared),
                    "pct_shared": round(100 * len(shared) / len(held), 1) if held else 0.0,
                }
            )
    return pd.DataFrame(rows)


def verdict(report: pd.DataFrame, must_not_share: list[str]) -> tuple[bool, list[str]]:
    """Fail the manifest if any grouping in ``must_not_share`` crosses partitions."""
    problems = []
    for grouping in must_not_share:
        offending = report[
            (report["grouping"] == grouping) & (report["n_shared_with_train"] > 0)
        ]
        for _, row in offending.iterrows():
            problems.append(
                f"{grouping} leaks into {row['partition']}: "
                f"{row['n_shared_with_train']} shared groups "
                f"({row['pct_shared']}% of held-out)"
            )
    return len(problems) == 0, problems


# What each scheme is required to isolate.
REQUIREMENTS = {
    "random": [],
    "scaffold": ["scaffold"],
    "variant": ["variant", "sequence"],
    "combined": ["scaffold", "variant", "sequence"],
}
