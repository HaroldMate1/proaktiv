"""Build the curated PROAKTIV modelling table and its reviewer-facing summaries.

Produces, under results/curation/:
  curated.parquet        one row per replicate group, windowed sequences
  exclusions.csv         every dropped row with its reason
  table1_composition.csv per-kinase dataset composition (Reviewer 2, point 7)
  assay_heterogeneity.csv  assay-class breakdown (Reviewer 2, point 3)
  noise_floor.json       within-replicate dispersion, the irreducible error
"""

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proaktiv.curation import filters  # noqa: E402
from proaktiv.curation.sequences import (  # noqa: E402
    SequenceWindowError,
    apply_window,
    load_windows,
    window_by_target_name,
)
from proaktiv.splits.strategies import add_scaffolds  # noqa: E402

DATA = ROOT / "data" / "egfr_alk_braf_merged.xlsx"
OUT = ROOT / "results" / "curation"


def window_sequences(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Window every sequence; return the windowed column and a failure reason."""
    by_name = window_by_target_name(load_windows())
    windowed = pd.Series(pd.NA, index=df.index, dtype="object")
    failure = pd.Series(pd.NA, index=df.index, dtype="object")

    for target, group in df.groupby("target_pref_name"):
        window = by_name.get(target)
        if window is None:
            failure.loc[group.index] = "unknown_target"
            continue
        wild = group.loc[
            group["assay_variant_mutation"].astype(str).str.lower().str.startswith("wild"),
            "variant_mutation_sequence",
        ]
        if wild.empty:
            failure.loc[group.index] = "no_wild_type_reference"
            continue
        wild_type = wild.iloc[0]
        cache: dict[str, tuple[str | None, str | None]] = {}
        for idx, sequence in group["variant_mutation_sequence"].items():
            if sequence not in cache:
                try:
                    cache[sequence] = (apply_window(window, sequence, wild_type), None)
                except SequenceWindowError as exc:
                    cache[sequence] = (None, str(exc))
            value, err = cache[sequence]
            windowed.loc[idx] = value
            failure.loc[idx] = err
    return windowed, failure


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    config = filters.load_config()
    raw = pd.read_excel(DATA)
    print(f"raw records: {len(raw):,}")

    eligible, excluded = filters.apply_eligibility(raw, config)
    print(f"eligible after rules: {len(eligible):,}  excluded: {len(excluded):,}")
    print(excluded["exclusion_reason"].value_counts().to_string())

    windowed, failure = window_sequences(eligible)
    bad = failure.notna()
    if bad.any():
        extra = eligible.loc[bad].assign(
            exclusion_reason="sequence_window_failed: " + failure.loc[bad]
        )
        excluded = pd.concat([excluded, extra])
        eligible = eligible.loc[~bad]
    eligible = eligible.assign(windowed_sequence=windowed.loc[eligible.index])
    print(f"after windowing: {len(eligible):,}")

    eligible["scaffold"] = add_scaffolds(eligible)

    curated = filters.aggregate_replicates(eligible, config)
    # Carry scaffold and the raw-row provenance onto the aggregated table.
    keys = config["replicates"]["group_keys"]
    extras = (
        eligible.groupby(keys, dropna=False)
        .agg(
            scaffold=("scaffold", "first"),
            windowed_sequence=("windowed_sequence", "first"),
            document_year=("document_year", "min"),
            source_rows=("#", lambda s: ";".join(map(str, s))),
        )
        .reset_index()
    )
    curated = curated.merge(extras, on=keys, how="left")
    curated = curated.drop(columns=["variant_mutation_sequence"])
    print(f"replicate groups: {len(curated):,}")

    curated.to_parquet(OUT / "curated.parquet", index=False)
    excluded[
        ["#", "molecule_chembl_id", "target_pref_name", "assay_variant_mutation",
         "standard_value", "standard_relation", "standard_units", "exclusion_reason"]
    ].to_csv(OUT / "exclusions.csv", index=False)

    # ---- Table 1: composition per kinase -------------------------------------
    rows = []
    for target, group in curated.groupby("target_pref_name"):
        raw_group = raw[raw["target_pref_name"] == target]
        mutants = group[~group["assay_variant_mutation"].astype(str).str.lower().str.startswith("wild")]
        rows.append({
            "kinase": target,
            "raw_measurements": len(raw_group),
            "retained_measurements": int(group["n_replicates"].sum()),
            "replicate_groups": len(group),
            "unique_compounds": group["molecule_chembl_id"].nunique(),
            "unique_scaffolds": group["scaffold"].nunique(),
            "variant_states": group["assay_variant_mutation"].nunique(),
            "mutant_groups": len(mutants),
            "wild_type_groups": len(group) - len(mutants),
            "compound_mutations": sum(
                "," in str(v) for v in group["assay_variant_mutation"].unique()
            ),
            "pic50_median": round(group["pic50"].median(), 2),
            "pic50_iqr": round(group["pic50"].quantile(.75) - group["pic50"].quantile(.25), 2),
            "groups_with_replicates": int((group["n_replicates"] > 1).sum()),
            "high_dispersion_groups": int(group["high_dispersion"].sum()),
        })
    table1 = pd.DataFrame(rows)
    table1.to_csv(OUT / "table1_composition.csv", index=False)
    print("\n--- Table 1: dataset composition ---")
    print(table1.to_string(index=False))

    # ---- Assay heterogeneity -------------------------------------------------
    heterogeneity = (
        curated.groupby(["target_pref_name", "assay_class"])
        .agg(replicate_groups=("pic50", "size"),
             measurements=("n_replicates", "sum"),
             pic50_median=("pic50", "median"))
        .round(2)
        .reset_index()
    )
    heterogeneity.to_csv(OUT / "assay_heterogeneity.csv", index=False)
    print("\n--- Assay heterogeneity ---")
    print(heterogeneity.to_string(index=False))

    # ---- Noise floor ---------------------------------------------------------
    floor = filters.noise_floor(curated)
    (OUT / "noise_floor.json").write_text(json.dumps(floor, indent=2))
    print("\n--- Assay noise floor ---")
    print(json.dumps(floor, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
