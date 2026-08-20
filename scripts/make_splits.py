"""Freeze split manifests and audit them for leakage.

Writes results/splits/<scheme>.csv (one row per replicate group with its
partition) and results/splits/leakage_audit.csv. The audit is the evidence for
the reviewers that the hard splits actually isolate what they claim to.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from proaktiv.splits import audit, strategies  # noqa: E402

CURATED = ROOT / "results" / "curation" / "curated.parquet"
OUT = ROOT / "results" / "splits"
SEED = 1


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(CURATED).reset_index(drop=True)
    # `document_chembl_id` is lost in aggregation; audit on what the table carries.
    columns = {k: v for k, v in audit.AUDIT_COLUMNS.items() if k in df.columns}

    manifests = {
        "random": strategies.random_split(df, SEED),
        "scaffold": strategies.scaffold_split(df, SEED),
        "variant": strategies.variant_split(df, SEED),
        "combined": strategies.combined_split(df, SEED),
    }

    reports, summary = [], []
    for name, manifest in manifests.items():
        df[f"split_{name}"] = manifest.assignment
        counts = manifest.counts()
        report = audit.overlap_report(df, manifest.assignment, columns)
        report.insert(0, "scheme", name)
        reports.append(report)

        passed, problems = audit.verdict(report, audit.REQUIREMENTS[name])
        summary.append({
            "scheme": name,
            "train": counts.get("train", 0),
            "valid": counts.get("valid", 0),
            "test": counts.get("test", 0),
            "unassigned": counts.get("unassigned", 0),
            "isolates": ", ".join(audit.REQUIREMENTS[name]) or "nothing (reference only)",
            "verdict": "PASS" if passed else "FAIL",
            "problems": "; ".join(problems),
        })

        manifest_df = df[[
            "molecule_chembl_id", "target_pref_name", "assay_variant_mutation",
            "assay_class", "scaffold", "pic50",
        ]].assign(split=manifest.assignment, scheme=name, seed=manifest.seed)
        manifest_df.to_csv(OUT / f"{name}.csv", index=False)

    leakage = pd.concat(reports, ignore_index=True)
    leakage.to_csv(OUT / "leakage_audit.csv", index=False)
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUT / "split_summary.csv", index=False)

    print("--- Split summary ---")
    print(summary_df.drop(columns=["problems"]).to_string(index=False))
    print("\n--- Leakage audit: groups shared between train and test ---")
    pivot = leakage[leakage["partition"] == "test"].pivot(
        index="grouping", columns="scheme", values="pct_shared"
    )
    print(pivot.to_string())
    failures = summary_df[summary_df["verdict"] == "FAIL"]
    if not failures.empty:
        print("\nFAILURES:")
        for _, row in failures.iterrows():
            print(f"  {row['scheme']}: {row['problems']}")
        return 1
    print("\nAll hard splits pass their isolation requirements.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
