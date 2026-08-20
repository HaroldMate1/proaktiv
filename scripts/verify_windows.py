"""Verify kinase-domain windowing against the curated data.

Checks that (a) every variant sequence can be windowed without losing content,
(b) every mutated residue survives the window, and (c) distinct variants remain
distinguishable after windowing -- the property that full-length truncation
destroyed for ALK.
"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from proaktiv.curation.sequences import (  # noqa: E402
    SequenceWindowError,
    apply_window,
    load_windows,
    window_by_target_name,
)

POSITION = re.compile(r"(\d+)")
DATA = Path(__file__).resolve().parents[1] / "data" / "egfr_alk_braf_merged.xlsx"


def positions(label: str) -> list[int]:
    if label.strip().lower().startswith("wild"):
        return []
    return [int(m) for m in POSITION.findall(label)]


def main() -> int:
    df = pd.read_excel(DATA)
    by_name = window_by_target_name(load_windows())
    failures = 0

    for target, group in df.groupby("target_pref_name"):
        window = by_name[target]
        wild_type = group.loc[
            group["assay_variant_mutation"] == "Wild Type", "variant_mutation_sequence"
        ].iloc[0]

        windowed: dict[str, str] = {}
        for label, sub in group.groupby("assay_variant_mutation"):
            sequence = sub["variant_mutation_sequence"].iloc[0]
            try:
                cut = apply_window(window, sequence, wild_type)
            except SequenceWindowError as exc:
                print(f"  FAIL {window.name} {label!r}: {exc}")
                failures += 1
                continue
            uncovered = [p for p in positions(label) if not window.covers(p)]
            if uncovered:
                print(f"  FAIL {window.name} {label!r}: positions {uncovered} outside window")
                failures += 1
            windowed[label] = cut

        collapsed = len(windowed) - len(set(windowed.values()))
        # Compare against what the submitted pipeline produced: truncate at 1022.
        truncated = {lbl: sub["variant_mutation_sequence"].iloc[0][:1022]
                     for lbl, sub in group.groupby("assay_variant_mutation")}
        collapsed_before = len(truncated) - len(set(truncated.values()))
        rows_before = sum(
            len(sub) for lbl, sub in group.groupby("assay_variant_mutation")
            if lbl != "Wild Type" and truncated[lbl] == truncated.get("Wild Type")
        )

        print(
            f"{window.name:5} variants={len(windowed):3} "
            f"window={window.start}-{window.end} ({window.size} aa) | "
            f"distinct after windowing: {len(set(windowed.values()))}/{len(windowed)} "
            f"(collapsed {collapsed}) | under old truncation: "
            f"{len(set(truncated.values()))}/{len(truncated)} (collapsed {collapsed_before}, "
            f"{rows_before} mutant rows indistinguishable from WT)"
        )

    print("\nOK" if failures == 0 else f"\n{failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
