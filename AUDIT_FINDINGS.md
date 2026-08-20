# PROAKTIV audit findings

Code- and data-level verification of the reviewer criticisms, run against commit
`662bc24` on 2026-08-20. Every claim below was reproduced from the repository and
`data/egfr_alk_braf_merged.xlsx`; the generating scripts are named for each item.

The purpose of this document is to separate what the reviewers got right from
what they inferred, and to give the point-by-point response letter (Phase 10 of
[`REVISION_PLAN.md`](REVISION_PLAN.md)) a factual basis. Several findings are
worse than the reviewers could have known from the manuscript alone.

---

## 1. ALK variants were invisible to the ESM2 model

**Status: confirmed, more severe than reported.** Supports Reviewer 2, point 2.

`src/training/fingerprints_plm.py` tokenizes with `max_length=1024,
truncation=True` (lines 71-73, default at line 673). Every ALK record in the
curated data carries the full 1,620 aa canonical sequence (UniProt Q9UM73), and
every ALK mutation lies beyond residue 1024:

| Variant | Position | Records | Inside 1024-token window? |
|---|---|---|---|
| T1151M | 1151 | 2 | no |
| L1152R | 1152 | 8 | no |
| C1156Y | 1156 | 44 | no |
| F1174L | 1174 | 42 | no |
| L1196M | 1196 | 524 | no |
| G1202R | 1202 | 120 | no |
| S1206Y | 1206 | 11 | no |
| G1269A | 1269 | 21 | no |
| R1275Q | 1275 | 5 | no |

All 10 ALK variant states therefore encoded to a single identical truncated
sequence. **777 ALK mutant records were indistinguishable from ALK wild type**
at the protein-encoder input. Any apparent ALK variant discrimination came from
the ligand branch alone.

EGFR (1,205-1,214 aa, mutations at 709-861) and BRAF (766 aa, mutations at
589-731) are unaffected — their mutated residues fall inside the window.

**Fix applied.** `src/proaktiv/curation/sequences.py` replaces truncation with an
explicit kinase-domain window (UniProt protein-kinase domain + 64 aa flank,
declared in `configs/kinases.yml`). The window is defined in wild-type
coordinates and transferred to variant sequences by preserving the untouched
prefix and suffix, which keeps indel-bearing variants in register and raises an
error rather than silently misaligning if a variant differs outside the window.

| Kinase | Window | Size | Distinct variants before | after |
|---|---|---|---|---|
| EGFR | 648-1043 | 396 aa | 18/19 | 18/19 |
| ALK | 1052-1456 | 405 aa | **1/10** | **10/10** |
| BRAF | 393-766 | 374 aa | 2/2 | 2/2 |

Verify with `python scripts/verify_windows.py`. The one EGFR collapse is a
single unparseable record (see finding 4), not a windowing failure.

Side benefit for the retrain: windows are 374-405 aa against the previous 1,022,
so ESM2 sequence length drops roughly 2.5x.

## 2. The random split leaks heavily, and the leakage is now quantified

**Status: confirmed.** Supports Reviewer 1 and Reviewer 2, point 1.

`src/training/fingerprints_plm.py:149-151` uses `train_test_split` on rows;
`src/training/fingerprints_cnn_rnn.py:77-79` uses DeepPurpose
`split_method="random"`. Measured overlap between train and test under each
scheme, as a percentage of held-out groups that also appear in training
(`results/splits/leakage_audit.csv`):

| Grouping | random (submitted) | scaffold | unseen-variant | combined |
|---|---|---|---|---|
| Compound identity | 35.0% | 0% | 81.1% | 0% |
| Bemis-Murcko scaffold | 66.9% | 0% | 80.1% | 0% |
| Variant identity | 100% | 100% | 0% | 0% |

Under the submitted scheme **every variant in the test set was also in
training**, two thirds of test scaffolds were seen, and a third of test
compounds were seen outright. Reviewer 1's phrasing — "random validation is
known to overestimate performance, also for the model presented here" — is
correct as stated.

**Fix applied.** `src/proaktiv/splits/strategies.py` adds scaffold,
unseen-variant, combined, leave-one-mutation-out and temporal schemes.
`src/proaktiv/splits/audit.py` fails any manifest whose claimed isolation does
not hold. Manifests are frozen under `results/splits/`. Regenerate with
`python scripts/make_splits.py`.

Note the cost of honesty: the combined scaffold-and-variant split leaves 263
test groups out of 18,492, with 6,707 rows unassignable without violating one of
the two guarantees. That is a real constraint on what the hardest split can
support, and it should be stated rather than worked around.

## 3. No relation or unit filtering — fabricated labels entered training

**Status: confirmed, not raised by either reviewer.**

`src/data_processing/dataset.py` retains `standard_relation` and
`standard_units` as columns but never filters on them (the only filter is
`potential_duplicate == 0` at line 176, which is a no-op: the flag is zero for
all 25,362 rows). Both training scripts then compute `-log10(value * 1e-9)`
over whatever they are given:

- **4,541 records (17.9%) are censored** (`>`, `<`, `>=`, `<=`, `>>`, `~`). A
  `>10000 nM` non-binder becomes an exact pIC50 label of 5.0. This creates an
  artificial spike of identical labels at the assay detection limit and rewards
  a model for predicting it.
- **105 records are not in nanomolar** (99 in `ug.mL-1`, 6 in `/uM`) and were
  converted as if they were.
- 9 records fall outside a plausible pIC50 range of 2-12.

**Fix applied.** Eligibility rules are declared in `configs/curation.yml` before
any model is fit, and `src/proaktiv/curation/filters.py` writes every excluded
row to `results/curation/exclusions.csv` with its reason. 20,733 of 25,362 raw
records survive.

## 4. One unresolved mutation label was treated as a distinct variant

**Status: confirmed, minor.**

A single EGFR record (`CHEMBL2105712`, assay description
`Selectivity interaction (Enzymatic activity assay) EUB0000596aBDA EGFR`) is
labelled `Other Mutation` and carries the **wild-type** sequence. It entered the
data as if it were its own variant state. Now quarantined by the
`mutation_labels.quarantine` rule.

## 5. ALK and BRAF rare-mutation patterns are dead code

**Status: confirmed.** Supports Reviewer 2, point 2.

`src/data_processing/dataset.py:424-466` defines `mutations_dict` with EGFR, ALK
and BRAF entries. Line 469 then builds the pattern from `mutations_dict["EGFR"]`
only:

```python
mutation_patterns = r"(?:" + "|".join(mutations_dict["EGFR"]) + r")"
```

The ALK and BRAF patterns are never compiled and never matched. Likewise
`mutation_general_map_1` (line 485) has only an `EGFR` key, so
`get_map1_transformation` is a pass-through for ALK and BRAF records.
Reviewer 2's observation that "ALK and BRAF are much less developed" understates
this: the rare-mutation curation path never ran for them at all.

## 6. Assay classes were pooled, and the pooling is measurable

**Status: confirmed.** Supports Reviewer 2, point 3.

Biochemical and cell-based IC50 were merged into one label column. Median pIC50
by class (`results/curation/assay_heterogeneity.csv`):

| Kinase | Biochemical | Cell-based | Gap |
|---|---|---|---|
| EGFR | 7.08 | 6.52 | 0.56 |
| ALK | 7.72 | 7.19 | 0.53 |
| BRAF | 7.72 | 6.55 | **1.17** |

The gap is systematic and, for BRAF, larger than a log unit. Pooling injects
this as label noise. `assay_class` is now part of the replicate key, so the two
classes no longer merge, and it is available as a stratification variable.

## 7. There is a measurable assay noise floor

**Status: new, not raised by either reviewer, and useful to us.**

Within replicate groups — the same compound, target, variant and assay class —
measured pIC50 disagrees substantially (`results/curation/noise_floor.json`):

- 1,421 groups have more than one measurement
- median within-group SD **0.40 pIC50**, mean 0.55, 90th percentile 1.28
- 246 groups disagree by more than 1 full log unit
- implied **RMSE floor 0.78 pIC50**

This is the irreducible error: no model can predict a label more precisely than
the label is measured. It gives us two things for the response letter. First, a
principled reference against which to report RMSE, which answers Reviewer 1's
demand for error metrics with a scale rather than a bare number. Second, a
leakage detector — a reported test RMSE far below this floor is memorisation of
duplicated measurements, not accuracy.

Note that pooling assay classes inflated this figure to 0.567 median SD;
stratifying dropped it to 0.40, which is independent corroboration of finding 6.

## 8. Figure 4 compares predictions to wild-type measurements, not to mutant data

**Status: confirmed. Reviewer 1 was right, and the mechanism is specific.**

Reviewer 1 wrote: *"unsure where predictions are compared to experimental data,
is this really in the plot? I cannot see this, apparently this shows only
predictions."*

In `src/inference/plm_prediction.py`:

- `wt_stats` (line 462) aggregates **experimental** pIC50 for **wild type** only.
- Line 507 computes `(m - r["wild_type_mean"]) ** 2`, where `m` is the
  **predicted** pIC50 for a **mutant**. This is the squared difference between a
  mutant prediction and a wild-type measurement — two different biological
  entities. It is neither a prediction error nor a resistance shift.
- Line 513 stores that quantity in a column named **`MSE`**.
- `classify` (line 516 onward) assigns resistance/sensitivity by asking whether
  the predicted mutant pIC50 falls inside the wild-type experimental
  interquartile range.

No measured mutant bioactivity enters this analysis anywhere. The
resistance/sensitivity calls, and the column a reader would naturally read as
prediction error, are both derived without a single mutant measurement. The
claim that predictions "align with established resistance mechanisms" cannot be
supported from this figure.

The curated data does contain mutant measurements (3,018 EGFR mutant replicate
groups, 709 ALK, 3,267 BRAF), so a genuine measured-versus-predicted case study
is buildable. That is Phase 6 work.

## 9. Uncertainty: labelling is wrong and the model never returns to eval mode

**Status: confirmed.** Supports Reviewer 1's Figures 2C/3C comments and
Reviewer 2, point 5.

- `predict_with_uncertainty` (`plm_prediction.py:352`) calls `model.train()` and
  **never restores `model.eval()`** — the string `model.eval()` does not appear
  in the file. Dropout is intentionally active for MC sampling, but the model is
  left in training mode for every subsequent call, so all reported point
  estimates are dropout-perturbed sample means rather than deterministic
  forward passes. This is defensible as MC dropout but is undocumented.
- `mc_samples` defaults to **10** (line 375), which is too few for a stable
  variance estimate and is never justified.
- The MC-dropout standard deviation is stored under the column name
  **`Precision`** (line 513). A standard deviation is a dispersion, not a
  precision.
- Two different quantities are both labelled `MSE`: the per-epoch training table
  header (`fingerprints_plm.py:318`) says `MSE` but is passed `val_rmse`
  (line 313), a root-mean-square error; the figure titles at lines 452 and 490
  use a genuine `mean_squared_error`. A reader comparing the table to the
  figures is comparing different quantities under one name.

Separately, `np.sqrt(np.mean(epoch_val))` averages per-batch losses before
taking the root, which is a slightly biased RMSE when the final batch is
smaller than the rest.

No coefficient, calibration metric, confidence interval or high-error detection
statistic is computed anywhere for the uncertainty–error relationship. The
manuscript's claim that uncertainty "tracked absolute error" is therefore
unsupported by any reported quantity, which is exactly what Reviewer 1 observed
from the figures.

## 10. Findings that were *not* confirmed

Recorded so the response letter does not concede more than the evidence
requires.

- **Mutation sequence construction is correct.** Spot-checking every variant
  state against canonical UniProt numbering, all substitutions are applied at
  the right residue (EGFR T790M, C797S, L858R, L861Q; ALK L1196M, G1202R,
  G1269A and the rest; BRAF V600E all verified). Deletion boundaries are correct
  too: `E746_A750del` removes exactly residues 746-750, and `E709_T710del`
  removes exactly 709-710. Substitutions co-occurring with deletions are applied
  in canonical coordinates before the deletion shifts them, so the composite
  variants are right.
- **`REVISION_PLAN.md` item 8** anticipated duplicated early-stopping and
  checkpoint logic in the ESM2 script. There are two separate training paths
  (a main loop and an Optuna objective) that each implement their own; this is
  duplication worth refactoring but not a correctness bug.

---

## Reproducing

```bash
python scripts/verify_windows.py    # finding 1
python scripts/build_dataset.py     # findings 3, 4, 6, 7 and Table 1
python scripts/make_splits.py       # finding 2
python -m pytest tests_unit -q      # 33 tests
```
