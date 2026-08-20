"""Reference baselines across every split scheme.

Phase 3 of REVISION_PLAN.md requires simple baselines on the frozen splits, and
Phase 4 requires RMSE-first reporting. These models are deliberately cheap and
CPU-only; their purpose is to establish the floor a deep model must clear and to
show how much of the submitted performance was split-dependent.

Baselines:
  global_mean   train-set mean pIC50
  kinase_mean   per-kinase mean
  variant_mean  per-(kinase, variant) mean -- no ligand information at all
  ligand_rf     Morgan fingerprint -> random forest, ligand only
  ligand_variant_rf  Morgan fingerprint + one-hot variant identity
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import RandomForestRegressor

RDLogger.DisableLog("rdApp.*")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CURATED = ROOT / "results" / "curation" / "curated.parquet"
SPLITS = ROOT / "results" / "splits"
OUT = ROOT / "results" / "baselines"
SEED = 1
N_BOOT = 1000


def morgan_matrix(smiles: pd.Series, bits: int = 2048, radius: int = 2) -> np.ndarray:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=bits)
    cache: dict[str, np.ndarray] = {}
    rows = []
    for smi in smiles:
        if smi not in cache:
            mol = Chem.MolFromSmiles(smi)
            fp = np.zeros(bits, dtype=np.uint8)
            if mol is not None:
                fp = np.frombuffer(
                    gen.GetFingerprintAsNumPy(mol).tobytes(), dtype=np.uint8
                )
            cache[smi] = fp
        rows.append(cache[smi])
    return np.vstack(rows)


def metrics(y_true: np.ndarray, y_pred: np.ndarray, rng: np.random.Generator) -> dict:
    """RMSE/MAE/R2 with bootstrap 95% CIs; correlation reported as supplementary."""
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    boot = []
    idx = np.arange(len(y_true))
    for _ in range(N_BOOT):
        pick = rng.choice(idx, size=len(idx), replace=True)
        boot.append(np.sqrt(np.mean(err[pick] ** 2)))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "rmse": round(rmse, 3),
        "rmse_ci_low": round(float(lo), 3),
        "rmse_ci_high": round(float(hi), 3),
        "mae": round(float(np.mean(np.abs(err))), 3),
        "r2": round(1 - ss_res / ss_tot, 3) if ss_tot > 0 else float("nan"),
        "pearson_r": round(float(np.corrcoef(y_true, y_pred)[0, 1]), 3)
        if len(y_true) > 1 and y_pred.std() > 0 else float("nan"),
        "n_test": int(len(y_true)),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(CURATED).reset_index(drop=True)
    fps = morgan_matrix(df["canonical_smiles"])
    variant_key = df["target_pref_name"] + "|" + df["assay_variant_mutation"].astype(str)
    variant_onehot = pd.get_dummies(variant_key).to_numpy(dtype=np.uint8)
    y = df["pic50"].to_numpy()

    records = []
    for scheme in ["random", "scaffold", "variant", "combined"]:
        manifest = pd.read_csv(SPLITS / f"{scheme}.csv")["split"].to_numpy()
        tr = manifest == "train"
        te = manifest == "test"
        if te.sum() == 0:
            continue
        rng = np.random.default_rng(SEED)

        preds = {"global_mean": np.full(te.sum(), y[tr].mean())}

        km = pd.Series(y[tr]).groupby(df.loc[tr, "target_pref_name"].values).mean()
        preds["kinase_mean"] = (
            df.loc[te, "target_pref_name"].map(km).fillna(y[tr].mean()).to_numpy()
        )

        vm = pd.Series(y[tr]).groupby(variant_key[tr].values).mean()
        preds["variant_mean"] = (
            variant_key[te].map(vm).fillna(y[tr].mean()).to_numpy()
        )

        rf = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, n_jobs=1, random_state=SEED
        )
        rf.fit(fps[tr], y[tr])
        preds["ligand_rf"] = rf.predict(fps[te])

        combo = np.hstack([fps, variant_onehot])
        rf2 = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, n_jobs=1, random_state=SEED
        )
        rf2.fit(combo[tr], y[tr])
        preds["ligand_variant_rf"] = rf2.predict(combo[te])

        for model, pred in preds.items():
            records.append({"scheme": scheme, "model": model, **metrics(y[te], pred, rng)})
            print(f"  {scheme:9} {model:18} RMSE {records[-1]['rmse']:.3f} "
                  f"[{records[-1]['rmse_ci_low']:.3f}, {records[-1]['rmse_ci_high']:.3f}]  "
                  f"r={records[-1]['pearson_r']}")

    results = pd.DataFrame(records)
    results.to_csv(OUT / "baseline_metrics.csv", index=False)

    print("\n--- RMSE by split scheme (pIC50 units) ---")
    print(results.pivot(index="model", columns="scheme", values="rmse").to_string())
    print("\n--- Pearson r by split scheme (what the manuscript reported) ---")
    print(results.pivot(index="model", columns="scheme", values="pearson_r").to_string())

    floor = json.loads((ROOT / "results" / "curation" / "noise_floor.json").read_text())
    print(f"\nAssay noise floor (replicate RMSE): {floor['implied_rmse_floor']:.3f} pIC50")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
