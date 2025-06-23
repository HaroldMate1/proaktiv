#!/usr/bin/env python3
"""
Script to predict IC50 (in nM) and pIC50 for protein–ligand pairs
using a fine-tuned ESM2 3B-parameter model + Morgan fingerprints,
with optional EGFR variant mutation testing.

Example single prediction:
    python predict_ic50.py \
      --model_path /scratch/s6273475/results/best_model_epoch35.pt \
      --smiles "CCO" \
      --sequence "MKT..."

Batch CSV prediction:
    python predict_ic50.py \
      --model_path /scratch/s6273475/results/best_model_epoch35.pt \
      --input_csv data/input_pairs.csv \
      --output_csv data/predictions.csv

EGFR mutation testing:
    python predict_ic50.py \
      --model_path /scratch/s6273475/results/best_model_epoch35.pt \
      --mutation_test \
      --data_excel /home4/s6273475/ml/master_project/data/EGFR_IC50_all_assays.xlsx
"""

import os
import re
import argparse
import logging
from tqdm.auto import tqdm
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from deepchem.feat import CircularFingerprint

# -------------------------------
# Configuration
# -------------------------------
DEFAULT_ESM = "facebook/esm2_t36_3B_UR50D"

# -------------------------------
# Mutation Testing on EGFR Variants with TKIs (Updated with MSE)
# -------------------------------

# Cache to store UniProt FASTA sequences for proteins to avoid repeated API requests (Faster processing)
fasta_cache = {}


# Function to fetch the UniProt ID for a given human protein name
def get_uniprot_id(protein_name):
    url = f"https://rest.uniprot.org/uniprotkb/search?query={protein_name}+AND+organism_id:9606&format=json"
    response = requests.get(url)
    if response.status_code == 200:
        try:
            data = response.json()
            if "results" in data and data["results"]:
                return data["results"][0]["primaryAccession"]
        except requests.exceptions.JSONDecodeError:
            print(f"Error decoding JSON response from UniProt API for {protein_name}")
    return None


# Function to fetch the FASTA sequence from UniProt, using a cache to avoid duplicate API calls
def get_cached_uniprot_fasta(protein_name):
    if protein_name in fasta_cache:
        return fasta_cache[protein_name], None
    uniprot_id = get_uniprot_id(protein_name)
    if not uniprot_id:
        return None, f"Error: Could not find UniProt ID for '{protein_name}'."
    url = f"https://www.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        fasta_sequence = "".join(response.text.split("\n")[1:])
        fasta_cache[protein_name] = fasta_sequence
        return fasta_sequence, None
    else:
        return (
            None,
            f"Error: Unable to retrieve FASTA sequence for UniProt ID {uniprot_id}.",
        )


# Dummy mutation resolver and extraction helpers for substitution only
def get_combined_transformation(mutation, protein_name):
    return mutation.strip()


# Function to extract deletion range from a mutation string
# RegEx to optionally match a letter before the number for both start and end.
# Example: "A123" or "123" or "123A" or "123-456" or "A123-A456" or "A123-456" or "123-A456"
def extract_deletion_range(mutation):
    match = re.search(r"[A-Z]?(\d+)[_-][A-Z]?(\d+)del", mutation, re.IGNORECASE)
    return (int(match.group(1)), int(match.group(2))) if match else (None, None)


# Function to extract insertion position and sequence from a mutation string
def extract_insertion_info(mutation):
    match = re.search(r"(\d+)ins([A-Z]+)", mutation, re.IGNORECASE)
    return (int(match.group(1)), match.group(2)) if match else (None, None)


# Function to extract delins information from a mutation string
def extract_delins_info(mutation):
    match = re.search(r"(\d+)[_-](\d+)delins([A-Z]+)", mutation, re.IGNORECASE)
    return (
        (int(match.group(1)), int(match.group(2)), match.group(3))
        if match
        else (None, None, None)
    )


# Function to apply a mutation to a given FASTA sequence
# The mutation name is expected to be a single mutation or a comma-separated list of mutations.
def apply_mutation(fasta, mutation_name, protein_name):
    if pd.isna(mutation_name):
        return fasta

    mutation_name = str(mutation_name).strip()

    # Catch both "Other Mutations" and variants of "Wild Type"
    if mutation_name.lower() in ["other mutations", "wild-type", "wild type"]:
        return fasta

    fasta_dict = {num: amino for num, amino in enumerate(fasta, start=1)}
    mutations = mutation_name.split(",")
    # Process each mutation in the list
    for mutation in mutations:
        mutation = mutation.strip()
        mutation_type = get_combined_transformation(mutation, protein_name)

        # Skip mutations classified as 'NA' or known membrane mutations.
        if mutation_type.strip().upper() == "MEMBRANE MUTATION" or mutation_type in [
            "EGFR dTCb mutation",
            "EGFR LTb mutation",
            "EGFR LTCb mutation",
        ]:
            continue

        print(f"Applying Mutation: {mutation} → {mutation_type}")

        # Process deletion and insertion mutations
        # Check for delins first, then deletion, then insertion
        if "delins" in mutation_type.lower():
            start, end, new_seq = extract_delins_info(mutation_type)
            if start is not None and end is not None and new_seq is not None:
                print(
                    f"Applying delins: deleting positions {start} to {end} and inserting {new_seq} at position {start}"
                )
                # Delete residues from start to end (inclusive)
                for pos in range(start, end + 1):
                    fasta_dict.pop(pos, None)
                # Insert the new sequence at the starting position
                fasta_dict[start] = new_seq
            continue
        # Process deletion mutations
        if "del" in mutation_type.lower():
            start, end = extract_deletion_range(mutation_type)
            if start is not None and end is not None:
                for pos in range(start, end + 1):
                    fasta_dict.pop(pos, None)
            continue

        if "ins" in mutation_type.lower():
            pos, inserted_seq = extract_insertion_info(mutation_type)
            if pos is not None and inserted_seq is not None:
                fasta_dict[pos] = inserted_seq + fasta_dict.get(pos, "")
            continue

        # Process substitution mutations (expected format: two, three and four digits mutations)
        # Example: A123B, 123A, 123-456, A123-A456, A123-456, 123-A456
        match = re.match(r"^([A-Z])(\d{2,4})([A-Z])$", mutation_type)
        if match:
            ref, pos_str, alt = match.groups()
            pos = int(pos_str)
            if fasta_dict.get(pos) == ref:
                fasta_dict[pos] = alt
            else:
                print(
                    f"Error: Expected {ref} at {pos}, found {fasta_dict.get(pos, 'N/A')}"
                )
        else:
            print(
                f"Mutation type '{mutation_type}' does not match expected substitution format. Skipping this mutation."
            )

    # Return the fully updated sequence after processing all mutations.
    return "".join([aa for pos, aa in sorted(fasta_dict.items())])


EGFR_mutations = [
    "A763_Y764insFQEA",
    "C797S",
    "C797S, E746_A750del",
    "C797S, E746_A750del, T790M",
    "C797S, T790M",
    "D761N",
    "D761Y",
    "D761Y, L858R",
    "D770_N771insG",
    "D770delinsGY",
    "E709K",
    "E709K, G719A",
    "E746_A750del",
    "E746_A750del, T790M",
    "E746_S752delinsV",
    "E746_S752delinsV, G724S",
    "G719A",
    "G719A, R776G",
    "G719A, S768I",
    "G719C",
    "G719C, S768I",
    "G719S",
    "G724S",
    "K754E",
    "K754E, L747_E749del",
    "L718Q",
    "L718Q, L858R",
    "L718V",
    "L718V, L858R",
    "L747P",
    "L747_A750delinsP",
    "L747_P753delinsS",
    "L747_T751del",
    "L792H",
    "L792H, L858R",
    "L792H, L858R, T790M",
    "L792H, T790M",
    "L858R",
    "L858R, S768I",
    "L858R, T790M",
    "L858R, V834L",
    "L858_A859delinsRS",
    "L861Q",
    "S768I",
    "S768I, V774M",
    "T725M",
    "T790M",
    "V774M",
]

TKI_DRUGS = {
    "First Generation": {
        "GEFITINIB": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1",
        "ERLOTINIB": "C#Cc1cccc(Nc2ncnc3cc(OCCOC)c(OCCOC)cc23)c1",
        "ICOTINIB": "C#Cc1cccc(Nc2ncnc3cc4c(cc23)OCCOCCOCCO4)c1",
    },
    "Second Generation": {
        "AFATINIB": "CN(C)C/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1O[C@H]1CCOC1",
        "DACOMITINIB ANHYDROUS": "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1NC(=O)/C=C/CN1CCCCC1",
    },
    "Third Generation": {
        "OSIMERTINIB": "C=CC(=O)Nc1cc(Nc2nccc(-c3cn(C)c4ccccc34)n2)c(OC)cc1N(C)CCN(C)C",
        "ROCILETINIB": "C=CC(=O)Nc1cccc(Nc2nc(Nc3ccc(N4CCN(C(C)=O)CC4)cc3OC)ncc2C(F)(F)F)c1",
        "NAZARTINIB": "Cc1cc(C(=O)Nc2nc3cccc(Cl)c3n2[C@@H]2CCCCN(C(=O)/C=C/CN(C)C)C2)ccn1",
    },
    "Fourth Generation": {
        "BLU-945": "CO[C@@H]1CCN(c2nccc(Nc3cc4c(C(C)C)ccc(N5C[C@H](CS(C)(=O)=O)[C@H]5C)c4cn3)n2)C[C@@H]1F"
    },
}


# -------------------------------
# Model & Prediction Utilities
# -------------------------------
class ProteinLigandDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, fp_generator, max_length):
        self.sequences = df["variant_mutation_sequence"].tolist()
        fps = fp_generator.featurize(df["canonical_smiles"].tolist())
        self.fps = torch.tensor(np.vstack(fps), dtype=torch.float32)
        ic50_m = df["standard_value"].astype(float).values * 1e-9
        ic50_m = np.clip(ic50_m, 1e-12, None)
        self.labels = torch.tensor(-np.log10(ic50_m), dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        enc = self.tokenizer(
            seq,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "fingerprints": self.fps[idx],
            "labels": self.labels[idx],
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "fingerprints": torch.stack([b["fingerprints"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


class ProteinLigandModel(nn.Module):
    def __init__(self, esm_model, fp_bits, hidden_size, dropout):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(esm_model)
        self.encoder.gradient_checkpointing_enable()
        embed_dim = self.encoder.config.hidden_size
        self.fp_mlp = nn.Sequential(
            nn.Linear(fp_bits, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, input_ids, attention_mask, fps):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        vec = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        fpv = self.fp_mlp(fps)
        return self.regressor(torch.cat([vec, fpv], dim=1)).squeeze(-1)


def predict_with_uncertainty(
    model, tokenizer, fp_gen, smiles, seq, max_length, mc_samples, device
):
    fps = torch.tensor(np.vstack(fp_gen.featurize([smiles])), dtype=torch.float32).to(
        device
    )
    enc = tokenizer(
        seq,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    input_ids = enc["input_ids"].to(device)
    mask = enc["attention_mask"].to(device)
    model.train()
    preds = []
    for _ in range(mc_samples):
        with torch.no_grad():
            out = model(input_ids, mask, fps)
        preds.append(out.cpu().item())
    arr = np.array(preds)
    return arr.mean(), arr.std()


# -------------------------------
# Main CLI
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--esm_model", default=DEFAULT_ESM)
    parser.add_argument("--fp_bits", type=int, default=2048)
    parser.add_argument("--fp_radius", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--mc_samples", type=int, default=10)
    parser.add_argument("--device", default=None)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smiles")
    group.add_argument("--input_csv")
    group.add_argument("--mutation_test", action="store_true")
    parser.add_argument("--sequence")
    parser.add_argument("--output_csv")
    parser.add_argument("--data_excel")
    args = parser.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logging.info(f"Using ESM model: {args.esm_model}")

    tokenizer = AutoTokenizer.from_pretrained(args.esm_model, do_lower_case=False)
    fp_gen = CircularFingerprint(size=args.fp_bits, radius=args.fp_radius)
    model = ProteinLigandModel(
        args.esm_model, args.fp_bits, args.hidden_size, args.dropout
    )
    state = torch.load(args.model_path, map_location="cpu")
    # remap protein_encoder.* → encoder.*
    new_state = {}
    for key, val in state.items():
        if key.startswith("protein_encoder."):
            new_key = "encoder." + key[len("protein_encoder.") :]
        else:
            new_key = key
        new_state[new_key] = val
    state = new_state

    model.load_state_dict(state, strict=False)

    model.to(device)

    # Single or batch prediction
    if args.smiles:
        assert args.sequence, "--sequence required with --smiles"
        mean, std = predict_with_uncertainty(
            model,
            tokenizer,
            fp_gen,
            args.smiles,
            args.sequence,
            args.max_length,
            args.mc_samples,
            device,
        )
        ic50_m = 10 ** (-mean)
        ic50_nM = ic50_m * 1e9
        print(f"pIC50: {mean:.4f} ± {std:.4f}")
        print(f"IC50: {ic50_nM:.2f} nM")
        return

    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        preds = []
        for _, row in df.iterrows():
            m, s = predict_with_uncertainty(
                model,
                tokenizer,
                fp_gen,
                row["canonical_smiles"],
                row["variant_mutation_sequence"],
                args.max_length,
                args.mc_samples,
                device,
            )
            ic50_nM = (10 ** (-m)) * 1e9
            preds.append((m, s, ic50_nM))
        out = pd.DataFrame(preds, columns=["pIC50", "Uncertainty", "IC50_nM"])
        res = pd.concat([df.reset_index(drop=True), out], axis=1)
        if args.output_csv:
            res.to_csv(args.output_csv, index=False)
            print(f"Wrote to {args.output_csv}")
        else:
            print(res)
        return

    # Mutation testing
    assert args.mutation_test and args.data_excel, (
        "--data_excel required for mutation_test"
    )
    df_wt = pd.read_excel(args.data_excel)
    df_wt = df_wt[df_wt["assay_variant_mutation"].str.lower() == "wild type"]
    df_wt["pIC50"] = -np.log10(df_wt["standard_value"] * 1e-9)
    wt_stats = (
        df_wt.groupby("molecule_pref_name")
        .agg(
            wild_type_min=("pIC50", lambda x: np.percentile(x, 25)),
            wild_type_max=("pIC50", lambda x: np.percentile(x, 75)),
            wild_type_mean=("pIC50", "mean"),
        )
        .reset_index()
    )
    # Prepare variants
    fasta, err = get_cached_uniprot_fasta("EGFR")
    if err:
        raise ValueError(err)
    samples = []
    for gen, drugs in TKI_DRUGS.items():
        for drug, sm in drugs.items():
            for mut in EGFR_mutations:
                seq = apply_mutation(fasta, mut, "EGFR")
                samples.append(
                    {
                        "Generation": gen,
                        "Drug": drug,
                        "Mutation": mut,
                        "canonical_smiles": sm,
                        "sequence": seq,
                    }
                )
    df_mt = pd.DataFrame(samples).merge(
        wt_stats, left_on="Drug", right_on="molecule_pref_name", how="left"
    )
    # Predict
    results = []
    # wrap the iterator in tqdm, so you see a progress bar labeled "Inference"
    for _, r in tqdm(df_mt.iterrows(), total=len(df_mt), desc="Inference"):
        m, s = predict_with_uncertainty(
            model,
            tokenizer,
            fp_gen,
            r["canonical_smiles"],
            r["sequence"],
            args.max_length,
            args.mc_samples,
            device,
        )
        mse = (
            (m - r["wild_type_mean"]) ** 2
            if not np.isnan(r["wild_type_mean"])
            else np.nan
        )
        results.append((m, s, mse))

    df_mt[["Estimated_pIC50", "Precision", "MSE"]] = pd.DataFrame(results)

    # Sensitivity classification
    def classify(row):
        # no wild‐type reference?
        if np.isnan(row.wild_type_mean):
            return "No ref"
        # 25th pct, 75th pct, mean
        q1, q3, wm = row.wild_type_min, row.wild_type_max, row.wild_type_mean
        pv = row.Estimated_pIC50

        # within float‐tolerance of wild‐type mean
        if np.isclose(pv, wm):
            return "No Impact"
        # lower pIC₅₀ than wild‐type → resistant
        if pv < wm:
            return "Slightly Resistant" if pv > q1 else "Resistant"
        # higher pIC₅₀ → sensitive
        return "Slightly Sensitive" if pv < q3 else "Sensitive"

    # apply and write out
    df_mt["Sensitivity"] = df_mt.apply(classify, axis=1)
    out_path = os.path.splitext(args.model_path)[0] + "_mutation_testing.xlsx"
    df_mt.to_excel(out_path, index=False)
    print(f"Mutation testing results saved to {out_path}")


if __name__ == "__main__":
    main()
