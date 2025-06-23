import os

# Set CUDA architecture list to include NVIDIA L40S (compute capability 8.9)
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

from DeepPurpose import utils, dataset
from DeepPurpose import DTI as models
import warnings
import pandas as pd
import matplotlib
import torch
import sys
import numpy as np
import logging
import requests
import re
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Use non-interactive backend for command-line usage.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

# Check if GPU is available; if not, exit the program.
if not torch.cuda.is_available():
    raise RuntimeError(
        "GPU not available. This script requires a GPU to run. Please run on a GPU-enabled machine."
    )

print("GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------
file_path = "/home4/s6273475/ml/master_project/data/egfr_alk_braf_merged.xlsx"
data = pd.read_excel(file_path, engine="openpyxl")

# Rename columns to the expected names by DeepPurpose.
data.rename(
    columns={
        "canonical_smiles": "smiles",
        "variant_mutation_sequence": "sequence",
        "standard_value": "Label",
    },
    inplace=True,
)

# Convert raw IC50 (in nM) to pIC50.
data["Label"] = -np.log10(data["Label"] * 1e-9)

# Extract the relevant columns.
X_drugs = data["smiles"].tolist()  # Drug representations (SMILES)
X_targets = data["sequence"].tolist()  # Protein sequences
y = data["Label"].tolist()  # pIC50 values

print("Drug 1:", X_drugs[0])
print("Target 1:", X_targets[0])
print("pIC50 1:", y[0])
print("Total pairs:", len(y))

# -------------------------------
# Data Splitting
# -------------------------------
# Set encoding for drug and target:
#   - Drug: Morgan fingerprint encoding
#   - Target: CNN_RNN encoding
drug_encoding, target_encoding = "Morgan", "CNN_RNN"

train, val, test = utils.data_process(
    X_drugs,
    X_targets,
    y,
    drug_encoding,
    target_encoding,
    split_method="random",
    frac=[0.70, 0.10, 0.20],
    random_seed=1,
)

print(train.head(1))

# -------------------------------
# Model Configuration for Morgan (drug) and CNN_RNN (target)
# -------------------------------
config = {
    "drug_encoding": drug_encoding,
    "target_encoding": target_encoding,
    # Parameters for Morgan encoding (for drugs)
    "morgan_fp_length": 1024,  # Fingerprint length; adjust as needed
    "morgan_radius": 2,  # Radius for fingerprint calculation
    # Keys required for the MLP in the drug encoder.
    "input_dim_drug": 1024,  # Set equal to morgan_fp_length
    "hidden_dim_drug": 256,
    "mlp_hidden_dims_drug": [256, 128],
    # Parameters for protein encoder (CNN+RNN on sequences)
    "cnn_target_filters": [32, 64, 96],
    "cnn_target_kernels": [4, 8, 12],
    "hidden_dim_protein": 256,
    "rnn_target_hid_dim": 64,
    "rnn_target_n_layers": 2,
    "rnn_target_bidirectional": True,
    "rnn_Use_GRU_LSTM_target": "LSTM",
    # Key for classifier/regression head
    "cls_hidden_dims": [512, 128],
    # Training parameters
    "train_epoch": 30,
    "LR": 0.001,
    "batch_size": 128,
    "use_cuda": True,
    # New key for result folder
    "result_folder": "/home4/s6273475/ml/master_project/models/morgan_cnn_rnn",
}

# -------------------------------
# Model Initialization and Multi-GPU Setup
# -------------------------------
model = models.model_initialize(**config)
print(model)

# Check if multiple GPUs are available; if so, wrap the model for DataParallel training.
if config["use_cuda"] and torch.cuda.device_count() > 1:
    print(
        f"Multiple GPUs detected: {torch.cuda.device_count()}. Using DataParallel for multi-GPU training."
    )
    model.model = torch.nn.DataParallel(model.model)

try:
    history = model.train(train, val, test)
except RuntimeError as e:
    if "no kernel image is available" in str(e):
        print("Encountered CUDA kernel image error. Falling back to CPU training.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        config["use_cuda"] = False
        model = models.model_initialize(**config)
        history = model.train(train, val, test)
    else:
        raise

if history is not None:
    train_losses = history.get("train_losses", [])
    val_losses = history.get("val_losses", [])
else:
    train_losses = []
    val_losses = []
    logging.warning("Training history is not available.")

# -------------------------------
# Model Saving
# -------------------------------
model_to_save = model.module if hasattr(model, "module") else model
model_save_path = os.path.join(config["result_folder"], "trained_model.pth")
torch.save(model_to_save, model_save_path)
print(f"Model saved to {model_save_path}")


# -------------------------------
# Uncertainty Estimation Function
# -------------------------------
def predict_with_uncertainty(model, data, n_iter=10):
    model.model.train()
    preds = []
    for _ in range(n_iter):
        pred = model.predict(data)
        preds.append(pred)
    preds = np.array(preds)
    pred_mean = preds.mean(axis=0)
    pred_std = preds.std(axis=0)
    return pred_mean, pred_std


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
    return "".join(fasta_dict.values()).strip()


# Define the list of EGFR mutations to test
BRAF_mutations = [
    "V600E",
    "V600K",
    "V600R",
    "G469A",
    "G469V",
    "G466E",
    "G466R",
    "D594G",
    "D594N",
    "L597R",
    "L597Q",
    "K601E",
    "K601Q",
]

# -------------------------------
# 1. Compute Wild Type pIC50 Range Per Drug from the Dataset
# -------------------------------
# Reload the original dataset to capture all relevant columns (including 'molecule_pref_name' and 'canonical_smiles')
wt_data = pd.read_excel(file_path, engine="openpyxl")

# Filter rows where the assay_variant_mutation is 'wild type' (case-insensitive)
wt_data = wt_data[wt_data["assay_variant_mutation"].str.lower() == "wild type"].copy()

# Convert standard_value (in nM) to pIC50
wt_data["pIC50"] = -np.log10(wt_data["standard_value"] * 1e-9)

# Group by drug name (molecule_pref_name) to compute wild type statistics and retrieve the canonical SMILES.
wt_stats = (
    wt_data.groupby("molecule_pref_name")
    .agg(
        wild_type_min=("pIC50", lambda x: np.percentile(x, 25)),
        wild_type_max=("pIC50", lambda x: np.percentile(x, 75)),
        wild_type_mean=("pIC50", "mean"),
        canonical_smiles=("canonical_smiles", "first"),
    )
    .reset_index()
)

print("Wild Type pIC50 range per drug:")
print(wt_stats)

# -------------------------------
# 2. Define the TKIs for Different Generations
# -------------------------------
# (The placeholders for SMILES will be replaced via the merge with wild type stats below.)
tki_drugs = {
    "First Generation": {
        "Sorafenib": "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1"
    },
    "Second Generation": {
        "Vemurafenib": "CCCS(=O)(=O)Nc1ccc(F)c(C(=O)c2c[nH]c3ncc(-c4ccc(Cl)cc4)cc23)c1F",
        "Dabrafenib": "CC(C)(C)c1nc(-c2cccc(NS(=O)(=O)c3c(F)cccc3F)c2F)c(-c2ccnc(N)n2)s1",
    },
    "Third Generation": {
        "Encorafenib": "COC(=O)N[C@@H](C)CNc1nccc(-c2cn(C(C)C)nc2-c2cc(Cl)cc(NS(C)(=O)=O)c2F)n1"
    },
}


# -------------------------------
# 3. Prepare Test Data for Each (Drug, Mutation) Pair
# -------------------------------
# Prepare test data
test_samples = []
wt_fasta, err = get_cached_uniprot_fasta("BRAF")
if err:
    raise ValueError(err)

for gen, drugs in tki_drugs.items():
    for drug_name, drug_smiles in drugs.items():
        for mut in BRAF_mutations:
            mutated_seq = apply_mutation(wt_fasta, mut, "BRAF")
            test_samples.append(
                {
                    "Generation": gen,
                    "Drug": drug_name,
                    "Mutation": mut,
                    "smiles": drug_smiles,
                    "sequence": mutated_seq,
                }
            )

mutation_test_df = pd.DataFrame(test_samples)
print(f"Prepared {len(mutation_test_df)} mutation test samples.")

mutation_test_df = mutation_test_df.merge(
    wt_stats[
        [
            "molecule_pref_name",
            "wild_type_min",
            "wild_type_max",
            "wild_type_mean",
            "canonical_smiles",
        ]
    ],
    left_on="Drug",
    right_on="molecule_pref_name",
    how="left",
)
mutation_test_df["smiles"] = mutation_test_df["canonical_smiles"].combine_first(
    mutation_test_df["smiles"]
)

# -------------------------------
# Prepare Encodings for Mutation Test Samples
# -------------------------------
# Use the same encoding settings used in training:
drug_encoding, target_encoding = "Morgan", "CNN_RNN"

# Extract the SMILES and sequence values from the mutation_test_df.
mutation_smiles = mutation_test_df["smiles"].tolist()
mutation_sequences = mutation_test_df["sequence"].tolist()
dummy_labels = [0.0] * len(
    mutation_test_df
)  # Dummy labels, as real labels are not used during inference

# Process these samples using DeepPurpose's data_process function.
# We use frac=[1.0, 0.0, 0.0] so that all samples are kept.
processed_mutation, _, _ = utils.data_process(
    mutation_smiles,
    mutation_sequences,
    dummy_labels,
    drug_encoding,
    target_encoding,
    split_method="random",
    frac=[1.0, 0.0, 0.0],
    random_seed=1,
)

# Append the computed encoding columns to mutation_test_df.
mutation_test_df["drug_encoding"] = processed_mutation["drug_encoding"].tolist()
mutation_test_df["target_encoding"] = processed_mutation["target_encoding"].tolist()
mutation_test_df["Label"] = dummy_labels  # Ensure the dummy Label column exists.

# -------------------------------
# 4. Predict pIC50 with Uncertainty for Each Sample using Drug-Specific Wild Type Values
# -------------------------------
estimated_pIC50_list = []
precision_list = []
mse_list = []  # MSE computed per sample using the drug-specific wild type mean
sensitivity_list = []

for index, row in mutation_test_df.iterrows():
    # Create a one-row DataFrame containing all required columns.
    sample_df = mutation_test_df.iloc[[index]].reset_index(drop=True)

    # Perform prediction with uncertainty estimation (n_iter=10)
    pred_mean, pred_std = predict_with_uncertainty(model, sample_df, n_iter=10)
    pred_value = pred_mean[0]  # scalar predicted pIC50
    uncertainty = pred_std[0]

    # Compute MSE for this prediction using the drug-specific wild type mean (if available)
    if pd.notnull(row["wild_type_mean"]):
        mse = (pred_value - row["wild_type_mean"]) ** 2
    else:
        mse = np.nan

    # Classify sensitivity based on the drug's wild type range, if available.
    # --- new classification logic ---
    if pd.notnull(row["wild_type_mean"]):
        wt_mean = row["wild_type_mean"]
        q1 = row["wild_type_min"]
        q3 = row["wild_type_max"]
        pred = pred_value

        # tolerance‐based equality
        if np.isclose(pred, wt_mean, rtol=1e-5, atol=1e-8):
            sensitivity = "No Impact"
        # lower pIC50 ⇒ resistant
        elif pred < wt_mean:
            sensitivity = "Slightly Resistant" if pred > q1 else "Resistant"
        # higher pIC50 ⇒ sensitive
        else:
            sensitivity = "Slightly Sensitive" if pred < q3 else "Sensitive"
    else:
        sensitivity = "No wild type reference"

    estimated_pIC50_list.append(pred_value)
    precision_list.append(uncertainty)
    mse_list.append(mse)
    sensitivity_list.append(sensitivity)

# Add computed metrics to the DataFrame.
mutation_test_df["Estimated_pIC50"] = estimated_pIC50_list
mutation_test_df["Precision"] = precision_list
mutation_test_df["MSE"] = mse_list
mutation_test_df["Sensitivity"] = sensitivity_list

# -------------------------------
# 5. Export the Results to an Excel File
# -------------------------------
output_excel_path = os.path.join(
    config["result_folder"], "BRAF_mutation_testing_results_q13.xlsx"
)
mutation_test_df.to_excel(output_excel_path, index=False)
print(f"Mutation testing results saved to {output_excel_path}")
