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
file_path = "/home4/s6273475/ml/master_project/data/BRAF_IC50_all_assays.xlsx"
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
    frac=[0.7, 0.1, 0.2],
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
# Evaluation on Validation and Test Sets
# -------------------------------
val_actual = np.array(val["Label"])
val_pred, val_uncertainty = predict_with_uncertainty(model, val, n_iter=10)
val_mse = mean_squared_error(val_actual, val_pred)
val_r2 = r2_score(val_actual, val_pred)
val_mae = mean_absolute_error(val_actual, val_pred)
val_errors = np.abs(val_actual - val_pred)

test_actual = np.array(test["Label"])
test_pred, test_uncertainty = predict_with_uncertainty(model, test, n_iter=10)
test_mse = mean_squared_error(test_actual, test_pred)
test_r2 = r2_score(test_actual, test_pred)  # Make sure this line executes successfully.
test_mae = mean_absolute_error(test_actual, test_pred)
test_errors = np.abs(test_actual - test_pred)

# -------------------------------
# Plotting: Training vs. Validation Loss
# -------------------------------
if train_losses and val_losses:
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Training vs. Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    loss_plot_path = os.path.join(
        config["result_folder"], "training_vs_validation_loss.png"
    )
    plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Training vs. Validation Loss plot saved as '{loss_plot_path}'")
else:
    logging.warning("No training/validation loss history available for plotting.")

# -------------------------------
# Plotting: Validation Inference Results
# -------------------------------
plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.scatter(val_actual, val_pred, alpha=0.5)
plt.plot([min(val_actual), max(val_actual)], [min(val_actual), max(val_actual)], "r--")
plt.xlabel("Actual pIC50 (standardized)")
plt.ylabel("Predicted pIC50 (standardized)")
plt.title(f"Validation: Predicted vs Actual\nMSE: {val_mse:.4f}, R²: {val_r2:.4f}")
plt.grid(True)

plt.subplot(1, 3, 2)
plt.hist(val_uncertainty, bins=50)
plt.xlabel("Prediction Uncertainty (σ)")
plt.ylabel("Count")
plt.title("Distribution of Prediction Uncertainties (Validation)")
plt.grid(True)

plt.subplot(1, 3, 3)
plt.scatter(val_uncertainty, val_errors, alpha=0.5)
plt.xlabel("Prediction Uncertainty (σ)")
plt.ylabel("Absolute Error")
plt.title("Validation: Error vs Uncertainty")
plt.grid(True)

plt.tight_layout()
val_plot_path = os.path.join(
    config["result_folder"], "validation_inference_results.png"
)
plt.savefig(val_plot_path, dpi=300, bbox_inches="tight")
plt.close()
logging.info(f"Validation inference results plot saved as '{val_plot_path}'")

# -------------------------------
# Plotting: Test Inference Results
# -------------------------------
plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.scatter(test_actual, test_pred, alpha=0.5)
plt.plot(
    [min(test_actual), max(test_actual)], [min(test_actual), max(test_actual)], "r--"
)
plt.xlabel("Actual pIC50 (standardized)")
plt.ylabel("Predicted pIC50 (standardized)")
plt.title(f"Test: Predicted vs Actual\nMSE: {test_mse:.4f}, R²: {test_r2:.4f}")
plt.grid(True)

plt.subplot(1, 3, 2)
plt.hist(test_uncertainty, bins=50)
plt.xlabel("Prediction Uncertainty (σ)")
plt.ylabel("Count")
plt.title("Distribution of Prediction Uncertainties (Test)")
plt.grid(True)

plt.subplot(1, 3, 3)
plt.scatter(test_uncertainty, test_errors, alpha=0.5)
plt.xlabel("Prediction Uncertainty (σ)")
plt.ylabel("Absolute Error")
plt.title("Test: Error vs Uncertainty")
plt.grid(True)

plt.tight_layout()
test_plot_path = os.path.join(config["result_folder"], "test_inference_results.png")
plt.savefig(test_plot_path, dpi=300, bbox_inches="tight")
plt.close()
logging.info(f"Test inference results plot saved as '{test_plot_path}'")

# Save any additional generated figures.
output_dir = config["result_folder"]
for fig_num in plt.get_fignums():
    plt.figure(fig_num)
    save_path = os.path.join(output_dir, f"figure_{fig_num}.png")
    plt.savefig(save_path)
    print(f"Figure {fig_num} saved to {save_path}")

# -------------------------------
# Print Summary Metrics
# -------------------------------
print("\nValidation Results:")
print(f"Number of samples: {len(val_pred)}")
print(f"Mean Squared Error: {val_mse:.4f}")
print(f"R² Score: {val_r2:.4f}")
print(f"Mean Absolute Error: {val_mae:.4f}")
print(f"Mean Uncertainty: {np.mean(val_uncertainty):.4f}")

print("\nTest Results:")
print(f"Number of samples: {len(test_pred)}")
print(f"Mean Squared Error: {test_mse:.4f}")
print(f"R² Score: {test_r2:.4f}")
print(f"Mean Absolute Error: {test_mae:.4f}")
print(f"Mean Uncertainty: {np.mean(test_uncertainty):.4f}")
