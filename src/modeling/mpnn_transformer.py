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

# Print GPU availability and device name.
print("GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))

# Define the file path and load the Excel file using openpyxl as the engine.
file_path = "/home4/s6273475/ml/master_project/data/EGFR_IC50_all_assays.xlsx"
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
# pIC50 = -log10(IC50 in M) = -log10(IC50 * 1e-9)
data["Label"] = -np.log10(data["Label"] * 1e-9)

# Extract the relevant columns into lists.
X_drugs = data["smiles"].tolist()  # Drug representations (SMILES)
X_targets = data["sequence"].tolist()  # Protein sequences
y = data["Label"].tolist()  # pIC50 values

# Print sample entries for verification.
print("Drug 1: " + X_drugs[0])
print("Target 1: " + X_targets[0])
print("pIC50 1: " + str(y[0]))
print("Total pairs:", len(y))

# Define encoding methods:
# - Use MPNN for drug encoding with the new parameters.
# - Use Transformer for target encoding.
drug_encoding, target_encoding = "MPNN", "Transformer"

# Process the data and split into training, validation, and test sets.
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

# Quick look at the training data.
print(train.head(1))

# Updated configuration:
# - For drugs: use MPNN with mpnn_hidden_size=256 and mpnn_depth=3.
# - For target: use Transformer with example parameters.
config = utils.generate_config(
    drug_encoding="MPNN",  # MPNN for drug encoding
    target_encoding="Transformer",  # Transformer for protein encoding
    cls_hidden_dims=[1024, 1024, 512],
    train_epoch=5,
    LR=0.001,
    batch_size=128,
    hidden_dim_drug=128,
    mpnn_hidden_size=256,  # Updated hidden size for MPNN
    mpnn_depth=3,  # Specified MPNN depth
)

# If using a Transformer for the target encoder, manually add transformer parameters.
if target_encoding == "Transformer":
    config["transformer_layers"] = 2
    config["transformer_heads"] = 4
    config["transformer_hidden_dim"] = 128
    config["transformer_dropout"] = 0.1

# Enable GPU usage in the configuration.
config["use_cuda"] = True

# Initialize the model with the configuration.
model = models.model_initialize(**config)
print(model)

# Check if multiple GPUs are available and wrap the model with DataParallel if so.
if torch.cuda.device_count() > 1:
    print("Multiple GPUs detected: Using", torch.cuda.device_count(), "GPUs.")
    model = torch.nn.DataParallel(model)

# Train the model and capture loss history if available.
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

# Retrieve training and validation losses from history if provided.
if history is not None:
    train_losses = history.get("train_losses", [])
    val_losses = history.get("val_losses", [])
else:
    train_losses = []
    val_losses = []
    logging.warning("Training history is not available.")

# Save the entire model object to disk.
model_save_path = (
    "/home4/s6273475/ml/master_project/models/mpnn_transformer/trained_model.pth"
)
torch.save(model, model_save_path)
print(f"Model saved to {model_save_path}")


# Function to perform prediction with uncertainty estimation using Monte Carlo dropout.
def predict_with_uncertainty(model, data, n_iter=10):
    """
    Runs multiple forward passes through the model (with dropout active)
    to estimate prediction uncertainty.
    """
    # Ensure dropout layers are active
    if isinstance(model, torch.nn.DataParallel):
        model.module.model.train()  # Access the underlying model
    else:
        model.model.train()  # Set the model to training mode

    preds = []
    for _ in range(n_iter):
        pred = model.predict(data)  # Perform prediction
        preds.append(pred)
    preds = np.array(preds)  # Shape: (n_iter, num_samples)
    pred_mean = preds.mean(axis=0)
    pred_std = preds.std(axis=0)
    return pred_mean, pred_std


# -------------------------------
# Evaluate on the Validation Set.
# -------------------------------
val_actual = np.array(val["Label"])
val_pred, val_uncertainty = predict_with_uncertainty(model, val, n_iter=10)
val_mse = mean_squared_error(val_actual, val_pred)
val_r2 = r2_score(val_actual, val_pred)
val_mae = mean_absolute_error(val_actual, val_pred)
val_errors = np.abs(val_actual - val_pred)

# -------------------------------
# Evaluate on the Test Set.
# -------------------------------
test_actual = np.array(test["Label"])
test_pred, test_uncertainty = predict_with_uncertainty(model, test, n_iter=10)
test_mse = mean_squared_error(test_actual, test_pred)
test_r2 = r2_score(test_actual, test_pred)
test_mae = mean_absolute_error(test_actual, test_pred)
test_errors = np.abs(test_actual - test_pred)

# -------------------------------
# Plot Training vs. Validation Loss (if available).
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
        "/home4/s6273475/ml/master_project/models/mpnn_transformer",
        "training_vs_validation_loss.png",
    )
    plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Training vs. Validation Loss plot saved as '{loss_plot_path}'")
else:
    logging.warning("No training/validation loss history available for plotting.")

# -------------------------------
# Plotting Validation Results.
# -------------------------------
plt.figure(figsize=(18, 5))
# Scatter plot: Actual vs. Predicted (Validation)
plt.subplot(1, 3, 1)
plt.scatter(val_actual, val_pred, alpha=0.5)
plt.plot([min(val_actual), max(val_actual)], [min(val_actual), max(val_actual)], "r--")
plt.xlabel("Actual pIC50 (standardized)")
plt.ylabel("Predicted pIC50 (standardized)")
plt.title(f"Validation: Predicted vs Actual\nMSE: {val_mse:.4f}, R²: {val_r2:.4f}")
plt.grid(True)

# Histogram of prediction uncertainties (Validation)
plt.subplot(1, 3, 2)
plt.hist(val_uncertainty, bins=50)
plt.xlabel("Prediction Uncertainty (σ)")
plt.ylabel("Count")
plt.title("Distribution of Prediction Uncertainties (Validation)")
plt.grid(True)

# Scatter plot: Uncertainty vs. Absolute Error (Validation)
plt.subplot(1, 3, 3)
plt.scatter(val_uncertainty, val_errors, alpha=0.5)
plt.xlabel("Prediction Uncertainty (σ)")
plt.ylabel("Absolute Error")
plt.title("Validation: Error vs Uncertainty")
plt.grid(True)

plt.tight_layout()
val_plot_path = os.path.join(
    "/home4/s6273475/ml/master_project/models/mpnn_transformer",
    "validation_inference_results.png",
)
plt.savefig(val_plot_path, dpi=300, bbox_inches="tight")
plt.close()
logging.info(f"Validation inference results plot saved as '{val_plot_path}'")

# -------------------------------
# Plotting Test Results.
# -------------------------------
plt.figure(figsize=(18, 5))
# Scatter plot: Actual vs. Predicted (Test)
plt.subplot(1, 3, 1)
plt.scatter(test_actual, test_pred, alpha=0.5)
plt.plot(
    [min(test_actual), max(test_actual)], [min(test_actual), max(test_actual)], "r--"
)
plt.xlabel("Actual pIC50 (standardized)")
plt.ylabel("Predicted pIC50 (standardized)")
plt.title(f"Test: Predicted vs Actual\nMSE: {test_mse:.4f}, R²: {test_r2:.4f}")
plt.grid(True)

# Histogram of prediction uncertainties (Test)
plt.subplot(1, 3, 2)
plt.hist(test_uncertainty, bins=50)
plt.xlabel("Prediction Uncertainty (σ)")
plt.ylabel("Count")
plt.title("Distribution of Prediction Uncertainties (Test)")
plt.grid(True)

# Scatter plot: Uncertainty vs. Absolute Error (Test)
plt.subplot(1, 3, 3)
plt.scatter(test_uncertainty, test_errors, alpha=0.5)
plt.xlabel("Prediction Uncertainty (σ)")
plt.ylabel("Absolute Error")
plt.title("Test: Error vs Uncertainty")
plt.grid(True)

plt.tight_layout()
test_plot_path = os.path.join(
    "/home4/s6273475/ml/master_project/models/mpnn_transformer",
    "test_inference_results.png",
)
plt.savefig(test_plot_path, dpi=300, bbox_inches="tight")
plt.close()
logging.info(f"Test inference results plot saved as '{test_plot_path}'")

# -------------------------------
# Save any additional generated figures.
# -------------------------------
output_dir = "/home4/s6273475/ml/master_project/models/mpnn_transformer"
for fig_num in plt.get_fignums():
    plt.figure(fig_num)
    save_path = os.path.join(output_dir, f"figure_{fig_num}.png")
    plt.savefig(save_path)
    print(f"Figure {fig_num} saved to {save_path}")

# -------------------------------
# Print summary metrics for Validation and Test sets.
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
