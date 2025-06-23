import os
import warnings
import logging
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for command-line usage
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Set CUDA architecture list for NVIDIA L40S (compute capability 8.9)
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"

from DeepPurpose import utils, dataset
from DeepPurpose import DTI as models

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

# Check if GPU is available; if not, exit the program.
if not torch.cuda.is_available():
    raise RuntimeError("GPU not available. This script requires a GPU to run.")
print("GPU Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device Name:", torch.cuda.get_device_name(0))

# Load dataset from Excel file
file_path = "/home4/s6273475/ml/master_project/data/EGFR_IC50_cell_based.xlsx"
data = pd.read_excel(file_path, engine="openpyxl")

# Rename columns to the expected names by DeepPurpose
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

# Extract relevant columns into lists
X_drugs = data["smiles"].tolist()
X_targets = data["sequence"].tolist()
y = data["Label"].tolist()

# Print sample entries for verification
print("Drug 1:", X_drugs[0])
print("Target 1:", X_targets[0])
print("pIC50 1:", y[0])
print("Total pairs:", len(y))

# Process the data and split into training, validation, and test sets.
# Here we specify Daylight for drugs and Transformer for targets.
train, val, test = utils.data_process(
    X_drugs,
    X_targets,
    y,
    drug_encoding="Daylight",
    target_encoding="Transformer",
    split_method="random",
    frac=[0.7, 0.1, 0.2],
    random_seed=1,
)
print("Training sample:", train.head(1))

# Define the folder to store model outputs and figures.
result_folder = "/home4/s6273475/ml/master_project/models/daylight_transformer"
os.makedirs(result_folder, exist_ok=True)

# Generate a configuration dictionary.
# Note that for the target, we now specify transformer-specific parameters.
config = utils.generate_config(
    drug_encoding="Daylight",  # Using Daylight (fingerprint) encoder for drugs
    target_encoding="Transformer",  # Using Transformer encoder for targets
    hidden_dim_drug=128,
    hidden_dim_protein=256,
    # Transformer target parameters:
    input_dim_protein=26,  # Vocabulary size for protein sequences
    transformer_emb_size_target=128,  # Embedding size for Transformer target encoder
    transformer_dropout_rate=0.1,  # Dropout rate for embeddings
    transformer_n_layer_target=2,  # Number of Transformer layers for targets
    transformer_intermediate_size_target=256,  # Intermediate feedforward layer size
    transformer_num_attention_heads_target=4,  # Number of attention heads
    transformer_attention_probs_dropout=0.1,  # Dropout rate for attention probabilities
    transformer_hidden_dropout_rate=0.1,  # Dropout rate in the Transformer layers
    cls_hidden_dims=[512, 128],  # Classifier/regression head dimensions
    train_epoch=5,
    LR=0.001,
    batch_size=128,
    result_folder=result_folder,
)

# Initialize the model with the configuration.
model = models.model_initialize(**config)
print(model)

# -------------------------------
# Train the model and capture loss history if available.
# -------------------------------
try:
    history = model.train(train, val, test)
except RuntimeError as e:
    if "no kernel image is available" in str(e):
        print("Encountered CUDA kernel image error. Falling back to CPU training.")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        model = models.model_initialize(**config)  # reinitialize for CPU-only execution
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

# Save the trained model.
model_save_path = os.path.join(result_folder, "trained_model.pth")
torch.save(model, model_save_path)
print(f"Model saved to {model_save_path}")


# -------------------------------
# Function to perform prediction with uncertainty estimation using MC dropout.
# -------------------------------
def predict_with_uncertainty(model, data, n_iter=10):
    """
    Runs multiple forward passes through the model (with dropout active)
    to estimate prediction uncertainty.
    """
    # Ensure dropout is active.
    model.model.train()
    preds = []
    for _ in range(n_iter):
        pred = model.predict(data)
        preds.append(pred)
    preds = np.array(preds)  # shape: (n_iter, num_samples)
    pred_mean = preds.mean(axis=0)
    pred_std = preds.std(axis=0)
    return pred_mean, pred_std


# -------------------------------
# Obtain predictions on the validation set.
# -------------------------------
val_actual = np.array(val["Label"])
val_pred, val_uncertainty = predict_with_uncertainty(model, val, n_iter=10)
val_mse = mean_squared_error(val_actual, val_pred)
val_r2 = r2_score(val_actual, val_pred)
val_mae = mean_absolute_error(val_actual, val_pred)
val_errors = np.abs(val_actual - val_pred)

# -------------------------------
# Obtain predictions on the test set.
# -------------------------------
test_actual = np.array(test["Label"])
test_pred, test_uncertainty = predict_with_uncertainty(model, test, n_iter=10)
test_mse = mean_squared_error(test_actual, test_pred)
test_r2 = r2_score(test_actual, test_pred)
test_mae = mean_absolute_error(test_actual, test_pred)
test_errors = np.abs(test_actual - test_pred)

# -------------------------------
# Plot Training vs. Validation Loss
# -------------------------------
if train_losses and val_losses:
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Training vs. Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    loss_plot_path = os.path.join(result_folder, "training_vs_validation_loss.png")
    plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Training vs. Validation Loss plot saved as '{loss_plot_path}'")
else:
    logging.warning("No training/validation loss history available for plotting.")

# -------------------------------
# Plotting Validation Results
# -------------------------------
plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.scatter(val_actual, val_pred, alpha=0.5)
plt.plot([min(val_actual), max(val_actual)], [min(val_actual), max(val_actual)], "r--")
plt.xlabel("Actual pIC50")
plt.ylabel("Predicted pIC50")
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
val_plot_path = os.path.join(result_folder, "validation_inference_results.png")
plt.savefig(val_plot_path, dpi=300, bbox_inches="tight")
plt.close()
logging.info(f"Validation inference results plot saved as '{val_plot_path}'")

# -------------------------------
# Plotting Test Results
# -------------------------------
plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.scatter(test_actual, test_pred, alpha=0.5)
plt.plot(
    [min(test_actual), max(test_actual)], [min(test_actual), max(test_actual)], "r--"
)
plt.xlabel("Actual pIC50")
plt.ylabel("Predicted pIC50")
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
test_plot_path = os.path.join(result_folder, "test_inference_results.png")
plt.savefig(test_plot_path, dpi=300, bbox_inches="tight")
plt.close()
logging.info(f"Test inference results plot saved as '{test_plot_path}'")

# Save any additional generated figures.
for fig_num in plt.get_fignums():
    plt.figure(fig_num)
    save_path = os.path.join(result_folder, f"figure_{fig_num}.png")
    plt.savefig(save_path)
    print(f"Figure {fig_num} saved to {save_path}")

# -------------------------------
# Print summary metrics for validation and test sets.
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
