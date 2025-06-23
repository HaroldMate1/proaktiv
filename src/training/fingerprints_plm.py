#!/usr/bin/env python3
import os
import argparse
import logging
import pandas as pd
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from tqdm import tqdm
import optuna  # For hyperparameter optimization
from deepchem.feat import CircularFingerprint
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import pearsonr
from lifelines.utils import concordance_index
from tqdm import tqdm
from tabulate import tabulate
import sys


# -------------------------------
# Dataset and Model Definitions
# -------------------------------
class ProteinLigandDataset(Dataset):
    """
    Dataset for protein variant sequences and ligand SMILES with measured IC50 values.
    Uses DeepChem's CircularFingerprint for batched, precomputed Morgan fingerprints.
    """

    def __init__(self, df, esm_model_name, fp_radius=2, fp_bits=2048, max_length=1024):
        self.df = df.reset_index(drop=True)
        self.smiles = self.df["canonical_smiles"].tolist()
        self.sequences = self.df["variant_mutation_sequence"].values

        # Convert IC50 from nM to M, then to pIC50
        ic50_m = self.df["standard_value"].astype(float).values * 1e-9
        ic50_m = np.clip(ic50_m, a_min=1e-12, a_max=None)
        self.labels = torch.tensor(-np.log10(ic50_m), dtype=torch.float32)

        # Tokenizer and fingerprint generator
        self.tokenizer = AutoTokenizer.from_pretrained(
            esm_model_name, do_lower_case=False
        )
        self.max_length = max_length
        self.fp_radius = fp_radius
        self.fp_bits = fp_bits
        self.fp_generator = CircularFingerprint(size=fp_bits, radius=fp_radius)

        # Precompute all fingerprints at once
        fps_list = []
        # break your full smiles list into chunks of 100
        chunks = [self.smiles[i : i + 100] for i in range(0, len(self.smiles), 100)]
        for chunk in tqdm(chunks, desc="Fingerprint chunks"):
            fps_list.extend(self.fp_generator.featurize(chunk))
        # then stack them back into a tensor
        self.fps = torch.from_numpy(np.vstack(fps_list).astype(np.float32))

    def __len__(self):
        return len(self.df)

    def tokenize_sequence(self, seq):
        encoded = self.tokenizer(
            seq,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        return encoded["input_ids"].squeeze(0), encoded["attention_mask"].squeeze(0)

    def __getitem__(self, idx):
        input_ids, attn_mask = self.tokenize_sequence(self.sequences[idx])
        fp = self.fps[idx]
        label = self.labels[idx]
        return {
            "input_ids": input_ids,
            "attention_mask": attn_mask,
            "fingerprints": fp,
            "labels": label,
        }


def collate_fn(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.stack([b["attention_mask"] for b in batch])
    fingerprints = torch.stack([b["fingerprints"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "fingerprints": fingerprints,
        "labels": labels,
    }


class ProteinLigandModel(nn.Module):
    """
    End-to-end fine-tuned model for bioactivity prediction.
    """

    def __init__(self, esm_model_name, fp_bits=2048, hidden_size=256, dropout=0.1):
        super().__init__()
        print(">>> ProteinLigandModel: loading ESM from", esm_model_name, flush=True)
        self.protein_encoder = AutoModel.from_pretrained(esm_model_name)
        # ── keep checkpointing for memory savings on the 3 B model, too
        self.protein_encoder.gradient_checkpointing_enable()
        print(">>> ProteinLigandModel: ESM loaded (checkpointing enabled)", flush=True)
        self.protein_embed_dim = self.protein_encoder.config.hidden_size
        self.fp_mlp = nn.Sequential(
            nn.Linear(fp_bits, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        combined_dim = self.protein_embed_dim + hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(combined_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, input_ids, attention_mask, fingerprints):
        outputs = self.protein_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        last_hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        summed = torch.sum(last_hidden * mask, dim=1)
        lengths = torch.clamp(torch.sum(mask, dim=1), min=1e-9)
        protein_vec = summed / lengths
        fp_vec = self.fp_mlp(fingerprints)
        combined = torch.cat([protein_vec, fp_vec], dim=1)
        return self.regressor(combined).squeeze(-1)


# -------------------------------
# Training, Evaluation, and Plotting
# -------------------------------
def load_and_split_data(data_path, train_frac, val_frac, test_frac, seed):
    df = pd.read_excel(data_path)
    train_val, test_df = train_test_split(df, test_size=test_frac, random_state=seed)
    train_df, val_df = train_test_split(
        train_val, test_size=val_frac / (train_frac + val_frac), random_state=seed
    )
    return train_df, val_df, test_df


def train_and_evaluate(args):
    print(">>> train_and_evaluate: start", flush=True)
    # Setup: log to file AND console
    os.makedirs(args.result_folder, exist_ok=True)
    log_path = os.path.join(args.result_folder, "train.log")
    handlers = [logging.FileHandler(log_path), logging.StreamHandler(sys.stdout)]
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )
    logging.info("Starting training")
    print("Logging configured. Starting train_and_evaluate()", flush=True)

    # ─── Data splits ──────────────────────────────────────────────────────────
    train_df, val_df, test_df = load_and_split_data(
        args.data_excel, args.train_frac, args.val_frac, args.test_frac, args.seed
    )
    splits_msg = f"Using splits → train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}"
    logging.info(splits_msg)
    print("✔️  " + splits_msg, flush=True)

    # ─── DataLoaders ─────────────────────────────────────────────────────────
    print("⏳  Building DataLoaders…", flush=True)
    train_loader = DataLoader(
        ProteinLigandDataset(
            train_df, args.esm_model, fp_bits=args.fp_bits, max_length=args.max_length
        ),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    print(">>> train_loader built", flush=True)
    print(">>> About to build val_loader", flush=True)
    val_loader = DataLoader(
        ProteinLigandDataset(
            val_df, args.esm_model, fp_bits=args.fp_bits, max_length=args.max_length
        ),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    print(">>> val_loader built", flush=True)

    test_loader = DataLoader(
        ProteinLigandDataset(
            test_df, args.esm_model, fp_bits=args.fp_bits, max_length=args.max_length
        ),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    print(
        f"✔️  DataLoaders ready: {len(train_loader)} train batches, {len(val_loader)} val batches",
        flush=True,
    )

    # freeze all but layer 5
    print("⏳  Loading ESM model…", flush=True)
    model = ProteinLigandModel(
        args.esm_model,
        fp_bits=args.fp_bits,
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    ).to(args.device)
    for name, param in model.protein_encoder.named_parameters():
        if "esm.encoder.layers.35" not in name:
            param.requires_grad = False

    # ─── Wrap for multi-GPU if available ────────────────────────────────────────
    num_gpus = torch.cuda.device_count() if args.device.startswith("cuda") else 0
    print(f">>> Available GPUs: {num_gpus}", flush=True)
    if num_gpus > 1:
        model = nn.DataParallel(model)
        print(f">>> Using DataParallel on devices: {model.device_ids}", flush=True)

    print("✔️  Model loaded. Beginning training loop…", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )
    scaler = GradScaler()
    criterion = nn.MSELoss()

    best_val_rmse = float("inf")
    epochs_no_improve = 0
    train_losses, val_losses = [], []
    best_ckpt_path = None

    table_rows = []

    # keep track of which epoch gave the best RMSE
    best_epoch = 0
    # keep only one checkpoint on disk
    best_ckpt_path = None

    # ─── Epoch loop with tqdm ────────────────────────────────────────────────
    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs", unit="ep"):
        # ---- Training ----
        model.train()
        epoch_train, train_preds, train_labels = [], [], []
        for batch in tqdm(train_loader, desc=f" Epoch {epoch} ▶ train", leave=False):
            optimizer.zero_grad()
            with autocast():
                preds = model(
                    batch["input_ids"].to(args.device),
                    batch["attention_mask"].to(args.device),
                    batch["fingerprints"].to(args.device),
                )
                labels = batch["labels"].to(args.device)
                loss = criterion(preds, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_train.append(loss.item())
            train_preds.extend(preds.detach().cpu().numpy())
            train_labels.extend(labels.detach().cpu().numpy())

        train_rmse = np.sqrt(np.mean(epoch_train))
        train_r, _ = pearsonr(train_labels, train_preds)
        train_losses.append(train_rmse)

        # ---- Validation ----
        model.eval()
        epoch_val, val_preds, val_labels = [], [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f" Epoch {epoch} ▶  val", leave=False):
                preds = model(
                    batch["input_ids"].to(args.device),
                    batch["attention_mask"].to(args.device),
                    batch["fingerprints"].to(args.device),
                )
                labels = batch["labels"]
                loss = criterion(preds, labels.to(args.device))
                epoch_val.append(loss.item())

                val_preds.extend(preds.detach().cpu().numpy())
                val_labels.extend(labels.numpy())

        val_rmse = np.sqrt(np.mean(epoch_val))
        val_r, _ = pearsonr(val_labels, val_preds)
        val_losses.append(val_rmse)

        # ---- Log & print ----
        # ---- Metrics table update & print ----
        # compute p-value and concordance index
        val_r, val_p = pearsonr(val_labels, val_preds)
        val_ci = concordance_index(val_labels, val_preds)

        # append this epoch's row (we’ll number epochs from 0)
        table_rows.append([epoch - 1, val_rmse, val_r, val_p, val_ci])

        # print the full table so far
        headers = ["# epoch", "MSE", "Pearson Corr.", "p-value", "Concordance Index"]
        print(tabulate(table_rows, headers=headers, tablefmt="github", floatfmt=".4f"))
        print()  # blank line for spacing

        # ---- Early stopping & checkpoint ----
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch

            ckpt_path = os.path.join(
                args.result_folder, f"egfr_alk_braf_best_model_epoch{epoch}.pt"
            )
            torch.save(model.state_dict(), ckpt_path)

            # remove old checkpoint
            if best_ckpt_path and os.path.exists(best_ckpt_path):
                os.remove(best_ckpt_path)
            best_ckpt_path = ckpt_path

            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                msg = f"Early stopping at epoch {epoch}"
                logging.info(msg)
                print(msg, flush=True)
                break

        # Save best
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse

            # build new checkpoint path
            ckpt_path = os.path.join(
                args.result_folder, f"egfr_alk_braf_best_model_epoch{epoch}.pt"
            )
            # save new best
            torch.save(model.state_dict(), ckpt_path)

            # delete previous best, if any
            if best_ckpt_path is not None and os.path.exists(best_ckpt_path):
                os.remove(best_ckpt_path)

            # remember this as the new best
            best_ckpt_path = ckpt_path

            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Save summary
    training_results_path = os.path.join(
        args.result_folder, "egfr_alk_braf_training_results.txt"
    )
    with open(training_results_path, "w") as f:
        f.write(f"Best validation RMSE: {best_val_rmse:.4f}\n")
        f.write(f"Best epoch: {best_epoch}\n")

    # -------------------------------
    # Plot Training vs. Validation Loss
    # -------------------------------
    if train_losses and val_losses:
        plt.figure(figsize=(8, 6))
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title("Training vs. Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE")
        plt.legend()
        loss_plot_path = os.path.join(
            args.result_folder, "egfr_alk_braf_training_vs_validation_loss.png"
        )
        plt.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        logging.info(f"Training vs. Validation Loss plot saved as '{loss_plot_path}'")
    else:
        logging.warning("No training/validation loss history available for plotting.")

    # Function to perform MC-dropout inference and collect stats
    def inference(loader):
        model.train()  # enable dropout
        all_actual, all_pred, all_unc, all_err = [], [], [], []
        for batch in tqdm(loader, desc="Inference"):
            inputs = {
                k: batch[k].to(args.device)
                for k in ["input_ids", "attention_mask", "fingerprints"]
            }
            labels = batch["labels"].numpy()
            mc_preds = []
            for _ in range(args.mc_samples):
                preds = model(**inputs).detach().cpu().numpy()
                mc_preds.append(preds)
            mc_preds = np.stack(mc_preds, axis=1)
            mean_pred = mc_preds.mean(axis=1)
            uncert = mc_preds.std(axis=1)
            all_actual.extend(labels)
            all_pred.extend(mean_pred)
            all_unc.extend(uncert)
            all_err.extend(np.abs(mean_pred - labels))
        return (
            np.array(all_actual),
            np.array(all_pred),
            np.array(all_unc),
            np.array(all_err),
        )

    # Inference on validation set
    val_actual, val_pred, val_uncertainty, val_errors = inference(val_loader)
    val_mse = mean_squared_error(val_actual, val_pred)
    val_r2 = r2_score(val_actual, val_pred)
    val_mae = mean_absolute_error(val_actual, val_pred)

    # Inference on test set
    test_actual, test_pred, test_uncertainty, test_errors = inference(test_loader)
    test_mse = mean_squared_error(test_actual, test_pred)
    test_r2 = r2_score(test_actual, test_pred)
    test_mae = mean_absolute_error(test_actual, test_pred)

    # -------------------------------
    # Plotting Validation Results
    # -------------------------------
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.scatter(val_actual, val_pred, alpha=0.5)
    plt.plot(
        [val_actual.min(), val_actual.max()],
        [val_actual.min(), val_actual.max()],
        "r--",
    )
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
    val_plot_path = os.path.join(
        args.result_folder, "egfr_alk_braf_validation_inference_results.png"
    )
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
        [test_actual.min(), test_actual.max()],
        [test_actual.min(), test_actual.max()],
        "r--",
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
    test_plot_path = os.path.join(args.result_folder, "test_inference_results.png")
    plt.savefig(test_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logging.info(f"Test inference results plot saved as '{test_plot_path}'")

    # -------------------------------
    # Print summary metrics
    # -------------------------------
    print("\nValidation Results:")
    print(f"Number of samples: {len(val_pred)}")
    print(f"Mean Squared Error: {val_mse:.4f}")
    print(f"R² Score: {val_r2:.4f}")
    print(f"Mean Absolute Error: {val_mae:.4f}")
    print(f"Mean Uncertainty: {val_uncertainty.mean():.4f}")

    print("\nTest Results:")
    print(f"Number of samples: {len(test_pred)}")
    print(f"Mean Squared Error: {test_mse:.4f}")
    print(f"R² Score: {test_r2:.4f}")
    print(f"Mean Absolute Error: {test_mae:.4f}")
    print(f"Mean Uncertainty: {test_uncertainty.mean():.4f}")


def run_training(args, hyperparams=None):
    # Merge default args and hyperparams
    params = vars(args).copy()
    if hyperparams:
        params.update(hyperparams)
    # Prepare data
    df = pd.read_excel(params["data_excel"])  # Updated to read Excel file
    tr_val, test_df = train_test_split(
        df, test_size=params["test_frac"], random_state=params["seed"]
    )
    train_df, val_df = train_test_split(
        tr_val,
        test_size=params["val_frac"] / (params["train_frac"] + params["val_frac"]),
        random_state=params["seed"],
    )
    train_loader = DataLoader(
        ProteinLigandDataset(
            train_df,
            params["esm_model"],
            fp_bits=params["fp_bits"],
            max_length=params["max_length"],
        ),
        batch_size=params["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        ProteinLigandDataset(
            val_df,
            params["esm_model"],
            fp_bits=params["fp_bits"],
            max_length=params["max_length"],
        ),
        batch_size=params["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )
    device = params["device"]
    model = ProteinLigandModel(
        params["esm_model"], params["fp_bits"], params["hidden_size"], params["dropout"]
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"])
    criterion = nn.MSELoss()

    best_val = float("inf")
    epochs_no_improve = 0
    table_rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for b in train_loader:
            opt.zero_grad()
            preds = model(
                b["input_ids"].to(device),
                b["attention_mask"].to(device),
                b["fingerprints"].to(device),
            )
            loss = criterion(preds, b["labels"].to(device))
            loss.backward()
            opt.step()
            train_losses.append(loss.item())

        # Validation loop
        model.eval()
        val_losses = []
        val_preds, val_labels = [], []
        with torch.no_grad():
            for b in val_loader:
                preds = model(
                    b["input_ids"].to(device),
                    b["attention_mask"].to(device),
                    b["fingerprints"].to(device),
                )
                val_losses.append(criterion(preds, b["labels"].to(device)).item())
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(b["labels"].cpu().numpy())

        val_rmse = np.sqrt(np.mean(val_losses))
        val_r, val_p = pearsonr(val_labels, val_preds)
        val_ci = concordance_index(val_labels, val_preds)

        # record & print table
        table_rows.append([epoch - 1, val_rmse, val_r, val_p, val_ci])
        headers = ["# epoch", "MSE", "Pearson Corr.", "p-value", "Concordance Index"]
        print(tabulate(table_rows, headers=headers, tablefmt="github", floatfmt=".4f"))
        print()

        # Early stopping check
        if val_rmse < best_val - params["min_delta"]:
            best_val = val_rmse
            epochs_no_improve = 0
            # save model
            os.makedirs(params["result_folder"], exist_ok=True)
            torch.save(
                model.state_dict(),
                os.path.join(params["result_folder"], "best_model.pt"),
            )
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= params["patience"]:
                print(f"Early stopping at epoch {epoch}")
                break
    return best_val


def objective(trial, args):
    hyperparams = {
        "lr": trial.suggest_loguniform("lr", 1e-5, 1e-3),
        "hidden_size": trial.suggest_categorical("hidden_size", [128, 256, 512]),
        "dropout": trial.suggest_uniform("dropout", 0.0, 0.5),
        "batch_size": trial.suggest_categorical("batch_size", [4, 8, 16]),
    }
    val_score = run_training(args, hyperparams)
    return val_score


# -------------------------------
# Argument Parsing
# -------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune ESM for bioactivity prediction with automatic HPO"
    )
    parser.add_argument(
        "--data_excel",
        type=str,
        default="/home4/s6273475/ml/master_project/data/egfr_alk_braf_merged.xlsx",
        help="/home4/s6273475/ml/master_project/data/egfr_alk_braf_merged.xlsx",
    )
    parser.add_argument(
        "--esm_model",
        type=str,
        default="facebook/esm2_t36_3B_UR50D",
        help="Pretrained ESM model name or path (3 B-param variant)",
    )

    parser.add_argument("--result_folder", type=str, default="results")
    parser.add_argument("--train_frac", type=float, default=0.7)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--fp_bits", type=int, default=2048)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mc_samples",
        type=int,
        default=10,
        help="Number of MC-dropout samples for uncertainty",
    )

    parser.add_argument(
        "--hpo_trials",
        type=int,
        default=20,
        help="Number of Optuna trials for hyperparameter optimization",
    )
    parser.add_argument(
        "--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu")
    )
    return parser.parse_args()


# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    # 1) Read CLI args
    args = parse_args()

    # 2) Run Optuna HPO if requested
    if args.hpo_trials > 0:
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, args), n_trials=args.hpo_trials)
        print(f"\n=== HPO complete ===")
        print(f"Best validation RMSE: {study.best_value:.4f}")
        print(f"Best hyperparameters: {study.best_params}\n")

        # Save best hyperparameters and validation RMSE
        os.makedirs(args.result_folder, exist_ok=True)
        hpo_results_path = os.path.join(args.result_folder, "hpo_results.txt")
        with open(hpo_results_path, "w") as f:
            f.write(f"Best validation RMSE: {study.best_value:.4f}\n")
            f.write("Best hyperparameters:\n")
            for name, val in study.best_params.items():
                f.write(f"{name}: {val}\n")

        # Overwrite args with best hyperparams
        for name, val in study.best_params.items():
            setattr(args, name, val)

    # 3) Ensure output folder exists, then train + evaluate
    os.makedirs(args.result_folder, exist_ok=True)
    train_and_evaluate(args)
