import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import h5py
import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from src.bnn.dataset import TabularDataset
from src.bnn.model import BNN
from torch.utils.data import DataLoader
from src.utils import get_device, get_logger, get_metrics, load_weights, plot_rejection_vs_variable

def predict(model, x, n_samples=50):
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.eval()
    with torch.no_grad():
        preds = torch.stack([model(x) for _ in range(n_samples)])
        preds = torch.sigmoid(preds)
    return preds.mean(0), preds.var(0)


if __name__ == "__main__":
    logger = get_logger("bnn_evaluation")

    n_samples = 10
    max_jets = None
    max_constits = 80
    batch_size = 1_000
    device = get_device()

    parser = argparse.ArgumentParser(description="BNN model testing")

    parser.add_argument(
        "model_checkpoint",
        type=str,
        help="Model to be tested",
    )

    parser.add_argument(
        "dataset_file",
        type=str,
        default="./data/test-preprocessed.h5",
        help="Path to the testing file",
    )

    args = parser.parse_args()

    dataset = TabularDataset(
        args.dataset_file, max_jets=max_jets, use_train_weights=False
    )
    x0, y0, *_ = dataset[0]
    input_dim = x0.numel()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = load_weights(BNN(input_dim), args.model_checkpoint, device=device)

    means, vars_, targets = [], [], []
    for x, y in tqdm(loader, desc="Predicting"):
        x = x.to(device).view(x.size(0), -1)
        mean, var = predict(model, x, n_samples)
        means.append(mean.detach().cpu())
        vars_.append(var.detach().cpu())
        targets.append(y.view(-1).detach().cpu())

    mean_all = torch.cat(means).detach().numpy()
    var_all = torch.cat(vars_).detach().numpy()
    target_all = torch.cat(targets).detach().numpy()

    df = pd.DataFrame(
        {
            "y_true": target_all.flatten(),
            "y_pred": mean_all.flatten(),
        }
    )

    metrics_tpr_05 = get_metrics(
        df["y_true"].values, df["y_pred"].values, tpr_threshold=0.5
    )
    metrics_tpr_08 = get_metrics(
        df["y_true"].values, df["y_pred"].values, tpr_threshold=0.8
    )

    logger.info("Metrics at TPR=0.5:")
    for metric, value in metrics_tpr_05.items():
        logger.info(f"{metric}: {value:.4f}")

    logger.info("\nMetrics at TPR=0.8:")
    for metric, value in metrics_tpr_08.items():
        logger.info(f"{metric}: {value:.4f}")

    figures_dir = Path("figures") / "bnn"
    y_pred_flat = mean_all.flatten()
    y_true_flat = target_all.flatten()
    n_jets = len(y_true_flat)

    with h5py.File(args.dataset_file, "r") as f:
        fjet_pt    = f["fjet_pt"][:n_jets] * 1e-3
        fjet_m     = f["fjet_m"][:n_jets] * 1e-3
        fjet_eta   = f["fjet_eta"][:n_jets]
        n_constits = f["n_constits"][:n_jets]

    for vals, fname, xlabel in [
        (fjet_pt,    "rejection_vs_pt.png",           r"Jet $p_T$ [GeV]"),
        (fjet_m,     "rejection_vs_mass.png",          r"Jet mass [GeV]"),
        (fjet_eta,   "rejection_vs_eta.png",            r"Jet $\eta$"),
        (n_constits, "rejection_vs_multiplicity.png",  "Constituent multiplicity"),
    ]:
        plot_rejection_vs_variable(y_pred_flat, y_true_flat, vals,
            xlabel=xlabel, out_path=figures_dir / fname)
        logger.info(f"Saved {fname}")
