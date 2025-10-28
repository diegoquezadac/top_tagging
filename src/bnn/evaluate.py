import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from src.bnn.dataset import TabularDataset
from src.bnn.model import BNN
from torch.utils.data import DataLoader
from src.utils import get_device, get_logger
from sklearn.metrics import roc_curve, accuracy_score, recall_score, precision_score


def load_model(path_to_checkpoint, input_dim, device):
    model = BNN(input_dim)
    checkpoint = torch.load(path_to_checkpoint, map_location=device)
    state = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.to(device)
    return model


def predict(model, x, n_samples=50):
    model.train()
    for m in model.modules():
        if isinstance(m, nn.BatchNorm1d):
            m.eval()
    with torch.no_grad():
        preds = torch.stack([model(x) for _ in range(n_samples)])
        preds = torch.sigmoid(preds)
    return preds.mean(0), preds.var(0)


def get_metrics(y_true, y_pred, tpr_threshold=0.5):
    # Find threshold that achieves the desired TPR
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    # Find the closest TPR to the desired threshold
    idx = np.argmin(np.abs(tpr - tpr_threshold))
    threshold = thresholds[idx]
    actual_tpr = tpr[idx]
    actual_fpr = fpr[idx]

    # Convert probabilities to binary predictions using the threshold
    y_pred_binary = (y_pred >= threshold).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)

    # Handle 1/FPR
    inverse_fpr = 1.0 / actual_fpr if actual_fpr > 0 else float("inf")

    return {
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision,
        "tpr": actual_tpr,
        "fpr": actual_fpr,
        "inverse_fpr": inverse_fpr,
        "threshold": threshold,
    }


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

    dataset = TabularDataset(args.dataset_file, max_jets=max_jets)
    x0, y0, *_ = dataset[0]
    input_dim = x0.numel()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=10,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    model = load_model(args.model_checkpoint, input_dim, device=device)

    means, vars_, targets = [], [], []
    for idx, batch in enumerate(loader):
        logger.info(f"Processing batch {idx}")
        x, y = batch[0], batch[1]
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

    # Log or print results
    logger.info("Metrics at TPR=0.5:")
    for metric, value in metrics_tpr_05.items():
        logger.info(f"{metric}: {value:.4f}")

    logger.info("\nMetrics at TPR=0.8:")
    for metric, value in metrics_tpr_08.items():
        logger.info(f"{metric}: {value:.4f}")
