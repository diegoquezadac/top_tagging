import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from pathlib import Path
from src.particle_net.dataset import PointDataset, create_data_loader
from src.particle_net.model import get_particle_net
from src.utils import get_logger, get_metrics, plot_rejection_vs_variable


def load_model(path_to_checkpoint, num_classes=2, max_constits=80, num_features=7):
    input_shapes = {"features": (max_constits, num_features), "points": (max_constits, 2)}
    model = get_particle_net(num_classes, input_shapes)

    checkpoint = tf.train.Checkpoint(model=model)
    latest = tf.train.latest_checkpoint(path_to_checkpoint)
    if latest is None:
        raise FileNotFoundError(f"No checkpoint found in {path_to_checkpoint}")
    checkpoint.restore(latest).expect_partial()
    return model


if __name__ == "__main__":
    logger = get_logger("particle_net_evaluation")

    max_jets = None
    max_constits = 80
    batch_size = 1_000
    num_classes = 2

    parser = argparse.ArgumentParser(description="ParticleNet model testing")

    parser.add_argument(
        "model_checkpoint",
        type=str,
        help="Path to the checkpoint directory",
    )

    parser.add_argument(
        "dataset_file",
        type=str,
        default="./data/test-preprocessed.h5",
        help="Path to the testing file",
    )

    args = parser.parse_args()

    dataset = PointDataset(
        args.dataset_file, max_constits=max_constits, max_jets=max_jets
    )
    loader = create_data_loader(dataset, batch_size=batch_size)

    model = load_model(
        args.model_checkpoint,
        num_classes=num_classes,
        max_constits=max_constits,
        num_features=dataset.num_features,
    )

    preds, targets = [], []
    for inputs, labels, _ in tqdm(loader, desc="Predicting"):
        outputs = model(inputs, training=False)
        preds.append(outputs.numpy())
        targets.append(labels.numpy())

    preds_all = np.concatenate(preds, axis=0)
    targets_all = np.concatenate(targets, axis=0)

    df = pd.DataFrame(
        {
            "y_true": targets_all[:, 0].flatten(),
            "y_pred": preds_all[:, 0].flatten(),
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

    figures_dir = Path("figures") / "particle_net"
    y_pred_flat = preds_all[:, 0].flatten()
    y_true_flat = targets_all[:, 0].flatten()
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
