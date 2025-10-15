import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import random
import numpy as np
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
import matplotlib.pyplot as plt
from src.bnn.dataset import TabularDataset
from src.bnn.model import BNN
from torch.utils.data import DataLoader, random_split
from src.utils import train_loop, test_loop, get_device, get_logger, count_parameters

SEED = 21
logger = get_logger("bnn_training")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


if __name__ == "__main__":
    epochs = 100
    lr = 1.2 * 1e-5
    batch_size = 256
    val_split = 0.25
    l1_lambda = 2e-4

    max_constits = 80

    parser = argparse.ArgumentParser(description="BNN model training")
    parser.add_argument(
        "input_path",
        type=str,
        default="./data/train-preprocessed.h5",
        help="Path to the training file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the best_model.pt checkpoint",
    )

    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers")

    args = parser.parse_args()

    logger.info("Defining datasets")
    dataset = TabularDataset(
        args.input_path,
        max_constits=max_constits,
        use_train_weights=True,
        max_jets=1_000_000,
    )
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    logger.info(f"Train dataset: {train_size} data points")
    logger.info(f"Validation dataset: {val_size} data points")

    logger.info(f"Defining dataloaders with {args.num_workers} workers")
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    logger.info("Preparing training")
    n_features = max_constits * 7
    model = BNN(n_features)
    logger.info(f"Total trainable parameters: {count_parameters(model)}")

    device = get_device()
    logger.info(f"Moving model to device {device}")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    checkpoint_dir = Path.cwd() / "checkpoints/bnn"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_model.pt"

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    start_epoch = 1

    if args.resume:
        if checkpoint_path.exists():
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(
                checkpoint_path, map_location=device, weights_only=False
            )
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["val_loss"]
            history = checkpoint.get("history", history)
            if (
                checkpoint["max_constits"] != max_constits
                or checkpoint["val_split"] != val_split
            ):
                logger.warning("Dataset parameters have changed since checkpoint!")
            logger.info(
                f"Resuming training from epoch {start_epoch} with best val loss {best_val_loss:.4f}"
            )
        else:
            logger.error(f"Checkpoint file {checkpoint_path} not found!")
            sys.exit(1)

    logger.info("Starting epochs")

    for epoch in range(1, epochs + 1):
        train_loss = train_loop(
            model, train_loader, criterion, optimizer, device, l1_lambda=l1_lambda
        )
        val_loss, val_acc = test_loop(model, val_loader, criterion, device)

        # save metrics
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # print progress
        logger.info(
            f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"
        )

        # save checkpoint if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "history": history,
                    "max_constits": max_constits,  # Save dataset parameters
                    "val_split": val_split,
                    "input_path": args.input_path,
                    "lr": optimizer.param_groups[0]["lr"],  # Save current learning rate
                },
                checkpoint_path,
            )
            logger.info(f"✅ Saved checkpoint: {checkpoint_path}")

        # --- Plot training curves ---
        plt.plot(history["train_loss"], label="Training")
        plt.plot(history["val_loss"], label="Validation")
        plt.ylabel("Cross-entropy Loss")
        plt.xlabel("Training Epoch")
        plt.ylim(0, 1)
        plt.legend()
        figure_dir = Path.cwd() / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(figure_dir / "loss_bnn.png", dpi=300)
        plt.clf()
