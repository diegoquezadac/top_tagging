import sys
import os
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
import matplotlib.pyplot as plt
from src.bnn.dataset import TabularDataset
from src.bnn.model import BNN
from torch.utils.data import DataLoader, random_split
from src.utils import train_loop, test_loop, get_device

# Configurar el formateador
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

# Configurar el logger
logger = logging.getLogger("BNN")
logger.setLevel(logging.INFO)

# Agregar handler para consola
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)



if __name__ == "__main__":
    # Define variables
    epochs = 100
    lr = 1.2 * 1e-5
    batch_size = 256
    val_split = 0.2
    l1_lambda = 2e-4

    max_constits = 80
    num_workers = 10

    logger.info("Defining dataset")
    dataset = TabularDataset(
        "./data/train-preprocessed.h5",
        max_constits=max_constits,
        use_train_weights=True,
    )
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    logger.info("Defining dataloaders")
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    logger.info("Preparing training")
    n_features = max_constits * 7
    model = BNN(n_features)

    device = get_device()
    logger.info(f"Moving model to device {device}")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    checkpoint_dir = Path.cwd() / "checkpoints/bnn"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    best_val_loss = float("inf")

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
                },
                checkpoint_path,
            )
            logger.info(f"✅ Saved checkpoint: {checkpoint_path}")

        # --- Plot training curves ---
        plt.plot(history["train_loss"], label="Training")
        plt.plot(history["val_loss"], label="Validation")
        plt.ylabel("Cross-entropy Loss")
        plt.xlabel("Training Epoch")
        plt.legend()
        figure_dir = Path.cwd() / "figures"
        figure_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(figure_dir / "loss_bnn.png", dpi=300)
        plt.clf()
