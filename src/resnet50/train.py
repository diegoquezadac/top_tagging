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
import torchvision.models as models
import matplotlib.pyplot as plt
from src.resnet50.dataset import ImageDataset
from src.resnet50.model import ResNet, Bottleneck
from torch.utils.data import DataLoader, random_split
from src.utils import train_loop, test_loop, get_device, get_logger

SEED = 21
logger = get_logger("resnet50_training")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

if __name__ == "__main__":
    epochs = 100
    lr = 1e-2
    batch_size = 250
    dropout_p = 0.5
    val_split = 0.2

    max_constits = 80
    num_workers = 10

    parser = argparse.ArgumentParser(description="ResNet50 model training")
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
    args = parser.parse_args()

    logger.info("Defining datasets")
    dataset = ImageDataset(args.input_path, use_train_weights=True)
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

    device = get_device()
    model = ResNet(Bottleneck, [3, 4, 6, 3], dropout_p=dropout_p)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=lr)

    checkpoint_dir = Path.cwd() / "checkpoints/resnet50"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_model.pt"

    # store history like keras
    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_val_loss = float("inf")
    start_epoch = 1

    if args.resume:
        if checkpoint_path.exists():
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_loss = checkpoint["val_loss"]
            history = checkpoint.get("history", history)
            if checkpoint["max_constits"] != max_constits or checkpoint["val_split"] != val_split:
                logger.warning("Dataset parameters have changed since checkpoint!")
            logger.info(
                f"Resuming training from epoch {start_epoch} with best val loss {best_val_loss:.4f}"
            )
        else:
            logger.error(f"Checkpoint file {checkpoint_path} not found!")
            sys.exit(1)

    for epoch in range(1, epochs + 1):
        train_loss = train_loop(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = test_loop(model, val_loader, criterion, device)

        # save metrics
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # print progress
        logger.info(
            f"Epoch {epoch}/{epochs} - "
            f"Train Loss: {train_loss:.4f} - "
            f"Val Loss: {val_loss:.4f} - "
            f"Val Acc: {val_acc:.4f}"
        )

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
    plt.legend()
    figure_dir = Path.cwd() / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_dir / "loss_resnet50.png", dpi=300)
    plt.clf()
