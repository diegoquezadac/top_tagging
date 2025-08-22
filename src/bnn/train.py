import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
import matplotlib.pyplot as plt
from src.bnn.dataset import TabularDataset
from src.bnn.model import BNN
from torch.utils.data import DataLoader, random_split


def train_one_epoch(model, loader, criterion, optimizer, device, l1_lambda):
    model.train()
    running_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(imgs).squeeze(1)
        loss = criterion(outputs, labels)
        l1_reg = sum(torch.sum(torch.abs(p)) for p in model.parameters() if p.dim() > 1)
        loss = loss + l1_lambda * l1_reg
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device).float()
            outputs = model(imgs).squeeze(1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds == labels.long()).sum().item()
    accuracy = correct / len(loader.dataset)
    return running_loss / len(loader.dataset), accuracy


if __name__ == "__main__":
    # Define variables
    epochs = 100
    lr = 1e-2 # 1.2 * 1e-5
    batch_size = 256
    val_split = 0.2
    l1_lambda = 2e-4

    # Define datasets
    dataset = TabularDataset("./data/test-public-small.h5")
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # Define data loader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Define model
    n_features = dataset.X.shape[1]
    model = BNN(n_features)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define stuff
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # checkpoint directory
    checkpoint_dir = Path.cwd() / "checkpoints/bnn"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # store history like keras
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    best_val_loss = float("inf")

    # Train model
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, l1_lambda
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # save metrics
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # print progress
        print(
            f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"
        )

        # save checkpoint if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / f"epoch{epoch:02d}-val{val_loss:.4f}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                checkpoint_path,
            )
            print(f"✅ Saved checkpoint: {checkpoint_path}")

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
