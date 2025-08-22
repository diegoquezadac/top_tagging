import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
from src.resnet50.dataset import ImageDataset
from src.resnet50.model import ResNet, Bottleneck
from torch.utils.data import DataLoader, random_split

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device).float()
        optimizer.zero_grad()
        outputs = model(imgs).squeeze(1)
        loss = criterion(outputs, labels)
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
    # TODO: Use logging instead of print
    # TODO: Probably move data to device
    # TODO: Check hyperparameters from paper

    # Define variables
    epochs = 100
    lr = 1e-2
    batch_size = 256
    dropout_p = 0.5
    val_split = 0.2
    max_jets = 500

    dataset = ImageDataset("./data/test-public-small.h5")
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Define model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_resnet50 = False
    if use_resnet50: # NOTE: Gigantic model ... Not used for jet tagging
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(model.fc.in_features, 1)
        )
    else:
        model = ResNet(Bottleneck, [3, 4, 6, 3], dropout_p=dropout_p)

    model.to(device)

    # Train model
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # checkpoint directory
    checkpoint_dir = Path.cwd() / "checkpoints/resnet50"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # store history like keras
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # save metrics
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # print progress
        print(
            f"Epoch {epoch}/{epochs} - "
            f"Train Loss: {train_loss:.4f} - "
            f"Val Loss: {val_loss:.4f} - "
            f"Val Acc: {val_acc:.4f}"
        )

        # save checkpoint if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / f"epoch{epoch:02d}-val{val_loss:.4f}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": val_loss,
            }, checkpoint_path)
            print(f"✅ Saved checkpoint: {checkpoint_path}")

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