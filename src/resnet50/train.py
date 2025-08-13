import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from src.utils import load_from_files
from src.resnet50.dataset import JetDataset
from src.resnet50.preprocess import preprocess
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
    # TODO: Receive the h5 file path as input
    # TODO: Receive the output file peth as input
    # TODO: Probably move data to device
    # TODO: Check hyperparameters from paper

    # Define variables
    epochs = 10
    lr = 1e-2
    batch_size = 256
    dropout_p = 0.5
    val_split = 0.2
    max_jets = 500

    # Load raw data
    data, labels, _, _, _ = load_from_files(
        ["./data/test-public-small.h5"], max_jets=max_jets, use_train_weights=False
    )

    # Define data loaders
    images = preprocess(data)
    dataset = JetDataset(images, labels)
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

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}"
        )