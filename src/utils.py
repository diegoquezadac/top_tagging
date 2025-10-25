import torch
import logging
import numpy as np


def train_loop(model, loader, criterion, optimizer, device, l1_lambda=None):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for X, y, w in loader:
        X, y, w = X.to(device), y.to(device).float(), w.to(device).float()
        optimizer.zero_grad()

        outputs = model(X).squeeze(1)

        # criterion must have reduction='none'
        loss = criterion(outputs, y)                     # shape: (batch_size,)
        weighted_loss = (loss * w).sum() / w.sum()       # weighted mean loss

        # Optional L1 regularization
        if l1_lambda:
            l1_reg = sum(p.abs().sum() for p in model.parameters() if p.dim() > 1)
            weighted_loss += l1_lambda * l1_reg / len(X) # scale L1 by batch size

        weighted_loss.backward()
        optimizer.step()

        running_loss += weighted_loss.item() * X.size(0) # convert back to sum
        total_samples += X.size(0)

    return running_loss / total_samples                  # mean loss per sample


def test_loop(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for X, y, w in loader:                           # even if weights unused
            X, y = X.to(device), y.to(device).float()
            outputs = model(X).squeeze(1)

            loss = criterion(outputs, y)                 # shape: (batch_size,)
            running_loss += loss.sum().item()
            total_samples += X.size(0)

            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds == y.long()).sum().item()

    mean_loss = running_loss / total_samples
    accuracy = correct / total_samples
    return mean_loss, accuracy


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_logger(name: str):
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
