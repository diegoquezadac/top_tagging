import torch
import logging


def train_loop(model, loader, criterion, optimizer, device, l1_lambda=None):
    model.train()
    running_loss = 0
    for X, y, w in loader:
        X, y, w = (
            X.to(device),
            y.to(device).float(),
            w.to(device).float(),
        )
        optimizer.zero_grad()
        outputs = model(X).squeeze(1)
        loss = criterion(outputs, y)
        weighted_loss = (loss * w).sum()

        if l1_lambda:
            l1_reg = sum(
                torch.sum(torch.abs(p)) for p in model.parameters() if p.dim() > 1
            )
            weighted_loss = weighted_loss + l1_lambda * l1_reg

        weighted_loss.backward()
        optimizer.step()
        running_loss += weighted_loss.item() * X.size(0)

    return running_loss / len(loader.dataset)


def test_loop(model, loader, criterion, device):
    model.eval()
    running_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y, _ in loader:
            X, y = X.to(device), y.to(device).float()
            outputs = model(X).squeeze(1)
            loss = criterion(outputs, y).sum()
            running_loss += loss.item() * X.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).long()
            correct += (preds == y.long()).sum().item()
    accuracy = correct / len(loader.dataset)
    return running_loss / len(loader.dataset), accuracy


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
