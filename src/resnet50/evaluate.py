import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import pandas as pd
from tqdm import tqdm
from src.resnet50.dataset import ImageDataset
from src.resnet50.model import ResNet, Bottleneck
from torch.utils.data import DataLoader
from src.utils import get_device, get_logger, get_metrics, load_weights


if __name__ == "__main__":
    logger = get_logger("resnet50_evaluation")

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

    dataset = ImageDataset(
        args.dataset_file, max_jets=max_jets, use_train_weights=False
    )
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

    model = load_weights(
        ResNet(Bottleneck, [3, 4, 6, 3], dropout_p=0.5),
        args.model_checkpoint,
        device=device,
    )

    model.eval()

    y_pred, y_true = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Predicting"):
            x = x.to(device)
            out = model(x)
            y_pred.append(out.cpu())
            y_true.append(y.view(-1))

    y_pred_all = torch.cat(y_pred).numpy()
    y_true_all = torch.cat(y_true).numpy()

    df = pd.DataFrame(
        {
            "y_true": y_true_all.flatten(),
            "y_pred": y_pred_all.flatten(),
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

    logger.info("Metrics at TPR=0.8:")
    for metric, value in metrics_tpr_08.items():
        logger.info(f"{metric}: {value:.4f}")
