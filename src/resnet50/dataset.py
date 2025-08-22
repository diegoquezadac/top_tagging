import torch
import numpy as np
from torch.utils.data import Dataset
from src.utils import load_from_files
from src.resnet50.preprocess import preprocess


class ImageDataset(Dataset):
    def __init__(self, file_path: str, max_jets: int = 500, max_constits: int = 80):
        data, labels, train_weights, _, _ = load_from_files(
            [file_path],
            max_jets=max_jets,
            max_constits=max_constits,
            use_train_weights=False,
        )
        X = preprocess(data)
        self.X = X.astype(np.float32)
        self.y = labels.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img = self.X[idx][None, :, :]
        y = self.y[idx]

        return torch.tensor(img, dtype=torch.float32), torch.tensor(y)
