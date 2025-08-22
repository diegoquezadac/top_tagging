import torch
import numpy as np
from torch.utils.data import Dataset
from src.utils import load_from_files


class TabularDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        max_jets: int = 500,
        max_constits: int = 80,
        use_train_weights: bool = True,
    ):
        data, labels, train_weights, _, _ = load_from_files(
            [file_path],
            max_jets=max_jets,
            max_constits=max_constits,
            use_train_weights=False,
        )

        X = data.reshape(-1, max_constits * data.shape[-1])
        self.X = X
        self.y = labels.astype(np.float32)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y)
