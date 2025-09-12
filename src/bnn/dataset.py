import h5py
import torch
from torch.utils.data import Dataset

class TabularDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        max_jets: int = None,
        max_constits: int = 80,
        use_train_weights: bool = True,
    ):
        self.file_path = file_path
        self.max_constits = max_constits
        self.use_train_weights = use_train_weights

        with h5py.File(file_path, "r") as f:
            self.num_samples = len(f["labels"])

        self.max_jets = (
            self.num_samples if max_jets is None else min(max_jets, self.num_samples)
        )

        self._h5file = None

    def _get_file(self):
        if self._h5file is None:
            self._h5file = h5py.File(self.file_path, "r", swmr=True)
        return self._h5file

    def __len__(self):
        return self.max_jets

    def __getitem__(self, idx):
        f = self._get_file()
        features = f["features"][idx].reshape(-1)
        label = f["labels"][idx]

        if self.use_train_weights:
            weight = f["weights"][idx]
            return (
                torch.tensor(features, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32),
                torch.tensor(weight, dtype=torch.float32),
            )
        else:
            return (
                torch.tensor(features, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32),
            )

    def __del__(self):
        if self._h5file is not None:
            try:
                self._h5file.close()
            except:
                pass