import torch
import h5py
from torch.utils.data import Dataset

class TabularDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        max_jets: int = None,  # Default to total jets in file
        max_constits: int = 80,
        use_train_weights: bool = False,
    ):
        self.file_path = file_path
        self.max_constits = max_constits
        self.use_train_weights = use_train_weights

        # Open HDF5 file once and keep it open
        self.h5file = h5py.File(file_path, "r", swmr=True)
        self.num_samples = len(self.h5file["labels"])

        # Set max_jets to total number of jets if not specified
        self.max_jets = (
            self.num_samples if max_jets is None else min(max_jets, self.num_samples)
        )

    def __len__(self):
        return self.max_jets

    def __getitem__(self, idx):
        features = self.h5file["features"][idx]
        features = features.reshape(-1)
        label = self.h5file["labels"][idx]

        if self.use_train_weights:
            return (
                torch.tensor(features, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32),
                torch.tensor(self.h5file["weights"][idx], dtype=torch.float32),
            )
        else:
            return (
                torch.tensor(features, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32),
            )

    def __del__(self):
        if hasattr(self, 'h5file') and isinstance(self.h5file, h5py.File):
            self.h5file.close()