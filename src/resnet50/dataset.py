import h5py
import torch
import numpy as np
from torch.utils.data import Dataset

def build_jet_images(jets):
    nbins = 64
    eta_range = (-2, 2)
    phi_range = (-2, 2)

    images = []
    for jet in jets:
        etas = jet[:, 0]
        phis = jet[:, 1]
        pts = np.exp(jet[:, 2])

        image, _, _ = np.histogram2d(
            etas, phis, bins=nbins, range=[eta_range, phi_range], weights=pts
        )

        total = image.sum()
        if total > 0:
            image /= total

        image = np.log1p(100 * image)

        images.append(image)

    images = np.array(images)

    return images

def build_jet_images_vectorized(jets):
    nbins = 64
    eta_range = (-2, 2)
    phi_range = (-2, 2)

    images = []
    for jet in jets:
        # Extract coordinates
        etas, phis, pts_log = jet[:, 0], jet[:, 1], jet[:, 2]
        pts = np.exp(pts_log)

        # Bin indices
        eta_bins = np.linspace(*eta_range, nbins + 1)
        phi_bins = np.linspace(*phi_range, nbins + 1)

        # Compute 2D histogram
        image, _, _ = np.histogram2d(etas, phis, bins=[eta_bins, phi_bins], weights=pts)

        # Normalize and apply log scaling
        total = image.sum()
        if total > 0:
            image /= total
        image = np.log1p(100 * image)

        images.append(image)

    return np.array(images)

class ImageDataset(Dataset):
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

        features = f["features"][idx]
        label = f["labels"][idx]

        features = features[None, :, :]
        image = build_jet_images_vectorized(features)        

        if self.use_train_weights:
            weight = f["weights"][idx]
            return (
                torch.tensor(image, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32),
                torch.tensor(weight, dtype=torch.float32),
            )
        else:
            return (
                torch.tensor(image, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32),
            )

    def __del__(self):
        if self._h5file is not None:
            try:
                self._h5file.close()
            except:
                pass