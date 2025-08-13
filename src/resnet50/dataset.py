import torch
import numpy as np
from torch.utils.data import Dataset

class JetDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images.astype(np.float32)
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        img = self.images[idx][None, :, :]
        label = self.labels[idx]

        return torch.tensor(img, dtype=torch.float32), torch.tensor(label)