import numpy as np
from src.utils import load_from_files


class PointDataset(object):
    def __init__(
        self,
        data,
        labels,
        max_jets: int = 500,
        max_constits: int = 80,
        use_train_weights: bool = True,
    ):
        data, labels, _, _, _ = load_from_files(
            [], max_jets=max_jets, max_constits=max_constits, use_train_weights=False
        )

        self.label = "label"
        self._values = {}
        self._values["features"] = data
        self._values["points"] = np.stack([data[:, :, 0], data[:, :, 1]], axis=-1)
        self._label = np.stack((labels, 1 - labels), axis=-1)

    def __len__(self):
        return len(self._label)

    def __getitem__(self, key):
        if key == self.label:
            return self._label
        else:
            return self._values[key]

    @property
    def X(self):
        return self._values

    @property
    def y(self):
        return self._label

    def shuffle(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        shuffle_indices = np.arange(self.__len__())
        np.random.shuffle(shuffle_indices)
        for k in self._values:
            self._values[k] = self._values[k][shuffle_indices]
        self._label = self._label[shuffle_indices]
