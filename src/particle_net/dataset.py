# src/particle_net/dataset.py
import h5py
import numpy as np
import tensorflow as tf

class PointDataset:
    def __init__(
        self,
        file_path: str,
        max_jets: int = None,
        max_constits: int = 80,
        indices: list = None,
        features_key: str = "features",
        labels_key: str = "labels",
        weights_key: str = "weights",
    ):
        self.file_path = file_path
        self.max_jets = max_jets
        self.max_constits = max_constits
        self.indices = indices
        self.features_key = features_key
        self.labels_key = labels_key
        self.weights_key = weights_key
        self._h5file = None
        self._initialize()

    def _initialize(self):
        with h5py.File(self.file_path, "r") as f:
            if self.labels_key not in f:
                raise KeyError(f"Dataset '{self.labels_key}' not found in HDF5 file. Available keys: {list(f.keys())}")
            self.num_samples = len(f[self.labels_key])
            self.num_features = f[self.features_key].shape[2]  # e.g., 7
            if self.indices is None:
                self.indices = list(range(self.num_samples))
            self.max_jets = (
                self.num_samples if self.max_jets is None else min(self.max_jets, len(self.indices))
            )
            self.indices = self.indices[:self.max_jets]

    def _get_file(self):
        if self._h5file is None:
            self._h5file = h5py.File(self.file_path, "r", swmr=True)
        return self._h5file

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            raise TypeError(f"Index must be an integer, got {type(idx)}")
        if idx < 0 or idx >= len(self.indices):
            raise IndexError(f"Index {idx} out of range [0, {len(self.indices)})")
        
        f = self._get_file()
        file_idx = self.indices[idx]
        features = f[self.features_key][file_idx].astype(np.float32)  # (80, 7)
        points = np.stack([features[:, 0], features[:, 1]], axis=-1)  # (80, 2)
        label = f[self.labels_key][file_idx].astype(np.int32)  # Scalar
        label_one_hot = np.stack([label, 1 - label], axis=-1)  # (2,)
        weight = f[self.weights_key][file_idx].astype(np.float32)  # Scalar
        return features, points, label_one_hot, weight

    def _generator(self):
        f = self._get_file()
        for idx in self.indices:
            features = f[self.features_key][idx].astype(np.float32)
            points = np.stack([features[:, 0], features[:, 1]], axis=-1)
            label = f[self.labels_key][idx].astype(np.int32)
            label_one_hot = np.stack([label, 1 - label], axis=-1)
            weight = f[self.weights_key][idx].astype(np.float32)
            yield features, points, label_one_hot, weight

    def get_dataset(self):
        output_types = (tf.float32, tf.float32, tf.float32, tf.float32)
        output_shapes = (
            (self.max_constits, self.num_features),  # (80, 7)
            (self.max_constits, 2),                  # (80, 2)
            (2,),                                    # (2,)
            ()                                       # ()
        )
        dataset = tf.data.Dataset.from_generator(
            self._generator,
            output_types=output_types,
            output_shapes=output_shapes
        )
        return dataset

    def inspect_hdf5(self):
        with h5py.File(self.file_path, "r") as f:
            print("HDF5 file keys:", list(f.keys()))
            for key in f.keys():
                print(f"Dataset '{key}' shape: {f[key].shape} dtype: {f[key].dtype}")

    def __del__(self):
        if self._h5file is not None:
            try:
                self._h5file.close()
            except:
                pass

def create_data_loader(dataset, batch_size=250):
    ds = dataset.get_dataset()
    def preprocess_data(features, points, label, weight):
        inputs = {"features": features, "points": points}
        return inputs, label, weight
    ds = ds.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds