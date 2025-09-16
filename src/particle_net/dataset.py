import h5py
import numpy as np
import tensorflow as tf

class PointDataset:
    def __init__(
        self,
        file_path: str,
        max_jets: int = None,
        max_constits: int = 80,
    ):
        self.file_path = file_path
        self.max_constits = max_constits
        self._h5file = None

        with h5py.File(file_path, "r") as f:
            if "labels" not in f:
                raise KeyError(f"Dataset 'labels' not found in HDF5 file. Available keys: {list(f.keys())}")
            self.num_samples = len(f["labels"])

        self.max_jets = (
            self.num_samples if max_jets is None else min(max_jets, self.num_samples)
        )

    def _get_file(self):
        if self._h5file is None:
            self._h5file = h5py.File(self.file_path, "r", swmr=True)
        return self._h5file

    def __len__(self):
        return self.max_jets

    def __getitem__(self, idx):
        """Get a specific sample by index, returning (features, points, label, weight)."""
        if not isinstance(idx, int):
            raise TypeError(f"Index must be an integer, got {type(idx)}")
        if idx < 0 or idx >= self.max_jets:
            raise IndexError(f"Index {idx} out of range [0, {self.max_jets})")
        
        f = self._get_file()
        # Load features for this sample
        features = f["features"][idx].astype(np.float32)  # Shape: (max_constits, num_features)
        # Derive points
        points = np.stack([features[:, 0], features[:, 1]], axis=-1)  # Shape: (max_constits, 2)
        # Load and process label
        label = f["labels"][idx].astype(np.int32)  # Scalar
        label_one_hot = np.stack([label, 1 - label], axis=-1)  # Shape: (2,)
        # Load weight
        weight = f["weights"][idx].astype(np.float32)  # Scalar
        return features, points, label_one_hot, weight

    def _generator(self):
        """Generator to yield samples one at a time from HDF5 file."""
        f = self._get_file()
        for idx in range(self.max_jets):
            # Define features for this sample (equivalent to self._values["features"] = data)
            features = f["features"][idx].astype(np.float32)  # Shape: (max_constits, num_features)
        
            # Define points for this sample (equivalent to self._values["points"] = np.stack([data[:, :, 0], data[:, :, 1]], axis=-1))
            points = np.stack([features[:, 0], features[:, 1]], axis=-1)  # Shape: (max_constits, 2)
            
            # Define label for this sample (equivalent to self._label = np.stack((labels, 1 - labels), axis=-1))
            label = f["labels"][idx].astype(np.int32)  # Scalar
            label_one_hot = np.stack([label, 1 - label], axis=-1)  # Shape: (2,)
            
            # Define weight for this sample
            weight = f["weights"][idx].astype(np.float32)  # Scalar
            
            yield features, points, label_one_hot, weight

    def __del__(self):
        if self._h5file is not None:
            try:
                self._h5file.close()
            except:
                pass

def create_data_loader(dataset, batch_size=250):
    """Create a tf.data.Dataset pipeline for the PointDataset."""
    # Define output signature
    output_signature = (
        tf.TensorSpec(shape=(dataset.max_constits, None), dtype=tf.float32),  # features
        tf.TensorSpec(shape=(dataset.max_constits, 2), dtype=tf.float32),     # points
        tf.TensorSpec(shape=(2,), dtype=tf.float32),                          # label
        tf.TensorSpec(shape=(), dtype=tf.float32),                           # weight
    )

    # Create dataset from generator
    tf_dataset = tf.data.Dataset.from_generator(
        dataset._generator,
        output_signature=output_signature
    )

    # Preprocessing function to format outputs
    def preprocess_data(features, points, label, weight):
        inputs = {"features": features, "points": points}
        return inputs, label, weight  # Return as (inputs, targets, sample_weights)

    # Apply preprocessing, batch, and prefetch (no shuffling)
    tf_dataset = tf_dataset.map(preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
    tf_dataset = tf_dataset.batch(batch_size)
    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)

    return tf_dataset