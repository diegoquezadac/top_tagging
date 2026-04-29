import os
import sys
import h5py
import logging
import argparse
import warnings
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import plot_preprocessed_jet_images
warnings.filterwarnings("ignore")

formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger("preprocessing")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

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

def compute_stats(input_path, max_constits=80):
    """Compute dataset-level sums for fjet_clus_pt and fjet_clus_E, ignoring padded constituents."""
    sum_pt = 0.0
    sum_energy = 0.0
    batch_size = 100_000
    data_vector_names = ["fjet_clus_pt", "fjet_clus_E"]
    
    with h5py.File(input_path, 'r', swmr=True) as h5file:
        num_samples = len(h5file['labels'])
        
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            data_dict = {key: h5file[key][start:end] for key in data_vector_names}
            
            # Extract raw data, truncated to max_constits
            pt = data_dict['fjet_clus_pt'][:, :max_constits]
            energy = data_dict['fjet_clus_E'][:, :max_constits]
            
            # Mask for padded constituents (pt == 0)
            mask = (pt == 0)
            
            # Sum only real (non-padded) constituents
            sum_pt += np.sum(pt[~mask])
            sum_energy += np.sum(energy[~mask])
    
    return sum_pt, sum_energy

def constituent(data_dict, max_constits=80, sum_pt_global=None, sum_energy_global=None):
    """Constituent function that normalizes pt and energy by global sums if provided, else per-jet sums, keeping 7 features."""
    # Pull data from data dict
    pt = data_dict['fjet_clus_pt'][:, :max_constits]
    eta = data_dict['fjet_clus_eta'][:, :max_constits]
    phi = data_dict['fjet_clus_phi'][:, :max_constits]
    energy = data_dict['fjet_clus_E'][:, :max_constits]

    # Find location of zero pt entries
    mask = np.asarray(pt == 0).nonzero()

    # Angular preprocessing
    eta_shift = eta[:, 0]
    phi_shift = phi[:, 0]
    eta -= eta_shift[:, np.newaxis]
    phi -= phi_shift[:, np.newaxis]
    phi = np.where(phi > np.pi, phi - 2 * np.pi, phi)
    phi = np.where(phi < -np.pi, phi + 2 * np.pi, phi)
    second_eta = eta[:, 1]
    second_phi = phi[:, 1]
    alpha = np.arctan2(second_phi, second_eta) + np.pi / 2
    eta_rot = (eta * np.cos(alpha[:, np.newaxis]) + phi * np.sin(alpha[:, np.newaxis]))
    phi_rot = (-eta * np.sin(alpha[:, np.newaxis]) + phi * np.cos(alpha[:, np.newaxis]))
    eta = eta_rot
    phi = phi_rot
    third_eta = eta[:, 2]
    parity = np.where(third_eta < 0, -1, 1)
    eta = (eta * parity[:, np.newaxis]).astype(np.float32)
    radius = np.sqrt(eta ** 2 + phi ** 2)

    # pT and energy features
    log_pt = np.log(pt)
    log_energy = np.log(energy)
    
    # Normalize by global sums if provided, else per-jet sums
    if sum_pt_global is not None and sum_energy_global is not None:
        lognorm_pt = np.log(pt / sum_pt_global)
        lognorm_energy = np.log(energy / sum_energy_global)
    else:
        sum_pt = np.sum(pt, axis=1)
        sum_energy = np.sum(energy, axis=1)
        lognorm_pt = np.log(pt / sum_pt[:, np.newaxis])
        lognorm_energy = np.log(energy / sum_energy[:, np.newaxis])

    # Reset padded entries to zero
    eta[mask] = 0
    phi[mask] = 0
    log_pt[mask] = 0
    log_energy[mask] = 0
    lognorm_pt[mask] = 0
    lognorm_energy[mask] = 0
    radius[mask] = 0

    # Stack along last axis (7 features)
    features = [eta, phi, log_pt, log_energy, lognorm_pt, lognorm_energy, radius]
    stacked_data = np.stack(features, axis=-1)

    return stacked_data

def _write_split(input_file, output_path, indices, max_constits, sum_pt_global, sum_energy_global, use_train_weights, batch_size):
    """Write a single split (train or test) to output_path, processing only the given indices."""
    data_vector_names = ['fjet_clus_pt', 'fjet_clus_eta', 'fjet_clus_phi', 'fjet_clus_E']
    n = len(indices)

    with h5py.File(output_path, 'w') as output_file:
        features_ds  = output_file.create_dataset('features',   shape=(n, max_constits, 7), dtype=np.float32)
        labels_ds    = output_file.create_dataset('labels',     shape=(n,), dtype=input_file['labels'].dtype)
        fjet_pt_ds   = output_file.create_dataset('fjet_pt',    shape=(n,), dtype=np.float32)
        fjet_m_ds    = output_file.create_dataset('fjet_m',     shape=(n,), dtype=np.float32)
        fjet_eta_ds  = output_file.create_dataset('fjet_eta',   shape=(n,), dtype=np.float32)
        n_constits_ds = output_file.create_dataset('n_constits', shape=(n,), dtype=np.int32)
        if use_train_weights:
            weights_ds = output_file.create_dataset('weights', shape=(n,), dtype=input_file['weights'].dtype)

        for i, start in enumerate(range(0, n, batch_size)):
            logger.info(f"  batch {i} / {1 + n // batch_size}")
            end = min(start + batch_size, n)
            src = indices[start:end]
            data_dict = {key: input_file[key][src] for key in data_vector_names}
            processed = constituent(data_dict, max_constits,
                                    sum_pt_global=sum_pt_global,
                                    sum_energy_global=sum_energy_global)
            features_ds[start:end]   = processed
            labels_ds[start:end]     = input_file['labels'][src]
            fjet_pt_ds[start:end]    = input_file['fjet_pt'][src]
            fjet_m_ds[start:end]     = input_file['fjet_m'][src]
            fjet_eta_ds[start:end]   = input_file['fjet_eta'][src]
            n_constits_ds[start:end] = (input_file['fjet_clus_pt'][src, :max_constits] > 0).sum(axis=1)
            if use_train_weights:
                weights_ds[start:end] = input_file['weights'][src]


def _plot_jet_images(h5_path, out_dir, logger, n_sample=20_000):
    from pathlib import Path
    out_path = Path("figures") / "preprocessing" / "jet_images_preprocessed.png"
    logger.info(f"Generating preprocessed jet image plot → {out_path}")
    with h5py.File(h5_path, "r") as f:
        n = min(n_sample, len(f["labels"]))
        features = f["features"][:n]
        labels   = f["labels"][:n]
    plot_preprocessed_jet_images(features, labels, out_path)


def preprocess(input_path, output_dir, max_constits=80, batch_size=100_000, use_train_weights=True, split=None):
    """Create preprocessed HDF5 file(s) in output_dir with globally normalized pt and energy features.

    Without --split: writes <output_dir>/train-preprocessed.h5
    With --split f:  writes <output_dir>/train-preprocessed.h5 and <output_dir>/test-preprocessed.h5
    """
    from pathlib import Path
    data_vector_names = ['fjet_clus_pt', 'fjet_clus_eta', 'fjet_clus_phi', 'fjet_clus_E']

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sum_pt_global, sum_energy_global = compute_stats(input_path, max_constits)

    with h5py.File(input_path, 'r', swmr=True) as input_file:
        num_samples = len(input_file['labels'])

        if split is not None:
            train_end = int(num_samples * split)
            train_idx = np.arange(0, train_end)
            test_idx  = np.arange(train_end, num_samples)

            train_path = str(out_dir / 'train-preprocessed.h5')
            test_path  = str(out_dir / 'test-preprocessed.h5')

            logger.info(f"Writing train split ({len(train_idx)} samples) → {train_path}")
            _write_split(input_file, train_path, train_idx, max_constits,
                         sum_pt_global, sum_energy_global, use_train_weights, batch_size)

            logger.info(f"Writing test split ({len(test_idx)} samples) → {test_path}")
            _write_split(input_file, test_path, test_idx, max_constits,
                         sum_pt_global, sum_energy_global, use_train_weights=False, batch_size=batch_size)

            _plot_jet_images(train_path, out_dir, logger)
            return

        train_path = str(out_dir / 'train-preprocessed.h5')
        with h5py.File(train_path, 'w') as output_file:
            features_ds   = output_file.create_dataset('features',   shape=(num_samples, max_constits, 7), dtype=np.float32)
            labels_ds      = output_file.create_dataset('labels',     shape=(num_samples,), dtype=input_file['labels'].dtype)
            fjet_pt_ds     = output_file.create_dataset('fjet_pt',    shape=(num_samples,), dtype=np.float32)
            fjet_m_ds      = output_file.create_dataset('fjet_m',     shape=(num_samples,), dtype=np.float32)
            fjet_eta_ds    = output_file.create_dataset('fjet_eta',   shape=(num_samples,), dtype=np.float32)
            n_constits_ds  = output_file.create_dataset('n_constits', shape=(num_samples,), dtype=np.int32)

            if use_train_weights:
                weights_ds = output_file.create_dataset('weights', shape=(num_samples,), dtype=input_file['weights'].dtype)

            for start in range(0, num_samples, batch_size):
                logger.info(f"Preprocessing batch {start//batch_size} / {1 + num_samples//batch_size}")
                end = min(start + batch_size, num_samples)
                data_dict = {key: input_file[key][start:end] for key in data_vector_names}

                processed_data = constituent(data_dict, max_constits,
                                             sum_pt_global=sum_pt_global,
                                             sum_energy_global=sum_energy_global)

                features_ds[start:end]  = processed_data
                labels_ds[start:end]    = input_file['labels'][start:end]
                fjet_pt_ds[start:end]   = input_file['fjet_pt'][start:end]
                fjet_m_ds[start:end]    = input_file['fjet_m'][start:end]
                fjet_eta_ds[start:end]  = input_file['fjet_eta'][start:end]
                n_constits_ds[start:end] = (input_file['fjet_clus_pt'][start:end, :max_constits] > 0).sum(axis=1)

                if use_train_weights:
                    weights_ds[start:end] = input_file['weights'][start:end]

        logger.info(f"Wrote {train_path}")
        _plot_jet_images(train_path, out_dir, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Atlas Top Tagging Open Data Set")
    parser.add_argument("input_path", type=str, help="Path to the input file")
    parser.add_argument("output_dir", type=str, help="Directory to save preprocessed file(s)")
    parser.add_argument("--max-constits", type=int, default=80, help="Maximum number of constituents (default: 80)")
    parser.add_argument("--batch-size", type=int, default=100_000, help="Batch size for processing (default: 100000)")
    parser.add_argument("--split", type=float, default=None,
                        help="Train fraction (e.g. 0.8). Produces train-preprocessed.h5 and test-preprocessed.h5")

    args = parser.parse_args()

    preprocess(
        input_path=args.input_path,
        output_dir=args.output_dir,
        max_constits=args.max_constits,
        batch_size=args.batch_size,
        split=args.split,
    )