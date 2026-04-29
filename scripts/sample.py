import h5py
import argparse
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser(description="Sample a subset of jets from an HDF5 file")
parser.add_argument("input_path", type=str, help="Path to the input HDF5 file")
parser.add_argument("output_dir", type=str, help="Directory to save the sampled file")
parser.add_argument("--n", type=int, default=100_000, help="Number of jets to sample (default: 100000)")
args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
stem = Path(args.input_path).stem
output_path = output_dir / f"{stem}_sample.h5"

with h5py.File(args.input_path, 'r') as f_in:
    total_size = f_in['labels'].shape[0]
    n = min(args.n, total_size)

    random_indices = np.random.choice(total_size, size=n, replace=False)
    sorted_idx = np.sort(random_indices)

    with h5py.File(output_path, 'w') as f_out:
        for key, value in f_in.attrs.items():
            f_out.attrs[key] = value

        for key in f_in.keys():
            data = f_in[key]
            sampled = data[sorted_idx] if data.shape[0] == total_size else data[:]
            f_out.create_dataset(key, data=sampled, compression='gzip', compression_opts=4)

print(f"Saved {n} jets to {output_path}")
