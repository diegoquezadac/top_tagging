# Top Tagging

## Python environment

### Using uv

```bash
uv venv
source .venv/bin/activate
uv sync
```

Additionally, considering the available hardware either install tensorflow-metal or tensorflow gpu:

```bash
uv add tensorflow-metal
uv add tensorflow[and-cuda]
```

### Using pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

```bash
pip install tensorflow-metal
pip install tensorflow[and-cuda]
```

## Data gathering

This project uses the [ATLAS Top Tagging Open Data](https://opendata.cern.ch/record/15013) from CERN OpenData. The datasets are HDF5 files containing jet constituents (up to 80 per jet) with four features each: transverse momentum (pt), pseudorapidity (eta), azimuthal angle (phi), and energy. Labels indicate whether a jet originates from a top quark or from background.

| Dataset | [OpenData](https://opendata.cern.ch/record/15013) | tarron | euler |
|---------|----------------------------------------------------|--------|-------|
| `train-public.h5` | Yes | `/home/rafa/respaldoRaquel/train-public.h5` | `/mnt/storage/rpezoa/train-public.h5` |
| `test-public.h5` | Yes | `/home/raquel/data/test-public.h5` | `/mnt/storage/rpezoa/test-public.h5` |
| `raw_string.h5` | No | `/home/raquel/data/raw_string.h5` | `/mnt/storage/rpezoa/raw_string.h5` |
| `raw_cluster.h5` | No | `/home/raquel/data/raw_cluster.h5` | `/mnt/storage/rpezoa/raw_cluster.h5` |
| `raw_angular.h5` | No | `/home/raquel/data/raw_angular.h5` | `/mnt/storage/rpezoa/raw_angular.h5` |
| `raw_dipole.h5` | No | `/home/raquel/data/raw_dipole.h5` | `/mnt/storage/rpezoa/raw_dipole.h5` |

## Data preprocessing

The preprocessing follows the same logic described in the [original ATLAS implementation](https://gitlab.cern.ch/atlas/ATLAS-top-tagging-open-data/-/blob/master/utils.py?ref_type=heads), but adapted to work in batches (default size: 100,000) so it can handle the large training dataset on machines with standard memory.

You can run the preprocessing with:

```bash
python src/preprocess.py <input.h5> <output.h5>
```

For example:

```bash
python src/preprocess.py data/train-public.h5 data/train-preprocessed.h5
python src/preprocess.py data/test-public.h5 data/test-preprocessed.h5
```

> In the original implementation, `lognorm_pt` and `lognorm_energy` require loading the entire column at once. In our version, the `compute_stats` function pre-computes the global sums needed for these features, enabling batch processing.

## Training

Activate the python environment and run the following command:

```bash
python src/bnn/train.py data/train-preprocessed.h5
```

If you want to resume the training process from the current best model, simply run:

```bash
python src/bnn/train.py data/train-preprocessed.h5 --resume
```

> The BNN model is used as example, for resnet50 and particlenet the command is analogous.

## Evaluation
