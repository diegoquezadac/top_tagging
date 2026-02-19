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

> In [ATLAS Top Tagging Open Data](https://gitlab.cern.ch/atlas/ATLAS-top-tagging-open-data), the variables `lognorm_pt` and `lognorm_energy` require loading the entire column at once. In this version, the `compute_stats` function pre-computes the global sums needed for these features, enabling batch processing.

## Models

Three models are implemented, each with a different approach to represent jet data. For each model, four Python files are provided: `dataset.py`, `model.py`, `train.py`, and `evaluate.py`.

* **BNN**: Bayesian Neural Network (PyTorch). Flattened input (80 constituents x 7 features) through 5 fully connected layers with batch normalization, ReLU, and dropout. Follows the DNN architecture from the Appendix A of *Constituent-Based Top-Quark Tagging with the ATLAS Detector* (2022). Uses Monte Carlo dropout at inference for uncertainty estimation.
* **ResNet50**: Residual network (PyTorch). Jet constituents are binned into 64x64 pT-weighted images in eta-phi space. Uses a Bottleneck-based ResNet50 with layers [3, 4, 6, 3] starting from 16 initial planes.
* **ParticleNet**: Graph neural network (Keras/TensorFlow). Operates on the point cloud of constituents using k-nearest neighbor graphs (k=18) with 3 EdgeConv blocks, following the [official implementation](https://github.com/hqucms/ParticleNet/blob/master/tf-keras/tf_keras_model.py).

## Training

All models are trained using the Adam optimizer and a cross-entropy loss function, with a 90/10 train/validation split. The best model (lowest validation loss) is saved automatically, and training curves are written to `figures/`.

```bash
python src/bnn/train.py ./data/train-preprocessed.h5
python src/resnet50/train.py ./data/train-preprocessed.h5
python src/particle_net/train.py ./data/train-preprocessed.h5
```

To resume training from the last best checkpoint, add the `--resume` flag:

```bash
python src/<model>/train.py ./data/train-preprocessed.h5 --resume
```

## Evaluation

Evaluation is performed on a test subset of 10,000 jets. The following metrics are computed at two signal efficiency working points (TPR=0.5 and TPR=0.8): accuracy, AUC, recall, precision, TPR, FPR, and background rejection (1/FPR). For the BNN, evaluation uses Monte Carlo dropout with 10 stochastic forward passes to produce mean predictions and uncertainty estimates.

```bash
python src/bnn/evaluate.py ./checkpoints/bnn/best_model.pt ./data/test-preprocessed.h5
python src/resnet50/evaluate.py ./checkpoints/resnet50/best_model.pt ./data/test-preprocessed.h5
```
