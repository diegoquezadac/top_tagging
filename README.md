# Top Tagging

Three models are implemented: BNN, ResNet50, and ParticleNet. See [`docs/`](docs/) for model architecture details.

## Setup

```bash
uv venv
source .venv/bin/activate
uv sync
```

Install the backend matching your hardware:

```bash
uv add tensorflow-metal       # Apple Silicon
uv add tensorflow[and-cuda]   # NVIDIA GPU
```

<details>
<summary>Using pip instead</summary>

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install tensorflow-metal   # or tensorflow[and-cuda]
```
</details>

## 1. Data

Download `train-public.h5` and `test-public.h5` from [ATLAS Top Tagging Open Data](https://opendata.cern.ch/record/15013) and place them in `data/`. See [`docs/data.md`](docs/data.md) for details.

## 2. EDA

```bash
python scripts/eda.py data/test-public.h5
```

Plots are saved to `figures/eda/`. Pass `--max-jets N` to run on a subset. If you have already preprocessed the data, pass `--preprocessed` to also generate the preprocessed jet image plots:

```bash
python scripts/eda.py data/test-public.h5 --preprocessed data/test-preprocessed.h5
```

## 3. Preprocessing

For full training and evaluation, preprocess the official train and test files separately:

```bash
python scripts/preprocess.py data/train-public.h5 data/
python scripts/preprocess.py data/test-public.h5 data/
```

For quick local testing, generate a small sample first and split it:

```bash
python scripts/sample.py data/train-public.h5 data/ --n 100000
python scripts/preprocess.py data/train-public_sample.h5 data/ --split 0.8
# → data/train-preprocessed.h5 + data/test-preprocessed.h5
```

## 4. Training

```bash
python src/bnn/train.py data/train-preprocessed.h5
python src/resnet50/train.py data/train-preprocessed.h5
python src/particle_net/train.py data/train-preprocessed.h5
```

To resume from the last best checkpoint:

```bash
python src/<model>/train.py data/train-preprocessed.h5 --resume
```

## 5. Evaluation

```bash
python src/bnn/evaluate.py checkpoints/bnn/best_model.pt data/test-preprocessed.h5
python src/resnet50/evaluate.py checkpoints/resnet50/best_model.pt data/test-preprocessed.h5
python src/particle_net/evaluate.py checkpoints/particle_net data/test-preprocessed.h5
```
