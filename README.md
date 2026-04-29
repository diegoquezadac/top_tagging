# Top Tagging

Three models are implemented: BNN, ResNet50, and ParticleNet. See [`docs/`](docs/) for details on data and model architectures.

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

## Preprocessing

```bash
python src/preprocess.py data/train-public.h5 data/train-preprocessed.h5
python src/preprocess.py data/test-public.h5 data/test-preprocessed.h5
```

See [`docs/data.md`](docs/data.md) for data sources and preprocessing details.

## Training

```bash
python src/bnn/train.py data/train-preprocessed.h5
python src/resnet50/train.py data/train-preprocessed.h5
python src/particle_net/train.py data/train-preprocessed.h5
```

To resume from the last best checkpoint:

```bash
python src/<model>/train.py data/train-preprocessed.h5 --resume
```

## Evaluation

```bash
python src/bnn/evaluate.py checkpoints/bnn/best_model.pt data/test-preprocessed.h5
python src/resnet50/evaluate.py checkpoints/resnet50/best_model.pt data/test-preprocessed.h5
python src/particle_net/evaluate.py checkpoints/particle_net data/test-preprocessed.h5
```
