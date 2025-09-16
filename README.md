# Top Tagging

## Python environment

Run the following commands:

```bash
uv venv
source .venv/bin/activate
uv sync
```

Additionally, considering the available hardware either install tensorflow-metal or tensorflow gpu:

```bash
uv add tensorflow-metal
```

## Data gathering

Download the datasets from [OpenData](https://opendata.cern.ch/record/15013).

## Data preprocessing

Activate the python environment and run the following commands:

```bash
python src/preprocess.py /home/raquel/data/test-public.h5 ./data/test-preprocessed.h5

python src/preprocess.py data/train-public.h5 data/train-preprocessed.h5

python src/preprocess.py data/test-public.h5 data/test-preprocessed.h5
```

> TODO: Add warning about time and space.

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
