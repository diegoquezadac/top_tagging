# Top Tagging

## Python Environment

Start by installing [uv](https://docs.astral.sh/uv/getting-started/installation/): a fast Python package and project manager.

Then, create and activate the Python environment:

```bash
uv venv
source .venv/bin/activate
```

Finally, install the dependencies:

```bash
uv pip install -r pyproject.toml 
```


## Training

Simply run the following command:

```bash
python src/restnet50/train.py
```