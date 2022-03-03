# GraphSage

The goal of this project is to reproduce the paper "[Inductive Representation
Learning on Large Graphs](https://arxiv.org/abs/1706.02216)".

## Folder structure

* `docs/`
Auto-Generated & manual code documentation.

* `data/`
Contains data original & intermediate synthetic data.

* `notebooks/`
All the notebooks, avoid defining functions here.

* `graphsage/`
Python package containing the main code for this research.

* `results/`
For results, e.g. tables (csv files), and plots (images)

* `scripts/`
Contains bash scripts, this scripts might just be launchers for python scripts defined in the main package.
Useful for running long experiments for example.

## Installation

```bash
# Create the conda environment
conda env create -f env.yml
# Add the environment to your jupyter kernels 
python -m ipykernel install --user --name graphsage 
# Activate the environment
conda activate graphsage
```

After installing, every time you want to work with this project run `conda activate graphsage` and after you 
finish, run `conda deactivate`.

### Usage
To use this package in a python module or notebook, you just have to include the `graphsage` package to your python 
path,
by adding these line at the top of 
your python script or jupyter notebook.
```python
import sys
sys.path.insert(0, '/path/to/graphsage')
```
You'll then be able to import things from the package, such as:
```python
from graphsage import datasets, utils
```

### Tests and documentation
To run tests, run
```
pytest --pyargs graphsage/tests/*
```
To generate documentation, run
```
scripts/makedoc.sh
```
The documentation entrypoint will be generated at `docs/_build/html/index.html`