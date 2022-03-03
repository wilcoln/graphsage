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

### Create the conda environment by running
```
conda env create -f env.yml
conda activate graphsage
python -m ipykernel install --user --name graphsage
```

After installing, every time you want to work with this project run `conda activate graphsage` and after you 
finish, run `conda deactivate`.

### Package installation
To install the package go to the root of this directory and run
```
conda develop graphsage/
```
Now every time you want to run some piece of code you can import it from `graphsage`.
Alternatively, you can add the `graphsage` package to your python path, by adding these line at the top of 
your python script or jupyter notebook.
```python
import sys
sys.path.insert(0, '/path/to/graphsage')
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