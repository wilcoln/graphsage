# GraphSage

The goal of this project is to reproduce the
paper "[Inductive Representation Learning on Large Graphs](https://arxiv.org/abs/1706.02216)".

## Folder structure
* `baselines/`
  Code for all the baselines.

* `experiments/`
  Code reproducing all the experiments.
  * `fig2a/`
    Code reproducing the figure 2a of the original paper.
  * `fig2b/`
    Code reproducing the figure 2b of the original paper.
  * `fig3/`
    Code reproducing the figure 3 of the original paper.
  * `table1/`
    Code reproducing the table 1 of the original paper.
* `graphsage/`
  Code related to graphsage models and extensions.
  * `datasets/`
    Code related to datasets and transformations.
  * `models/`
    Code related to models.
  * `layers/`
    Code related to reusable pytorch modules.
  * `samplers/`
    Code related to samplers.
  * `trainers/`
    Custom abstractions to monitor training.
* `examples`/
  Direct application of models on datasets.
* `docs/`
  Auto-Generated & manual code documentation.
* `scripts/`
  Contains bash scripts, this scripts might just be launchers for python scripts defined in the main package. Useful for
  running long experiments for example.

* `data/`
  Auto-generated, contains original or intermediate synthetic data.

* `examples/`
  All the examples, python scripts or notebooks, illustrating the usage of the package.

* `graphsage/`
  Python package containing the main code for this project.

* `results/`
  Auto-generated, For results, e.g. tables (csv files), and plots (images)


## Installation

```bash
# Create the conda environment
conda env create -f env.yml
# Add the environment to your jupyter kernels 
python -m ipykernel install --user --name graphsage 
# Activate the environment
conda activate graphsage
```

### Tests Documentation

To generate documentation, run

```
scripts/makedoc.sh
```

The documentation entrypoint will be generated at `docs/_build/html/index.html`