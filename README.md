# Metric Learning for Kernel Ridge Regression (MLKRR)

This repository contains the a python module with the code to execute the MLKRR algorithm. The provided `MLKRR` class has a similar API to standard `scikit-learn` regressors and uses a fit method.

Install the required dependencies from .

First, the original representations (to be transformed by the MLKRR algorithm) need to be loaded:
1. Run `src/generate_data.py` to save the representations to the `data/` dir.
2. Run `src/mlearn.py` on a powerful machine or cluster, since this takes several days to run.
3. Use the saved models in the `models` directory in the notebook `3-Reproduce_paper_figures.ipynb` to generate the figures as in the paper.



