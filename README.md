# Metric Learning for Kernel Ridge Regression (MLKRR)

This repository contains the a python module with the code to execute the MLKRR algorithm. The provided `MLKRR` class has a similar API to standard `scikit-learn` regressors and uses a fit method.
This is an updated version from the algorithm published with the paper "Metric learning for kernel ridge regression: assessment of molecular similarity" by Fabregat, van Gerwen, Haeberle, Eisenbrand and Corminboeuf.
The previous version can be found in the `paper` branch including a notebook illustrating how to generate the figures in the paper.
The version provided here is faster and now optimises the hyperparameter sigma as part of the metric learning algorithm.

## Examples
Download the required dependencies as `pip install requirements.txt`
To generate the necessary data, first run
`data/generate_reps.py`. 
Then the corresponding representations will be generated in the `data` directory, needed for the `examples/qm9.py` to run.


