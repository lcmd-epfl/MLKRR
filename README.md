# Metric Learning for Kernel Ridge Regression (MLKRR)

- This repository contains the a python module with the code to execute the MLKRR algorithm. The provided `MLKRR` class has a similar API to standard `scikit-learn` regressors and uses a fit method.
- This is an updated version from the algorithm published with the paper "Metric learning for kernel ridge regression: assessment of molecular similarity" by Fabregat, van Gerwen, Haeberle, Eisenbrand and Corminboeuf.
- The previous version can be found in the `paper` branch including a notebook illustrating how to generate the figures in the paper.
- The version provided here is faster and now optimises the hyperparameter sigma as part of the metric learning algorithm.

## Examples
- Download the required dependencies as `pip install requirements.txt`
- To generate the necessary data, first run
`examples/generate_reps.py`. 
- Then the corresponding representations will be generated in the `data` directory, needed for the `examples/qm9.py` to run.

## Citing this work
- If you want to use the MLKRR algorithm, please use the following citation:

```
@article{Fabregat_2022,
doi = {10.1088/2632-2153/ac8e4f},
year = {2022},
volume = {3},
number = {3},
pages = {035015},
author = {Raimon Fabregat and Puck van Gerwen and Matthieu Haeberle and Friedrich Eisenbrand and Clémence Corminboeuf},
title = {Metric learning for kernel ridge regression: assessment of molecular similarity},
journal = {Machine Learning: Science and Technology},
abstract = {Supervised and unsupervised kernel-based algorithms widely used in the physical sciences depend upon the notion of similarity. Their reliance on pre-defined distance metrics—e.g. the Euclidean or Manhattan distance—are problematic especially when used in combination with high-dimensional feature vectors for which the similarity measure does not well-reflect the differences in the target property. Metric learning is an elegant approach to surmount this shortcoming and find a property-informed transformation of the feature space. We propose a new algorithm for metric learning specifically adapted for kernel ridge regression (KRR): metric learning for kernel ridge regression (MLKRR). It is based on the Metric Learning for Kernel Regression framework using the Nadaraya-Watson estimator, which we show to be inferior to the KRR estimator for typical physics-based machine learning tasks. The MLKRR algorithm allows for superior predictive performance on the benchmark regression task of atomisation energies of QM9 molecules, as well as generating more meaningful low-dimensional projections of the modified feature space.}
}
```
