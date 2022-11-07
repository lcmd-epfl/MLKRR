# Metric Learning for Kernel Ridge Regression (MLKRR)

This repository contains the a python module with the code to execute the MLKRR algorithm. The provided `MLKRR` class has a similar API to standard `scikit-learn` regressors and uses a fit method.

Dependencies are minimal:
- `numpy`
- `scipy`
- `scikit-learn`

## Examples
An example of how to run the algorithm is provided for the QM9 dataset. The example has an additional dependency:
- `pip install git+https://github.com/qmlcode/qml@develop`

To generate the necessary data, first run
`data/generate_reps.py`. 
Then the corresponding representations will be generated in the `data` directory, needed for the `examples/qm9.py` to run.


