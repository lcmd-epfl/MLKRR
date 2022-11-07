# Applies MLKRR on a random subset of size 10 000 of fragments of qm9, to learn their energies. 
# The initial set is split into 10 000 fragments for the learning, and 2 000 for test.

# The data points are FCHL representations (vectors of dimension 720), and the labels are the associated energies (u0) in Ha
# The FCHL data fchls_glob_qm9.npy is in the data dir after running generate_qm9_reps.py 

# At each iteration of the minimization algorithm (possibly multiple steps before making progess),
# the predictions are compared with the labels, appending test_maes, test_rmses, and train_maes, train_rmses.

# Before that, the variance sigma is optimized if learn_sigma is set to True.
# The total number of iterations is equal to the product of shuffles with max_iter_per_shuffle.

# For a data set of 20 000, each iteration takes up to 20 seconds. Optimizing sigma takes an additional 200 seconds.
# It should take around 5 hours for 900 iterations (eg. 30 shuffles, 30 iterations per shuffle).
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
sys.path.insert(0,'..')
import mlkrr

X=np.load("data/fchls_glob_qm9.npy", allow_pickle=True)
m, n=X.shape
data=pd.read_csv("data/data_red.csv")
data=data['u0'].to_numpy() # in Ha 

# takes 12000 random indices, and 2000 random indices among them for data and test
S=12000
indices=np.random.choice(range(m), size=S, replace=False)
indices_test=np.random.choice(range(S), size=2000, replace=False)
mask=np.ones(S, dtype=bool)
mask[indices_test]=0
ind_test=indices[np.logical_not(mask)]
ind_data=indices[mask]
print(len(ind_data), len(ind_test))

# initialize parameters
M = mlkrr.MLKRR(
        size_A=0.5, 
        size_alpha=0.5,
        verbose=True,
        shuffle_iterations=2,
        max_iter_per_shuffle=10, 
        test_data=[X[ind_test],y[ind_test]],
        sigma=55.0,
        learn_sigma=True,
        krr_regularization=1e-9
        )
# run optimization and save object
M.fit(X[ind_data], y[ind_data])
np.save("examples/MLKRR.npy",M)

# plot mean average errors for train and test data 
train_maes=M.train_maes
test_maes=M.test_maes
plt.plot(train_maes)
plt.plot(test_maes)
plt.show()

