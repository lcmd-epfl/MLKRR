import mlkrr
import pickle
import os
import sklearn.ensemble
import numpy as np
import pandas as pd
import sklearn as sk
import ase
import ase.io as aio
import time
from sklearn.model_selection import train_test_split
import gc
gc.collect()


reps = np.load('data/fchls_glob_qm9.npy', allow_pickle=True)

data = pd.read_csv('data/data_red.csv')
target = data.u0.values # in Ha 

target -= target.mean()
target /= target.std()

train_size = 20000
test_size = 2000

indices_train, indices_test = sk.model_selection.train_test_split(
    np.arange(len(data)), train_size=train_size, test_size=test_size, random_state=0)


X = reps[:, 0:]

w_init = np.diag(np.ones(shape=[X.shape[1]]))

model = mlkrr.MLKRR(
    max_iter=30,
    shuffle_iterations=200,
    verbose='True',
    init=w_init,
    #     init='pca',
    #     l2_constraint=0,
    #     smoothness=0,
    sigma=55,
    #     tol=1e-20,
    #     diag=True,
    vstep=1,
    test_data=[X[indices_test], target[indices_test]],
    krr_regularization=1e-9,
    size1=0.5,
    size2=0.5
)


res = model.fit(X[indices_train], target[indices_train])


np.save('models/trained_mlkrr.npy', model)
