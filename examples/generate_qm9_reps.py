import qml 
import pandas as pd
import numpy as np

data = pd.read_csv("../data/data_red.csv")
indices = np.arange(len(data))
fnames = ["../data/qm9/qm9_"+str(idx)+".xyz" for idx in indices]

mols = [qml.Compound(x) for x in fnames]
ncharges = [x.nuclear_charges for x in mols]
atomtypes = np.unique(np.concatenate([x.nuclear_charges for x in mols]))
max_natoms = np.max([len(x.nuclear_charges) for x in mols])

reps = np.array([qml.representations.generate_fchl_acsf(x.nuclear_charges, x.coordinates, pad=max_natoms,
    elements=atomtypes) for x in mols])
reps_global = np.sum(reps, axis=1)
print(reps_global.shape)

np.save('../data/fchls_glob_qm9.npy', reps_global)
