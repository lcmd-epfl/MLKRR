import numpy as np
import pandas as pd
import qml

data = pd.read_csv('data/data_red.csv')
indices = np.arange(len(data))
fnames = ["data/qm9/qm9_"+str(idx)+".xyz" for idx in indices]
mols = [qml.Compound(x) for x in fnames]
ncharges = [x.nuclear_charges for x in mols]
atomtypes = np.unique(np.concatenate([x.nuclear_charges for x in mols]))
max_natoms = np.max([len(x.nuclear_charges) for x in mols])
reps = np.array([qml.representations.generate_fchl_acsf(x.nuclear_charges, x.coordinates, pad=max_natoms,
    elements=atomtypes) for x in mols])
reps = np.sum(reps, axis=1)
np.save('data/fchls_glob_qm9.npy', reps)
