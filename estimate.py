import functions
import numpy as np
import os
os.chdir('./exp_data/processed/Ba2Bi2O6/')
merged=np.genfromtxt("Ba2Bi2O6-merged.txt")
merged[:]
vec = ([1, 120, 200, 1.0, 820, 300, -10, 500, 0.1, 0.2])
merged[:,1] = (merged[:,1]/merged[-1,1]) - 1 #Transform epsilon to deviation
for i in range(0,3):
    vec = vec - functions.GaussNewtonIter(vec, merged, [np.array([]),np.array([]),np.array([])])
    print(vec)
#functions.GaussNewtonIter(vec, merged, [np.array([]),np.array([]),np.array([])])
