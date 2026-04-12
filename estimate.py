import functions
import numpy as np
import os
os.chdir('./exp_data/processed/Ba2Bi2O6/')
merged=np.genfromtxt("Ba2Bi2O6-merged.txt")
merged[:]
vec = np.array([1, 206, 2174, 0.1, 1000, 1000, 300, 465, 11, 80.7])
#vec=np.array([1.00, 267.626642, 2816.00832, 0.136298552,
# 993.576284, 424.511673, 841.797779, 84.4752746, 5.07102987, 18.2699931])
merged[:,1] = (merged[:,1]/merged[-1,1]) - 1 #Transform epsilon to deviation
for i in range(0,100):
    print("Iteration ", i)
    change = functions.GaussNewtonIter(vec, merged, [np.array([]),np.array([]),np.array([])])
    # for j in range (0,len(change)):
    #     if np.abs(change[j]) > 0.2*vec[j]:
    #         change[j] = 0.2*vec[j]*np.sign(change[j])
    vec=vec+ 0.1*change
    print(vec)
#functions.GaussNewtonIter(vec, merged, [np.array([]),np.array([]),np.array([])])
