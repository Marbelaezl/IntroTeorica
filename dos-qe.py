import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir("./qe/dos")

files=["Fm-3m-dos.dat","I4_m-dos(experimental).dat","I4_m-dos2.dat","P2-1-dos3.dat"]
Ef=[11.864, 10.903, 11.782, 11.065] #Fermi energies of the files, in the order that they appear in
k=8.617333262e-05 #Boltzmann constant in eV/k for fermi-dirac distribution
datasets=[]
interval=0.01 #Interval betweed DOS measurements. Could be read from file, but it is always the same anyways
#Read data
for i in files:
    datasets.append(np.genfromtxt(i))
T0=0
Tend=1200
n=5000
fig,ax=plt.subplots()
print(datasets)
for j in range(0,len(datasets)):
    prov=np.repeat(datasets[j][None,:], n, axis=0) #make array with n copies of dataset
    print(prov[:,:,0])
    temps = np.linspace(T0,Tend,n)
    prov[:,:,3] = 1/(np.exp(prov[:,:,0]/(k*temps[:,None]))+1)
    print(prov[:,:,0]/(k*temps[:,None])+1)
    print("dos:")
    print(prov[:,:,3])
    Eup = np.sum(prov[:,:,3]*(prov[:,:,0]+Ef[j])*prov[:,:,1],axis=1)*interval
    for k in range(0,prov.shape[1]):
        ax.plot(prov[k,:,0],prov[k,:,3])
    