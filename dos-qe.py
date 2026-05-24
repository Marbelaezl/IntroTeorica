import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
os.chdir("./qe/dos")
mpl.rcParams.update({'font.size': 10})
files=["Fm-3m-dos.dat","I4_m-dos(experimental).dat","I4_m-dos2.dat","P2-1-dos3.dat"]
labels=[r"$Fm\bar 3 m$", "I4/m (two-tilt)","I4/m (one-tilt)", r"$P2_1$"]
Ef=[11.864, 10.903, 11.782, 11.065] #Fermi energies of the files, in the order that they appear in
n_cells=[4,2,1,1]
colors=["black","red","blue","green"]
kb=8.617333262e-05 #Boltzmann constant in hartree/k for fermi-dirac distribution
tf= 1.37e+05
datasets=[]
interval=0.01 #Interval betweed DOS measurements. Could be read from file, but it is always the same anyways
#Read data
for i in files:
    datasets.append(np.genfromtxt(i))
T0=0
Tend=13700
n=5000
Es=[]
print(datasets)
fig,ax=plt.subplots()
for j in range(0,len(datasets)):
    temps = np.linspace(T0,Tend,n)
    eup=np.zeros_like(temps)
    edown=np.zeros_like(temps)
    for k in range(0,n):
        fermi_dist = 1/(np.exp(datasets[j][:,0]/(kb*temps[k]))+1)
        eup[k] = np.sum((datasets[j][:,0]+Ef[j])*datasets[j][:,1]*fermi_dist)*interval/n_cells[j]
        edown[k] = np.sum((datasets[j][:,0]+Ef[j])*datasets[j][:,2]*fermi_dist)*interval/n_cells[j]
    Es.append(eup[0]+edown[0])
    ax.plot(temps/tf,eup,color=colors[j],linestyle="--")
    ax.plot(temps/tf,edown,color=colors[j],linestyle="--")
    ax.plot(temps/tf, (edown+eup)/2,color=colors[j],label=labels[j])
   
ax.plot([990/tf,990/tf],[92,109], linestyle="--",color="orangered")
ax.plot([1270/tf,1270/tf],[92,109], linestyle="--",color="purple")
ax.legend()
ax.set_xlabel("$T/T_f$")
ax.set_ylabel("E (eV)")