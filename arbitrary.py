#arbitrary.py - macroscopic behaviour of the system for arbitrary coefficients
import numpy as np
import matplotlib.pyplot as plt
import functions

vec=([-1,1,0.3,-1.5,5,0.6,-0.2,0.25,0.1,-0.2])
results=[[0,0,0,0]]
status=[0]
params,couplings=functions.GenParams(vec)
index=0
for i in np.linspace(0, 15,1000):
    print("T = ",i)
    variables=np.array([np.max([np.random.rand()*100,results[index][0]]),np.max([np.random.rand()*100,results[index][1]]),i,np.max([np.random.rand()*100,results[index][3]])])
    prov=functions.MinimizeEnergy(params, variables, couplings,mask=np.array([1,1,0,1]),verbose=False,cycles=2000,delta=1e-16)
    results.append(prov[1])
    status.append(prov[0])
    index+=1
fig, ax = plt.subplot_mosaic([[0, 1],
                               [2, 2]],
                              figsize=(9, 6), layout="constrained")
results=np.array(results)
status=np.array(status)
errors=results[np.where(status!=0)]
ax[0].scatter(results[:,2],results[:,0])
ax[0].scatter(errors[:,2],errors[:,0],color="red")
ax[0].set_ylabel(r'$q_+$')
ax[1].set_ylabel(r'$q_-$')
ax[2].set_ylabel(r'$\epsilon$')
for i in range(0,3):
    ax[i].plot([vec[1],vec[1]], [-5,5],linestyle="--",color="blue")
    ax[i].plot([vec[4],vec[4]], [-5,5],linestyle="--",color="blue")
    
ax[2].scatter(results[:,2],results[:,3])
ax[2].scatter(errors[:,2],errors[:,3],color="red")
ax[1].scatter(results[:,2],results[:,1])
ax[1].scatter(errors[:,2],errors[:,1],color="red")
fig,ax=plt.subplots()
ax.scatter(results[:,0],results[:,3])
ax.scatter(errors[:,0],errors[:,3],color="red")
energy=np.zeros(len(results))
for i in range(0,len(results)):
    energy[i] = functions.FreeEnergy(params,results[i],couplings)
fig,ax=plt.subplots()
ax.scatter(results[:,2],energy)