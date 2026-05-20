import functions
import numpy as np
import os
import matplotlib.pyplot as plt

name="Ba2Bi2O6"
os.chdir('./exp_data/processed/'+name)
merged=np.genfromtxt(name+"-merged-certain.txt")
#unmerged=np.genfromtxt("Ba2GdMoO6-unmerged_q.txt")
#unmerged[:,1] = (unmerged[:,1]/merged[-1,1]) - 1
prov= merged[np.where(merged[:,3]==0)]

vref=np.min(prov[:,1])
merged[:,1] = (merged[:,1]/vref) - 1 

pressure=np.array([np.zeros_like(merged[:,0])]).T
merged=np.hstack([merged,pressure])

#Swap q1,q2 for Sr2CrSbO6
#merged[:,2] = merged[:,3]
#merged[:,3] = 0


mask=np.array([0,1,1,1,1,1,1,1,1,1,1,1,0])
vec0=np.array([ 1, #a+
               1.96822295e+02, #T+
                2.91040058e+03, #b+
                4.13747797e-01, #a-
                8.36843427e+02, #T-
                2.13386092e+02, # b-
                2.96902563e+02,#c
                7.05136541e+01,#k
                1.82895015e+00, #la+
                1.03970073e+01,#la-
                2.63786548e+02,#d+
                2.51822437e+03, #d-
                0, #kp
                ]) 

# old_estimation=np.load("../../results/Ba2Bi2O6/old.npy")
# vec=old_estimation[-1]
# change,r2,initials,cov = functions.GaussNewtonIter(vec, merged,
#                                    [np.array([]),np.array([]),np.array([])],return_data=True,mask=mask)
#merged[16:,2]=0 #Separate GM5+ from X5+ for La perovskite


old_estimation=np.array([vec0])

finals=[]
r2s=[]
r2prov=[]
covs=[]
estimations=[]



num_iters=50
nruns=1

if nruns !=1:
    initial_estims=[]
    final_estims=[]

os.chdir("../../../results/"+name)

for j in range(0,nruns):
    
    
    if nruns==1:
        vec=vec0
        # try:
        #     old_estimation=np.load("../../results/Ba2Bi2O6/old.npy")
        #     vec=old_estimation[-1]
        #     print("succesfully recovered initial conditions ", vec, "from old.npy")
        # except:
        #     old_estimation=np.array([vec])  
    else:
        vec=np.random.normal(1,0.5,size=np.size(vec0)) #Multiplicative Gaussian noise from central estimation. 
        vec[0]=1
        vec[1:] = vec0[1:] * vec[1:]
        vec[[1,2,3,4,5,7,10,11]] = np.abs(vec[[1,2,3,4,5,7,10,11]])
        initial_estims.append(vec)
    for i in range(0,num_iters):
        print("Iteration ", i)
        print(vec)
        change,r2,data,cov = functions.GaussNewtonIter(vec, merged,
                                           [np.array([]),np.array([]),np.array([])],return_data=True,mask=mask,
                                           model="2t66p")
        r2prov.append(r2)
        if i==0:
             initials=data
        fig,ax=plt.subplots()
        ax.plot(np.array(r2prov),color="black")
        fig.suptitle(r'$R^2$ as a function of iterations for run'+str(j+1))
    
        
        fig2, ax2 = plt.subplot_mosaic([[0, 0],
                                       [1, 2]],
                                       figsize=(9, 6), layout="constrained")
        #fig2,ax2=plt.subplots(2,layout="constrained")
        fig2.suptitle(r'$\epsilon, q_+,q_-$ as a function of temperature')
        ax2[0].set_xlabel("T(K)")
        ax2[1].set_xlabel("T(K)")
        ax2[2].set_xlabel("T(K)")
        ax2[0].set_ylabel(r"$\epsilon$")
        ax2[1].set_ylabel(r"$q_-(\AA, R5^+)$")
        ax2[2].set_ylabel(r"$q_+ (\AA, GM5^+)$")
        for i in range(0,3):
            ax2[i].scatter(merged[:,0],merged[:,i+1],color="black",label="data")
            ax2[i].plot(data[:,0],data[:,i+1],color="red",label="Final guess")
            ax2[i].plot(initials[:,0],initials[:,i+1],color="blue",label="initial guess")
            ax2[i].legend()
        fig.savefig("r2-certain.png")
        fig2.savefig("current-estimation.png")
        
        # if r2 < 0.8:  #Limit change magnitude until fit is good enough
        #     for j in range (0,len(change)):
        #          if np.abs(change[j]) > 0.9*vec[j]:
        #              change *= min(1,np.abs((0.9*vec[j]/change[j])))
        vec=vec+ 0.5*change
        vec[[7,10,11]] =np.abs(vec[[7,10,11]]) #Ensure b+- and k are positive
        if r2 < 0.5:
            vec[[1,3,4]]=np.abs(vec[[1,3,4]])
        estimations.append(vec)
        print(vec)
    if nruns==1:
        estimations =np.vstack([old_estimation,np.array(estimations)])
        np.save("old-certain.npy",estimations)
        fig,ax=plt.subplots()
        for i in [4,5,7,9]:
            ax.plot(np.arange(len(estimations))+1,estimations[:,i])
    elif i==num_iters-1:
        r2s.append(r2prov)
        r2prov=[]
        final_estims.append(vec)
        covs.append(cov)
        
        np.savez(name+".npz", np.array(initial_estims),np.array(final_estims),np.array(covs),np.array(r2s))
        print("run ", j, "finished. Data has been saved")




#functions.GaussNewtonIter(vec, merged, [np.array([]),np.array([]),np.array([])])
