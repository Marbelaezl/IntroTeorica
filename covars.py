import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib as mpl

#ignore divide by zero and NaN, as these are expected for missing data
np.seterr(divide='ignore')
np.seterr(invalid='ignore')
mpl.rcParams.update({'font.size': 14})
os.chdir("./results/")
data=np.genfromtxt("merged-results.txt")
uncert=np.genfromtxt("uncertainties.txt")
#align uncertainties with data
data_aligned=data[:,1:]
tolerances=data[:,0]
#Assign columns to variables to make code more readable


a_plus =data_aligned[:,0]
delta_a_plus = uncert[:,0]

T_plus=data_aligned[:,1]
delta_T_plus=uncert[:,1]

b_plus=data_aligned[:,2]
delta_b_plus=uncert[:,2]

a_minus =data_aligned[:,3]
delta_a_minus = uncert[:,3]

T_minus=data_aligned[:,4]
delta_T_minus=uncert[:,4]

b_minus=data_aligned[:,5]
delta_b_minus=uncert[:,5]

c = data_aligned[:,6]
delta_c=uncert[:,6]

k = data_aligned[:,7]
delta_k =uncert[:,7]

la_plus =data_aligned[:,8]
delta_la_plus=uncert[:,8]

la_minus=data_aligned[:,9]
delta_la_minus=uncert[:,9]

d_plus=data_aligned[:,10]
delta_d_plus=uncert[:,10]

d_minus=data_aligned[:,11]
delta_d_minus=uncert[:,11]

r2=data_aligned[:,12]

#Define effective parameters (intermediate step):
b_plus_eff = b_plus - (la_plus**2/(2*k))
delta_b_plus_eff = delta_b_plus + (2*delta_la_plus*la_plus/k) + (delta_k*la_plus**2/(k**2))

b_minus_eff = b_minus - (la_minus**2/(2*k))
delta_b_minus_eff = delta_b_minus + (2*delta_la_minus*la_minus/k) + (delta_k*la_minus**2/(k**2))

a_ratio = a_minus/a_plus
delta_a_ratio = delta_a_minus/a_plus + (delta_a_plus*a_minus/(a_plus**2)) 


c_eff= (c + (la_plus*la_minus/k))
delta_c_eff = delta_c + (delta_la_plus/la_plus + delta_la_minus/la_minus + delta_k/k) *(la_plus*la_minus/k)

#Finally, calculate the scale-independent parameters (adimensional or with units of Angstrom)
#1,2: dist_ab: Amplitude at which the two leading order terms, aq^2T and bq^4, are equal
dist_ab_plus = np.sqrt(b_plus_eff/(a_plus*T_plus))
delta_dist_ab_plus = 0.5 * (delta_a_plus/a_plus + delta_b_plus_eff/b_plus_eff + delta_T_plus/T_plus) * dist_ab_plus

dist_ab_minus = np.sqrt(b_minus_eff/(a_minus*T_minus))
delta_dist_ab_minus = 0.5 * (delta_a_minus/a_minus + delta_b_minus_eff/b_minus_eff + delta_T_minus/T_minus) * dist_ab_minus

#3,4: dist_bd: Amplitude at which the two positive terms, bq^4 and dq^6, are equal
dist_bd_plus = np.sqrt(d_plus/b_plus_eff)
delta_dist_bd_plus = 0.5 * (delta_d_plus/d_plus + delta_b_plus_eff/b_plus_eff) * dist_bd_plus

dist_bd_minus = np.sqrt(d_minus/b_minus_eff)
delta_dist_bd_minus = 0.5 * (delta_d_minus/d_minus + delta_b_minus_eff/b_minus_eff) * dist_bd_minus

#Ratios that relate the strength of transitions
#5: c_ratio: c/sqrt(b+b-), compares how connected the transitions are to each other
c_ratio = c/np.sqrt(b_plus_eff*b_minus_eff)
delta_c_ratio = delta_c/np.sqrt(b_plus_eff*b_minus_eff)+ 0.5*c*(delta_b_plus_eff/b_plus_eff + delta_b_minus_eff/b_minus_eff)/np.sqrt(b_plus_eff*b_minus_eff)
#6: a_ratio, compares the strenght of one transition and the other
a_ratio = a_minus/a_plus
delta_a_ratio = delta_a_minus/a_plus + delta_a_plus*a_minus/(a_plus**2)

#7,8: Parameters related to the volume coupling
la_plus_norm = la_plus/k
delta_la_plus_norm = delta_la_plus/k + delta_k*la_plus/(k**2)

la_minus_norm = la_minus/k
delta_la_minus_norm = delta_la_minus/k + delta_k*la_minus/(k**2)
#9,10: T+,T-

#Calculate covariance without counting entries that are 0 or NaN
def IsValid(x):
    return np.all(np.array([x!=0, np.logical_not(np.isnan(x)), x!=np.inf, x!=-np.inf]),axis=0)

def cov(x,y):
    n=0
    xy=0
    xsum=0
    ysum=0
    try:
        assert(len(x) == len(y))
    except:
        return [0,0]
    for i in range(0,len(x)):
        if IsValid(x[i]) and IsValid(y[i]):
            xy += x[i]*y[i]
            n+=1
            xsum += x[i]
            ysum += y[i]
    
    return [(xy-(xsum*ysum/n))/n,n ]
#variables to compare against tolerance index
variables=[dist_ab_plus,dist_ab_minus,
           dist_bd_plus,dist_bd_minus,
           c_ratio,a_ratio,
           la_plus_norm,la_minus_norm,
           T_plus,T_minus]
delta_variables=np.array([delta_dist_ab_plus,delta_dist_ab_minus,
                 delta_dist_bd_plus,delta_dist_bd_minus,
                 delta_c_ratio,delta_a_minus,
                 delta_la_plus_norm,delta_la_minus_norm,
                 delta_T_plus,delta_T_minus])
stdevs=[cov(tolerances,tolerances)]

labels=([r'$\sqrt{b_+/a_+}$',r'$\sqrt{b_-/a_-}$',r'$\sqrt{d_+/b_+}$',r'$\sqrt{d_-/b_-}$',
         r'$\frac{c}{\sqrt{b_+b_-}}$',r"$a_-/a_+$",r"$\lambda_+/k$",r"$\lambda_-/k$",r"$T_+$",r"$T_-$"])
correlations=[]
def corr(x,y):
    covariance=cov(x,y)
    varx = cov(x[np.where(IsValid(y))],x[np.where(IsValid(y))])
    vary = cov(y[np.where(IsValid(x))],y[np.where(IsValid(x))])

    return [covariance[0]/np.sqrt(varx[0]*vary[0]),covariance[1]]
corrs=[]
pvals=[]

for i in range(0,len(variables)):
    print("comparison number ", i+1)
    print("Tolerance factor vs ", labels[i])
    corr_current=corr(tolerances,variables[i])
    print(corr_current)
    std_corr = (1-corr_current[0]**2)/np.sqrt(corr_current[1]-1)
    p_value = stats.t.sf(np.abs(corr_current[0]/std_corr), corr_current[1]-1)
    print("P_value: ", p_value)
    if p_value < 0.005:
        correlations.append(True)
    else:
        correlations.append(False)
    corrs.append(corr_current[0])
    pvals.append(p_value)
        
fig,ax=plt.subplots()

for i in range(0, len(variables)):
    if correlations[i]:

        
        error=delta_variables[i]
        fig,ax=plt.subplots()
        ax.errorbar(tolerances, variables[i],yerr=np.abs(error),linestyle="",label=labels[i],color="black",capsize=2,markersize=2,marker="s")
        ax.legend()
        ax.set_xlabel("Goldschmidt tolerance ratio")
        text="r = " + str(round(corrs[i],3)) + ", p = " + str(round(pvals[i],5)) 
        ax.text(x=0.93, y=np.min(variables[i][np.where(IsValid(variables[i]))]), s=text )
#Go back to usual behaviour
np.seterr(divide='warn')
np.seterr(invalid='warn')
temps=np.genfromtxt("T_shift.txt")
names_graphs=["$BaBiO_3$","$Ba_2BiSbO_6$","$Ba_2GdMoO_6$","$La_2CoMnO_6$","$Sr_2CrSbO_6$",
              "$Sr_2CuWO_6$","$Sr_2InTaO_6$","$Sr2NiMoO_6$","$Sr_2ScSbO_6$"]
#Compounds included in the analysis for which data was obtained from tables
fig1,ax1=plt.subplots(figsize=(12,6))
fig2,ax2=plt.subplots(figsize=(12,6))
ticks1=[[],[]]
ticks2=[[],[]]
currents=[0,0]
for i in range(0,int(len(names_graphs)-1)):
    if temps[2*i,0]!=0:
        ax1.errorbar(temps[2*i,0],currents[0],xerr=temps[2*i,1],color="black",marker="^",capsize=4,markersize=10)
        ax1.errorbar(temps[2*i,2],currents[0],xerr=temps[2*i,3],color="blue",marker="s",capsize=2)
        ax1.errorbar(temps[2*i,4],currents[0],xerr=temps[2*i,5],color="red",marker="o",capsize=2)
        ticks1[0].append(currents[0])
        ticks1[1].append(names_graphs[currents[0]])
        currents[0] +=1
    if temps[2*i+1,0]!=0:
         ax2.errorbar(temps[2*i+1,0],currents[1],xerr=temps[2*i+1,1],color="black",marker="^",capsize=4,markersize=10)
         ax2.errorbar(temps[2*i+1,2],currents[1],xerr=temps[2*i+1,3],color="blue",marker="s",capsize=2)
         ax2.errorbar(temps[2*i+1,4],currents[1],xerr=temps[2*i+1,5],color="red",marker="o",capsize=2)
         ticks2[0].append(currents[1])
         ticks2[1].append(names_graphs[currents[1]])
         currents[1] +=1
ax1.set_yticks(ticks1[0],ticks1[1])
ax2.set_yticks(ticks2[0],ticks2[1])
ax1.errorbar([],[],color="black",label="Literature",marker="^")
ax1.errorbar([],[],color="blue",label="Biased estimation",marker="s",capsize=2)
ax1.errorbar([],[],color="red",label="Unbiased estimation",marker="o",capsize=2)

ax2.errorbar([],[],color="black",label="Literature",marker="^",capsize=4)
ax2.errorbar([],[],color="blue",label="Biased estimation",marker="s",capsize=2)
ax2.errorbar([],[],color="red",label="Unbiased estimation",marker="o",capsize=2)
ax1.legend()
ax2.legend()
ax1.set_xlabel("$T_+$ (K)")
ax2.set_xlabel("$T_-$ (K)")