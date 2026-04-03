import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

#Compounds included in the analysis for which data was obtained from graphs
names_graphs=["Ba2Bi2O6","Ba2BiSbO6","La2CoMnO6","Sr2CrSbO6",
             # "fake_dir_test",
              "Sr2CuWO6","Sr2InTaO6","Sr2NiMoO6","Sr2ScSbO6"]
#Compounds included in the analysis for which data was obtained from tables
names_tables=["Ba2GdMoO6"]
#data from current compound
current=[]
#data from all compounds
alldata=[]
#Iterate over al directories and parse files into numpy arrays
dirname="./preprocessed/"
for i in names_graphs:
    dirname= dirname+i
    try:
        os.chdir(dirname)
        current.append(np.genfromtxt("q+-q--T-"+i+".csv",delimiter=",",comments="#"))
        current.append(np.genfromtxt("epsilon-T-"+i+".csv",delimiter=",",comments="#"))
        alldata.append(current)
        current=[]
        print("Loaded data from ", dirname)
    except:
        print("Error while parsing data from" , dirname)
        print("Check whether the directory exists and verify naming conventions")
    
    dirname=("../")
os.chdir("..") 
#Define useful functions to merge observations
#merge_epsilon: used to calculate cell volume by merging a, b, and c observations
def merge_epsilon (arr,tolerance=3,max_cycles=5,warn=True):
    "Merge data from multiple entries, eliminating inconsistent entries."
    "arr is an array of nx2 numpy arrays which share a common parameter (like temperature)"
    "and that the observations correspond to a consistent value of the common parameter (within tolerance)"
    "max_cycles is the maximum number of data-selection cycles to perform before giving up on each observation"
    #Check if all arrays have the same size. 
    length=len(arr[0])
    flag=False
    for i in arr:
        if (len(i) != length):
            length=min(length, len(i))
            #If one of the arrays has a different length, trigger flag
            flag=True
    if flag and warn:
        print("Warning: Arrays have inconsistent length. This may result in loss of data")
    result=np.zeros((length,len(arr)+1)) #result matrix has one line per observation and one column per variable
    indices=np.zeros((len(arr)),dtype=int) #index of current element on each array
    #Iterate until reaching the end of one array
    data_index=0
    cycles=0 
    while (indices < length).all():
        consistent=True
        shared_val=0
        for i in range(0,len(arr)):
            shared_val += arr[i][indices[i],0] #take shared element from each array and calculate average
        shared_val = shared_val/len(arr)
        for i in range(0,len(arr)):
            #Check for consistency.
            #If the value of one of the indices has a shared value too low, increase it.
            if ((shared_val - arr[i][indices[i],0]) > tolerance):
                indices[i] = min(indices[i]+1, length)
                consistent=False
            #If the value is too high, try previous observation
            elif ( arr[i][indices[i],0] - shared_val) > tolerance:
                #prevent from going back to previous observations
                indices[i] = max(data_index,indices[i]-1)
                consistent=False  
        #If no errors were detected, add datum and reset count of unsuccesful cycles
        if consistent:
            result[data_index,0]=shared_val
            for i in range(0,len(arr)):
                result[data_index,i+1] = arr[i][indices[i],1]
            data_index +=1
            indices +=1
            cycles=0
        else:
            #if no consistent observation was found, add one to cycle counter
            cycles += 1
        #Add 1 to all indices
        if cycles >= max_cycles:
            data_index+=1 #To prevent data from going back uniformly
            indices+=1
            cycles=0
        consistent=True
    return result

#Combine volume distortion data by breaking the array at selected points
breakpoints_epsilon=[
    #Ba2Bi2O6
    [[0,37,78,120], #monoclinic phase
     [120,137,154] #hexagonal phase
     #cubic phase does not need merging
     ],
    #Ba2BiSbO6
    [
     [0,20,39,58], #monoclinic phase
     [59, 95, 125] #hexagonal phase
     #cubic phase does not need merging
     ],
    #La2CoMnO6
    [
     [0,13,26,39], #monoclinic phase
     [39,83,127]#hexagonal
     #cubic phase does not need merging
     ],
    #Sr2CrSbO6
    [
     [0,22,44] #only tetragonal phase
     ],
    #Sr2CuWO6
    [
     [0,33,66] #only tetragonal?
     ],
    #Sr2InTaO6
    [
     [0,17,34,51], #monoclinic
     [51,62,73]#tetragonal
     #cubic does not need merging
     ],
    #Sr2NiMoO6
    [
     [0,9,18] #tetragonal
     #cubic does not need merging
     ],
    #Sr2ScSbO6
    [
     [0,10,20,30], #monoclinic P21/n
     [30,46,62,78], #monoclinic I2/m
     [78,87,96] #tetragonal
     #cubic does not need merging
     ]
    ]
data_epsilon=[]

#i indexes the entry in breakpoints_epsilon
for i in range(0,len(breakpoints_epsilon)):
    print(names_graphs[i])
    #j indexes the phase within a material
    for j in range(0,len(breakpoints_epsilon[i])):
        #the first breakpoint indicates the start of data (and is inclusive)
        current_phase=[alldata[i][1][breakpoints_epsilon[i][j][0]:breakpoints_epsilon[i][j][1]]]
        #for the rest of the breakpoints, start at previous +1 and end at the next one
        for k in range(1, len(breakpoints_epsilon[i][j])-1):
            current_phase.append(alldata[i][1][breakpoints_epsilon[i][j][k]:breakpoints_epsilon[i][j][k+1]])
        # print(merge_epsilon(current_phase))
        data_epsilon.append(merge_epsilon(current_phase))

print("Applying manual correction for 'lost data' warnings... ")
#manual processing of mising data
#Ba2Bi2O6, T = 120
data_epsilon[0][11] = merge_epsilon([alldata[0][1][11,None], alldata[0][1][50,None], alldata[0][1][92,None] ],tolerance=5,warn=False)[0]
#Ba2Bi2O6, 225 < T < 372
data_epsilon[0][21:29] = merge_epsilon([alldata[0][1][21:29], alldata[0][1][61:70], alldata[0][1][103:112]],tolerance=5,warn=False)
#Ba2Bi2O6, 415 < T < 450
data_epsilon[0][-4:] = merge_epsilon([alldata[0][1][33:37], alldata[0][1][74:78], alldata[0][1][116:120]],tolerance=5,warn=False)
#Ba2BiSbO6, T = 225
data_epsilon[2][-3] = merge_epsilon([alldata[1][1][17,None], alldata[1][1][36,None], alldata[1][1][55,None]],tolerance=5,warn=False)
#Ba2BiSbO6, 440 < T < 490 
data_epsilon[3][-5:] =  merge_epsilon([alldata[1][1][89:95], alldata[1][1][120:125]],warn=False)
#Check that no missing data (all zeros) remain

flag=True
for i in range(0, len(data_epsilon)):
    if np.all(((data_epsilon[i])==0),axis=-1).any():
        print("Error: Some data was detected as missing in block ",i)
        flag=False
if flag:
    print("Epsilon data parsed succesfully")

#transform temperature to K and lengths to AA.
#Ba2Bi2O6, monoclinic
data_epsilon[0][:,1] *= 2
data_epsilon[0][:,2] *= np.sqrt(2)
data_epsilon[0][:,3] *= np.sqrt(2)
#hexagonal
data_epsilon[1][:,1] *= np.sqrt(12)
data_epsilon[1][:,2] *= np.sqrt(2)
#Ba2BiSbO6, monoclinic
data_epsilon[2][:,1] *= 2
data_epsilon[2][:,2] *= np.sqrt(2)
data_epsilon[2][:,3] *= np.sqrt(2)
#hexagonal
data_epsilon[3][:,1] *= np.sqrt(12)
data_epsilon[3][:,2] *= np.sqrt(2)
#La2CoMnO6, monoclinic
data_epsilon[4][:,1] *= 1/np.sqrt(2)
data_epsilon[4][:,2] *= 1/np.sqrt(2)
#hexagonal 
data_epsilon[5][:,1] *= np.sqrt(3)
data_epsilon[5][:,2] *= 1/np.sqrt(2)
#Sr2CrSbO6, tetragonal
data_epsilon[6][:,1] *= 1/np.sqrt(2)
#Sr2CuWO6
data_epsilon[7][:,0] += 273.15
#Sr2InTaO6
data_epsilon[8][:,0] = 10*data_epsilon[8][:,0] +273.15
data_epsilon[9][:,0] = 10*data_epsilon[9][:,0] +273.15
#Sr2InMoO6
data_epsilon[10][:,1] *= 1/np.sqrt(2)
#Sr2ScSbO6, P21/n
data_epsilon[11][:,2] *= 1/np.sqrt(2)
data_epsilon[11][:,3] *= 1/np.sqrt(2)
# I2/m
data_epsilon[12][:,2] *= 1/np.sqrt(2)
data_epsilon[12][:,3] *= 1/np.sqrt(2)
#I4/m
data_epsilon[13][:,1] *= 1/np.sqrt(2)

#NOTE: AT THIS STAGE, ONLY CELL PARAMETERS AND TEMPERATURES HAVE BEEN CALCULATED
#PROCESSING BEYOND THIS POINT WILL OVERWRITE INFORMATION

#Calculate volume in definitive array
volumes=[]
#Ba2Bi2O6, Ba2BiSbO6 and La2CoMnO6 (monoclinic -> hexagonal -> cubic)
for i in range(0,3):
#number of data points: monoclinic + tetragonal + cubic part
    num_data = len(data_epsilon[2*i])+len(data_epsilon[2*i+1])+len(alldata[i][1][breakpoints_epsilon[i][1][2]:])
    current_vol = np.zeros((num_data,2))
    current_vol[:,0]=np.hstack([data_epsilon[2*i][:,0],data_epsilon[2*i+1][:,0],alldata[i][1][breakpoints_epsilon[i][1][2]:,0]])
    #calculate volume for each phase
    #NOTE: NO CORRECTION RELATED TO THE MONOCLINIC ANGLE IS APPLIED IN ANY CASE.
    #THE MAXIMUM RELATIVE DISPLACEMENT FOR THE WHOLE DATASET IS 1-COS(0.01º) = 1.5*10^-8
    v1 = (data_epsilon[2*i][:,1] * data_epsilon[2*i][:,2] * data_epsilon[2*i][:,3])
    v2 = (data_epsilon[2*i+1][:,1] * data_epsilon[2*i+1][:,2]**2 *np.sqrt(3)/2)
    v3 = alldata[i][1][breakpoints_epsilon[i][1][2]:,1]**3
    #Vivided by number of formula units per cell
    if(i==2):
        v1=v1/2
        v2=v2/3
        v3=v3/4
    else:
        v1=v1/4
        v2=v2/6
    current_vol[:,1] = np.hstack([v1,v2,v3])
    volumes.append(current_vol)
#Sr2CrSbO6 Sr2CuWO6: tetragonal
for i in [6,7]:
    current_vol = np.zeros((len(data_epsilon[i]),2))
    current_vol[:,0] = data_epsilon[i][:,0]
    current_vol[:,1] = data_epsilon[i][:,1]**2 * data_epsilon[i][:,2]/2
    volumes.append(current_vol)
#Sr2InTaO6: monoclinic -> tetragonal -> cubic
num_data = len(data_epsilon[8])+len(data_epsilon[9])+len(alldata[5][1][breakpoints_epsilon[5][1][2]:])
current_vol = np.zeros((num_data,2))
current_vol[:,0]=np.hstack([data_epsilon[8][:,0],data_epsilon[9][:,0],10*alldata[5][1][breakpoints_epsilon[5][1][2]:,0]+273.15])
v1 = data_epsilon[8][:,1] * data_epsilon[8][:,2] * data_epsilon[8][:,3]
v2= data_epsilon[9][:,1] **2 * data_epsilon[9][:,2]
v3 =alldata[5][1][breakpoints_epsilon[5][1][2]:,1]**3
current_vol[:,1] = np.hstack([v1,v2,v3])
current_vol[:,1] *= 2
volumes.append(current_vol)
#Sr2NiMoO6: Tetragonal -> cubic
num_data = len(data_epsilon[10])+len(alldata[6][1][breakpoints_epsilon[6][0][2]:])
current_vol = np.zeros((num_data,2))
current_vol[:,0]=np.hstack([data_epsilon[10][:,0],alldata[6][1][breakpoints_epsilon[6][0][2]:,0]])
v1 = data_epsilon[10][:,1] **2 * data_epsilon[10][:,2]/2
v2 = alldata[6][1][breakpoints_epsilon[6][0][2]:,1]**3/4
current_vol[:,1] = np.hstack([v1,v2])
volumes.append(current_vol)
#Sr2ScSbO6 monoclinic -> monoclinic -> tetragonal -> cubic
num_data = len(data_epsilon[11])+len(data_epsilon[12])+len(data_epsilon[13]) +len(alldata[7][1][breakpoints_epsilon[7][2][2]:])
current_vol = np.zeros((num_data,2))
current_vol[:,0]=np.hstack([data_epsilon[11][:,0],data_epsilon[12][:,0],data_epsilon[13][:,0],alldata[7][1][breakpoints_epsilon[7][2][2]:,0]])
v1 = data_epsilon[11][:,1] * data_epsilon[11][:,2] * data_epsilon[11][:,3]
v2=data_epsilon[12][:,1] * data_epsilon[12][:,2] * data_epsilon[12][:,3]
v3= data_epsilon[13][:,1] **2 * data_epsilon[13][:,2]
v4 =alldata[7][1][breakpoints_epsilon[7][2][2]:,1]**3/2
current_vol[:,1] = np.hstack([v1,v2,v3,v4])/2
volumes.append(current_vol)

#TODO: processing of order parameters
def merge(qp,qm,v,tolerance=10):
    "merge volumetric and order parameter data. into as few observations as possible"
    "The output are two arrays: the first one contains merged data, while the second contains unmerged data"
    "This function assumes that all arrays are ordered from low to high temperature"
    merged=[]
    other =[[],[],[]]
    #Current element in each array. 
    indices =[0,0,0]
    #pad the array with (inf,inf) to avoid accessing out-of range memory
    #This looks weird but it makes sense given that indices only go forward if the temperature is either right or too low
    #also, using for i in [qp,qm,v] does not work because it affects only the copies of these arrays inside the list used for indexing
    qp = np.vstack([qp,np.array([[np.inf,np.inf]])])
    qm = np.vstack([qm,np.array([[np.inf,np.inf]])])
    v = np.vstack([v,np.array([[np.inf,np.inf]])])
    for i in range(0,len(v)+len(qp)+len(qm)): #Worst-case scenario: all data is disjoint
        #This approach does not assume that V is the largest dataset (even though it usually is) and can handle observations of order parameters without 
        #an associated volume observation
        with np.errstate(invalid='ignore'): #ignore warnings caused by reaching end of list (infinity technically does not support '-')
            dt1 = np.abs(qp[indices[0],0]- v[indices[2],0])
            dt2 = np.abs(qm[indices[1],0]- v[indices[2],0])
        if (dt1 < tolerance and dt2 < tolerance):
            #if consistent observation is found, record it in merged array
            merged.append([(qp[indices[0],0] +qm[indices[1],0] + v[indices[2],0])/3, v[indices[2],1], qp[indices[0],1], qm[indices[1],1]])
            #go to next element of all arrays
            indices[0] = min(indices[0]+1, len(qp)-1)
            indices[1] = min(indices[1]+1, len(qm)-1)
            indices[2] = min(indices[2]+1, len(v)-1)
        elif dt1 < tolerance:
            #if only 1 of the order parameters has an entry, assume that the other is 0
            merged.append([(qp[indices[0],0] + v[indices[2],0])/2,   v[indices[2],1], qp[indices[0],1], 0])
            indices[0] = min(indices[0]+1, len(qp)-1)
            indices[2] = min(indices[2]+1, len(v)-1)
        elif dt2 < tolerance:
            merged.append([(qm[indices[1],0] + v[indices[2],0])/2, v[indices[2],1],0, qm[indices[1],1],  ])
            indices[1] = min(indices[1]+1, len(qm)-1)
            indices[2] = min(indices[2]+1, len(v)-1)
        else:
            #if both observations are out of tolerance, check if the temperatures are lower than expected, and add to unmerged if they are
            if qp[indices[0],0] < v[indices[2],0]:
                other[0].append(qp[indices[0]])
                indices[0] = min(indices[0]+1, len(qp)-1)
            elif qm[indices[1],0] < v[indices[2],0]:
                other[1].append(qm[indices[1]])
                indices[1] = min(indices[1]+1, len(qm)-1)
            else:
                #Alternatively, it is possible that one volume observation was missing
                other[2].append(v[indices[2]])
                indices[2] = min(indices[2]+1, len(v)-1)
    
    return [np.array(merged), other]

# Convert degree to AA
alldata[0][0][:,1] = np.sin(alldata[0][0][:,1]*np.pi/180)*2.14 
alldata[1][0][:,1] = np.sin(alldata[1][0][:,1]*np.pi/180)*2.145
alldata[3][0][:,1] = np.sin(alldata[3][0][:,1]*np.pi/180)*1.99
alldata[4][0][:,1] = np.sin(alldata[4][0][:,1]*np.pi/180)*2.11
alldata[6][0][:,1] = np.sin(alldata[6][0][:,1]*np.pi/180)*1.96

breakpoints_q =[50,None,55,None,None,28,None,None]


alldata[3][0]=alldata[3][0][::-1] #Sr2CrSbO6 is ordered from low to high temperature
alldata[4][0][:,0] +=273.15 #Sr2CuWO6 has T in celsius
alldata[5][0][:,0] +=273.15 #Sr2InTaO6 has T in celsius
for i in range(0,len(breakpoints_q)):
    if breakpoints_q[i] is None:
        result=merge(np.array([[0,0]]),alldata[i][0][::-1],volumes[i])
    else:
        result = merge(alldata[i][0][:breakpoints_q[i]:-1], alldata[i][0][breakpoints_q[i]::-1],volumes[i])
    dirname = "../processed/" + names_graphs[i]
    Path(dirname).mkdir(parents=True, exist_ok=True)
    fname=dirname+"/"+names_graphs[i]
    np.savetxt(fname+"-merged.txt", result[0],fmt='%4.6f', header="T(K) V(AA^3) q1(AA) q2(AA)")
    print("Writing processed data in directory", dirname)
    np.savetxt(fname+"-unmerged_T.txt", np.array(result[1][2])[np.where(np.array(result[1][2])[:,0] != np.inf)],fmt='%2.6f', header="T(K) V(AA^3)")
    #3 data sources have more than one unmerged observation
    if len(np.array(result[1][1])!=0):
        np.savetxt(fname+"-unmerged_q.txt", np.array(result[1][1])[np.where(np.array(result[1][1])[:,0] != np.inf)],fmt='%2.6f', header="T(K) q(AA)")
data_table = np.genfromtxt("./"+names_tables[0]+"/"+names_tables[0]+".csv",delimiter=" ") #not a real csv :/
data_table[:,1] = data_table[:,1]*data_table[:,2] *data_table[:,3]/2
data_table[[0,1],1] *= 0.5
data_table[:,2] = np.sin(data_table[:,4]*np.pi/180)* 2.119
data_table[:,3] = np.sin(data_table[:,5]*np.pi/180)* 2.119
dirname = "../processed/" + names_tables[0]
Path(dirname).mkdir(parents=True, exist_ok=True)
print("Writing processed data in directory", dirname)
fname=dirname+"/"+names_tables[0]
np.savetxt(fname+"-merged.txt", result[0],fmt='%4.6f', header="T(K) V(AA^3) q1(AA) q2(AA)")
print("All data processed successfully")
    
    