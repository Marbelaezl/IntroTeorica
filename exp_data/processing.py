import os
import numpy as np
import matplotlib.pyplot as plt

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
    
#Define useful functions to merge observations
#merge_epsilon: used to calculate cell volume by merging a, b, and c observations
def merge_epsilon (arr,tolerance=3,max_cycles=5):
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
    if flag:
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
#print(data_epsilon)
#manual processing of mising data
#Ba2Bi2O6, T = 120
data_epsilon[0][11] = merge_epsilon([alldata[0][1][11,None], alldata[0][1][50,None], alldata[0][1][92,None] ],tolerance=5)[0]
#Ba2Bi2O6, 225 < T < 372
data_epsilon[0][21:29] = merge_epsilon([alldata[0][1][21:29], alldata[0][1][61:70], alldata[0][1][103:112]],tolerance=5)
#Ba2Bi2O6, 415 < T < 450
data_epsilon[0][-4:] = merge_epsilon([alldata[0][1][33:37], alldata[0][1][74:78], alldata[0][1][116:120]],tolerance=5)
#Ba2BiSbO6, T = 225
data_epsilon[2][-3] = merge_epsilon([alldata[1][1][17,None], alldata[1][1][36,None], alldata[1][1][55,None]],tolerance=5)
#Ba2BiSbO6, 440 < T < 490 
data_epsilon[3][-5:] =  merge_epsilon([alldata[1][1][89:95], alldata[1][1][120:125]])
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

