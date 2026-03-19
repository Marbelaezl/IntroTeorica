import numpy as np

def FreeEnergy(params,variables,couplings):
    "Calculates the free energy for a given combination of quoefficients and order parameters (q+,q-,epsilon)"
    #Params is a numpy array of length n*len(variables), with n the highest power to include
    #Couplings is an array containing coupling terms, in the form [k,power1,power2,...,powern]
    #Coupling of higher order than non-coupled terms is not allowed because it is unphysical.
    #If you really need to include such couplings, you may pad the params array with zeroes 
    res=0
    #Check that parameters have been passed appropriately
    try:
        assert (len(params) == len(variables))
    except:
        print("Error in FreeEnergyTwoTilt: argument mismatch between params (length ", str(len(params)), ") and variables (length ", str(len(variables)),")")
        return np.inf
    #Add zero vector as first column in params. This is necessary for faster coupling calculation
    params=np.hstack([np.zeros((len(params),1)),params])    
    #Generate vandermonde matrix which contains the relevant powers of the variables
    var_powers=np.vander(variables,N=params.shape[1],increasing=True)
    res=np.sum(params*var_powers)
    #Add couplings. This part uses unomptimized python for-loops and may change in the future if it becomes rate-limiting
    for i in range(0,len(couplings)):
        #identify relevant entries in vandermonde matrix
        indices = np.vstack([np.arange(params.shape[0]),couplings[i,1:]]).astype(int)
        extra = couplings[i,0] *np.prod(var_powers[indices[0],indices[1]]) 
        res = res+extra
    return res

def GenParams(vec,model="2t44"):
    "Generates the parameter matrix as required for FreeEnergy according to the quoefficient vector vec"
    "and the model. Valid models are:"
    "'2t44' (2tilt-4th order + temperature & deformation)"
    options=["2t44"]
    if model not in options:
        print("Error in GenParams: invalid model '", model, "'. Valid models are: ", options)
        raise ValueError
    elif model == "2t44":
        try:
            assert(len(vec)==10)
        except:
            print("Error in GenParams: number of entries in vec (", len(vec), "does not match model '2t44' (length 10 required)")
            raise ValueError
    #model parameters (10): a+, T+, b+, a-, T-, b-, c, k, lambda+, lambda-
    # 4 lines (q+,q-,T,epsilon) and 4 columns (maximum order 4)
        params=np.zeros((4,4))
    # 5 couplings: a+T+q+^2, a-T-q-^2, cq+^2q-^2, lambda+epq+^2, lambda-epq-^2     
        couplings=np.zeros((5,5))
    #terms related to in-phase tilts
        params[0,1]= -vec[0]*vec[1] #-a+T+q+^2
        params[0,3]= vec[2] #bq+^4
        couplings[0] = np.array([vec[6],2,2,0,0]) #cq+^2q-^2
        couplings[1] =np.array([vec[0],2,0,1,0]) #aTq+^2
        couplings[2] =np.array([vec[8],2,0,0,1]) #lambda+ epsilon q+^2
    #terms related to out-of-phase tilts
        params[1,1]= -vec[3]*vec[4] #-a-T-q-^2
        params[1,3]= vec[5] #bq+^4
        couplings[3] = np.array([vec[3],0,2,1,0]) #aTq-^2
        couplings[4] = np.array([vec[9],0,2,0,1]) #lambda- epsilon q-^2
    #Elastic deformation 1/2k epsilon^2
        params[3,1] = 0.5*vec[7]
        return params,couplings

def MinimizeEnergy(params,variables,couplings, mask=None, maxchange=1000, cycles=1000, delta=1e-12, verbose=False):
    "Find the values for variables that minimize FreeEnergy for a system with fixed params and couplings"
    "Optionally, a mask can be specified to fix which variables are minimized (1) or not (0)."
    "By default, FreeEnergy is minimized against all variables. It is assumed that all variables are at least 0"
    "In order to ensure convergence, a maximum change of 1000 (either additive or multiplicative in 1 cycle) is added to halt the system"
    #If no mask is provided, refine over all variables
    if mask is None:
        mask=np.ones_like(variables,dtype=bool)
    else:
        #Resize if needed. The behaviour should be intuitive: if only some values are provided,
        #refine only over the specified values. If asked to refine over more parameters than there are
        #variables, do nothing with the extra information
        mask.resize((len(variables)))
        mask = mask.astype(bool)
    #Save original value to check for stability
    original = variables
    delta=1e-12 #Delta of the same order of numerical tolerance for calculation in tests
    delta_arr=np.zeros_like(variables) #array with delta in one entry and 0 in all others
    beta=0.8 #parameter between 0 and 1 for backtracking line search
    cycle=0
    mag=0 #magnitude of last change
    while True:
        cycle+=1
        gradient=np.zeros_like(variables) #Generate empty array for gradient
        if verbose:
            print("Cycle ", cycle, ":")
        index = 0 #For assignment of refinable parameters
        for i in range(0,len(mask)):
            if mask[i] == True:
                delta_arr[index] = delta #This implementation makes it easier to specify the derivative without modifying variables
                gradient[index] = (FreeEnergy(params, variables+delta_arr, couplings) - FreeEnergy(params, variables-delta_arr, couplings))/(2*delta) #Central difference
                delta_arr[index]=0 #Reset delta
            index = index+1 #Go to the next position, no matter if gradient was calculated or not
        if verbose:
            print("Free energy at start of step: ", FreeEnergy(params, variables, couplings))
        #Determine step size via backtracking line search
        step=1
        if verbose:
            print("RHS of step inequality: ", FreeEnergy(params, variables - gradient, couplings))
        step_cycles=0
        while  FreeEnergy(params, variables-step*gradient, couplings) > FreeEnergy(params, variables - step*gradient*beta, couplings):
            step=step*beta
            step_cycles +=1
            if verbose:
                print("Gradient: ", gradient)
                print("Cycle ",cycle, ": updating step size from", step/beta, "(FreeEnergy ", FreeEnergy(params, variables-step*gradient/beta, couplings), "to ", step,
                      "(FreeEnergy ", FreeEnergy(params, variables-step*gradient, couplings), ")" )
            if(step_cycles > cycles):
                print("Error in MinimizeEnergy : no suitable step size was found for cycle ", cycle)
                return [-2,variables] #Error code
            
        #"Bad" halting conditions:
        #1. Instability (System has no local minima)
        if( (gradient*step > maxchange).any() or (np.abs((variables-step*gradient)-original) > maxchange).any()):
            variables = original
            print("Error in MinimizeEnergy: maxchange exceeded in cycle", cycle, ". Returning original values")
            if verbose:
                print("Error type: -1 (maxstep exceeded)")
                print("Step: ",step, "delta: ", gradient*step )
                print("Deviation from original variables: ",(variables-step*gradient)-original)
            return [-1,variables] #Error code
        #2. Number of cycles exceeded. Return last value
        if(cycle > cycles):
            print("Warning: MinimizeEnergy did not converge to a solution (Number of cycles exceeded). Check that your system has minima, and change cycles if needed")
            if verbose:
                print("Number of cycles: ", cycles)
                print("Initial FreeEnergy:", FreeEnergy(params, original, couplings))
                print("Final FreeEnergy:",FreeEnergy(params, variables, couplings))
                print("Magnitude of last change: ", mag)
            return [1,variables] # Warning
        
        if verbose:
            print("Updating variables. Old: ", variables)
        #If no errors ocurred, update variables
        variables = variables - step*gradient
        #Make variables 0 if they go to negative values
        #variables=np.max([variables,np.zeros_like(variables)],axis=0)
        if verbose:
            print("New variables: ", variables)
        mag = step*(np.sum(gradient*gradient))**0.5
        if verbose:
            print("Change magnitude: ", mag)
            print("Free energy after update: ", FreeEnergy(params, variables, couplings))
        
        #1. Derivative along all variables is below delta
        #2. Change in FreeEnergy is below delta
        if(np.abs(gradient) < delta).all() and  (FreeEnergy(params, variables+step*gradient,couplings) - FreeEnergy(params, variables, couplings) < delta):
            if verbose:
                print("Halting condition reached: all derivatives or FreeEnergy change below delta")
                print("Initial FreeEnergy:", FreeEnergy(params, original, couplings))
                print("Final FreeEnergy: ", FreeEnergy(params, variables, couplings))
            return [0,variables]

    
    