#Import dependencies
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import time
except:
    print("Fatal error: dependencies not found")
    exit(404)
#Import project files
try:
    import functions
except:
    print("Fatal error: functions.py not found")
    exit(404)

verbose = True
start_time = time.time()
passed=0
failed=[]
total_tests=0

#---------------------------------------------------------------------------------------------
#Test 1: Free energy function works as intended for an easy second-order, 4-variables system
if verbose:
    print("Test 1: Free energy function for small system. (3 variables, 1 coupling)")
total_tests+=1

manual_res=0
try:
    delta=0
    for j in range(0,10):
        coeffs=np.random.rand(4,2)
        order=np.random.rand(4)
        couplings=np.array([[np.random.rand(),1,0,1,0]])
        manual_res=0
        for i in range(0,4):
            manual_res += coeffs[i,0]*order[i] + coeffs[i,1]*order[i]**2
        manual_res += couplings[0,0]*order[0]*order[2]
        delta += np.abs(manual_res-functions.FreeEnergy(coeffs, order,couplings))
    if verbose:
        print("Test 1: Acummulated difference between manual and automatic calculation over 10 trials: ",delta)
    if( delta < 1e-12):
        print("Test 1 passed")
        passed += 1
    else:
        print("Test 1 failed: Unacceptable difference in Free Energy calculation ( delta = ",delta,")")
        failed.append(1)
        print("Manual: ", manual_res)
        print("Auto: ", functions.FreeEnergy(coeffs, order,couplings))
except:
    print("Test 1 failed: unknown")
    failed.append(1)
if verbose:
    print("Time elapsed: ", time.time()-start_time, "s")
#---------------------------------------------------------------------------------------------
#Test 2: Check that the free energy function works as intended
if verbose:
    print("Test 2: Behaviour of 2t44 model ")
total_tests+=1
delta=0
try:
    for i in range(0,10):
        vec=np.random.rand(10)
        ap,Tp,bp,am,Tm,bm,c,k,lap,lam = vec
        params,couplings= functions.GenParams(vec)
        test_vars=np.random.random(4)
        qp,qm,T,ep = test_vars
        #Calculate free energy manually
        manual = (ap*(T-Tp)*qp**2) + bp*qp**4 + (am*(T-Tm)*qm**2) + (bm*qm**4) + c*(qp*qm)**2
       #Add distorsion couplings
        manual += 0.5*k*ep**2 + lap*ep*qp**2 + lam*ep*qm**2
        delta = delta +np.abs(manual-functions.FreeEnergy(params, test_vars, couplings))
    if verbose:
        print("Test 2: accumulated difference along 10 trials =",delta)
    if delta < 1e-12:
        passed +=1
        print("Test 2 passed")
    else:
        print("Test 2 failed: Unacceptable difference in Free Energy calculation (", delta, ") for 2t44 model")
except:
    print("Test 2 failed: unknown")
    failed.append(2)
if verbose:
    print("Time elapsed: ", time.time()-start_time, "s")
    

    
print("Final statistics: ", passed, " out of ", total_tests, "passed (", 100.0*passed/total_tests, "% )")
print("Time elapsed: ", time.time()-start_time, "s")
print("Failed tests: ", failed)
