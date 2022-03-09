import numpy as np
from numpy.linalg import eigh


def LeadingEigenVector(At):

    EigenValues , EigenVectors = eigh(At)
    LeV = EigenVectors[:,-1]

    return LeV

def CoherenceMap(Theta):
    N,T = Theta.shape
    
    LEiDA_Signal = np.zeros((N,T))

    At = np.zeros((N,N))

    for t in range(T):
        CurrentSample = Theta[:,t]
        for j in range(N):
            for k in range(N):
                At[j,k]=np.cos(CurrentSample[j]-CurrentSample[k])
        
        LEiDA_Signal[:,t] = LeadingEigenVector(At)

    return LEiDA_Signal