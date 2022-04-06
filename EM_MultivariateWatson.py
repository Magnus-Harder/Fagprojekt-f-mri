#%%
from WatsonDistribution import WatsonDistribution
from CoherenceMap import LeadingEigenVector
import numpy as np




def EM_MWD(X,K,p):
    MWD = WatsonDistribution(p)
    n = len(X[0])
    #Initialize mu and kappa and Pi
    Pis = np.ones(K)/K
    kappas = np.ones(K)
    mus = np.zeros((p,K))
    for j in range(K):
        mus[j*int(p/K),j] = 1
    

    # E step. Calculate Bij
    B = np.zeros((K,n))
    for i in range(n):
        Bi = np.array([Pis[j]*MWD.pdf(X[:,i],mus[j],kappas[j]) for j in range(K)])
        for j in range(K):
            B[j,i] =   Bi[j]/np.sum(Bi)

    # M step. Update Pis,Mus,kappas
    Pis = np.mean(B,axis=1)
    for j in range(K):
        S = np.zeros((n,n))
        for i in range(n):
            S +=  B[j,i] * (X[:,i] @ X[:,i].T)
        S *= 1/sum(B[j,:])
        muj = LeadingEigenVector(S)
        mus[:,j] = muj
        rj = muj.T @ S @ muj
        


#%%