#%%
from WatsonDistribution import WatsonDistribution
from CoherenceMap import LeadingEigenVector
from tqdm import tqdm
import numpy as np
from numba import njit

@njit
def matrix_vector_product(a,b):
    M = np.zeros((90,90))
    for i in range(90):
        for j in range(90):
            M[i,j] = a[i]*b[j]
    return M

def EM_MWD(X,K,p, theta = 1e-6,maxiter=10**6):
    MWD = WatsonDistribution(p)
    n = len(X[0])



    #Initialize mu and kappa and Pi
    Pis = np.ones(K)/K
    kappas = np.ones(K)
    mus = np.zeros((p,K))

    # Assuming K < p. we create mus as k different vector in standard basis E of R^p
    for j in range(K):
        mus[j*int(p/K),j] = 1
    
    
    
    # Estimating initial likelihood of data and hyperparameters
    likelihood = sum([sum(np.log([ Pis[j]* MWD.pdf(x,mus[:,j],kappas[j]) for i in range(K)])) for x in X.T])



    # Fitting loop - will run for maxiteration or until convergence
    for _ in tqdm(range(maxiter)):


        # Expectation step. Calculate Bij
        B = np.zeros((K,n))
        for i in range(n):
            Bi = np.array([Pis[j]*MWD.pdf(X[:,i],mus[:,j],kappas[j]) for j in range(K)])
            for j in range(K):
                B[j,i] =   Bi[j]/np.sum(Bi)


        # M step. Update Pis,Mus,kappas
        Pis = np.mean(B,axis=1)
        for j in range(K):
            S = np.zeros((90,90))

            for i in range(n):
                #S +=  B[j,i] * (X[:,i] @ X[:,i].T)
                S +=  B[j,i] * matrix_vector_product(X[:,i],X[:,i])
            S *= 1/sum(B[j,:])

            muj = LeadingEigenVector(S)

            mus[:,j] = muj

            rj = muj.T @ S @ muj
            
            kappas[j] = 1/MWD.g(1/2,p/2,rj) # Inverse function or this?


        # Estimating the likelihood of data for newly found hyperparameters in order to access convergence
        # likelihood_iter = sum([sum(np.log([ Pis[j]* MWD.pdf(x,mus[:,j],kappas[j]) for i in range(K)])) for x in X.T])
        # if abs(likelihood - likelihood_iter) < theta:
        #     break
        # else:
        #     likelihood = likelihood_iter
    

    return Pis,kappas,mus

#%%