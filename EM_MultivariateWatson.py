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

def EM_MWD(X,K,p, theta = 1e-2,maxiter=10**6):
    
    MWD = WatsonDistribution(p)
    n = len(X[0])
    a = 1/2
    c = p/2


    #Initialize mu and kappa and Pi
    Pis = np.ones(K)/K
    kappas = np.ones(K)
    mus = np.zeros((p,K))

    # Assuming K < p. we create mus as k different vector in standard basis E of R^p
    for j in range(K):
        mus[j*int(p/K),j] = 1
    
    
    
    # Estimating initial likelihood of data and hyperparameters
    likelihood = sum(np.log([sum([ Pis[j]* MWD.pdf(x,mus[:,j],kappas[j]) for j in range(K)]) for x in X.T]))



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

            # Estimating kappa from bounds
            LB = (rj*c-a)/(rj*(1-rj)) * (1- (1-rj)/(c-a))
            Bo = (rj*c-a)/(2*rj*(1-rj)) * (1+ np.sqrt(1+(4*(c+1)*rj*(1-rj))/(a*(c-a))))
            UB = (rj*c-a)/(rj*(1-rj)) *(1 + rj/a) 
            
            if a/c < rj and rj < 1:
                kappas[j] = (LB+Bo)/2
                #kappas[j] = LB
            elif 0 < rj  and rj < a/c:
                kappas[j] = (UB+Bo)/2
                #kappas[j] = Bo
            elif rj == a/c:
                kappas[j] = 0
            else:
                print("Nothing was found")
            #kappas[j] = 1/MWD.g(1/2,p/2,rj) # Inverse function or this?

        print(kappas)

        #Estimating the likelihood of data for newly found hyperparameters in order to access convergence
        likelihood_iter = sum(np.log([sum([ Pis[j]* MWD.pdf(x,mus[:,j],kappas[j]) for j in range(K)]) for x in X.T]))
        if abs(likelihood - likelihood_iter) < theta or _ == maxiter-1:

            # Assigning clusters if convergence
            print(f"Algorithem converged after {_} iterations, Converged = {abs(likelihood - likelihood_iter) < theta}, Max iterations reached = {_ == maxiter}")
            B = np.zeros((K,n))
            for i in range(n):
                Bi = np.array([Pis[j]*MWD.pdf(X[:,i],mus[:,j],kappas[j]) for j in range(K)])
                for j in range(K):
                    B[j,i] =   Bi[j]/np.sum(Bi)
            Assignments = np.argmax(B,axis=0)
            break
        else:
            likelihood = likelihood_iter
    

    return Pis,kappas,mus,Assignments

#%%


