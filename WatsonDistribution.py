#%%
from numba import njit
from math import factorial
from scipy.special import poch
import numpy as np

@njit
def faculty(x):
    if x==0: 
        return 1
    else: 
        return x*faculty(x-1)
@njit
def rising_factorial(a,j):
    if j == 0: 
        return 1
    else: 
        return (a+j-1)*rising_factorial(a,j-1)
class WatsonDistribution:
    def __init__(self,p):
        self.p = p



    def pdf(self,x,mu,kappa):
        Wp = self.c(self.p,kappa) * np.exp(kappa * (mu.T @ x )**2)
        return Wp

    def log_likelihood(self,mu,k,X):
        n = len(X[0])
        y = 0
        likelihood = n * (k * mu.T @ self.Scatter_matrix(X) @ mu - np.log(self.M(1/2,self.p/2,k)) + y)
        return likelihood
    
    def Scatter_matrix(self,X):
        S = np.zeros((self.p,self.p))
        for x in X:
            S += x@x.T
        return S
    
    def Gamma(self,n):
        return faculty(n-1)

    def Mj(self,a,c,k,j):
        return poch(a,j)/poch(c,j) * k**j / factorial(j)
    
    def M(self,a,c,k):
        j=0
        Mf = self.Mj(a,c,k,j)

        while True:
            M_add = self.Mj(a,c,k,j+1)
            
            if (M_add)  / Mf < 1e-10:
                #print(f"M(a,c,j) Converged after j = {j} iterations")
                break
            else:
                Mf += M_add
                j += 1 
        return Mf

    def Mdj(self,a,c,k,j):
        return poch(a,j)/poch(c,j) * j*k**(j-1) / factorial(j)

    def Md(self,a,c,k):
        j=0
        Mf = self.Mdj(a,c,k,j)

        while True:
            M_add = self.Mdj(a,c,k,j+1)
            
            if (M_add)  / Mf < 1e-10:
                #print(f"M(a,c,j) Converged after j = {j} iterations")
                break
            else:
                Mf += M_add
                j += 1 
        return Mf
    
    def c(self,p,k):
        return self.Gamma(p/2) / (2 * np.pi**(p/2) * self.M(1/2,p/2,k))

    def g(self,a,c,k):
        return self.Md(a,c,k)/self.M(a,c,k)
