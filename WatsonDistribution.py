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

[rising_factorial(6,i) for i in range(6)]



# M stoppes når den konvergere altså ændringen ved j+1 er tilpas lille.

# Information kriteria AIC.
# 

# 7
#%%
import plotly.express as px

y = [sum([poch(1/2,j)/poch(45,j) * 100**j / factorial(j) for j in range(js)]) for js in range(100)]
x = [j for j in range(100)]


px.line(x=x,y=y)

#%%
class WatsonDistribution:
    def __init__(self,p,mu,k):
        self.p = p
        self.mu = mu
        self.k = k


    def pdf(self,x,mu,k):
        Wp = self.c(self.p,self.k) * np.exp(self.k * (self.mu.T @ x )**2)
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
            
            if (Mf - (Mf + M_add))  / Mf < 1e-10:
                break
            else:
                Mf += M_add
                j += 1 
        return Mf
    
    def c(self,p,k):
        return self.Gamma(p/2) / (2 * np.pi**(p/2) * self.M(1/2,p/2,k))
#%%

mu = np.array([1,0])
k = 1
p = 2


WD = WatsonDistribution(p,mu,k)



x1=np.array([1,1])/np.sqrt(2)
x2=np.array([-1,1])/np.sqrt(2)
x3=np.array([1,-1])/np.sqrt(2)
x4=np.array([-1,-1])/np.sqrt(2)

x5=np.array([-2,-1])/np.sqrt(5)

X=np.array([x1,x2,x3,x4,x5])

