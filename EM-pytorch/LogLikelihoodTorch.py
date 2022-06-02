#%%
import torch
import numpy as np

torch.set_default_dtype(torch.float64)
Softmax = torch.nn.Softmax(0)
Softplus = torch.nn.Softplus()
import numba

# Normal Functions

@numba.njit
def M(a,c,k):
    
    M0 = 1
    Madd = 1

    for j in range(1,100000):
        Madd = Madd * (a+j-1)/(c+j-1) * k/j
        M0 += Madd
        if Madd < 1e-10:
            break
    return M0

def pdf(x,mu,kappa,p):
        Wp = c(p,kappa) * torch.exp(kappa * (mu.T @ x )**2)
        return Wp

def Gamma(n):
        return float(torch.jit._builtins.math.factorial(n-1))

def c(p,k):
        return Gamma(p/2) / (2 * np.pi**(p/2) * M(1/2,p/2,k))

# Log Scale Functions
def log_M(a,c,k):
    
    M0 = 1
    Madd = 1

    for j in range(1,100000):
        Madd = Madd * (a+j-1)/(c+j-1) * k/j
        M0 += Madd
        if Madd < 1e-10:
            break
    return M0

@torch.jit.script
def log_c(p,k):
        return torch.lgamma(torch.tensor([p/2])) - torch.log(torch.tensor(2 * np.pi**(p/2))) - torch.log(M(1/2,p/2,k))

@torch.jit.script
def log_pdf(x,mu,kappa,p):
        Wp = log_c(p,kappa) + kappa * (mu.T @ x )**2
        return Wp

@torch.jit.script
def log_likelihood(X,pi,kappa,mu,p=90,K=7):
    p = 90
    K = 7
    # Constraining Parameters:
    pi_con = Softmax(pi)
    kappa_con = Softplus(kappa)
    mu_con = torch.zeros((p,K))
    for k in range(K):
            mu_con[:,k] =  mu[:,k] / torch.sqrt(mu[:,k].T @ mu[:,k])

    # Calculating Log_Likelihood
    outer = 0
    for idx,x in enumerate(X.T):
        inner = torch.zeros(K)
        for j in range(K):
                inner[j] = torch.log(pi_con[j]) + log_pdf(x,mu_con[:,j],kappa_con[j],p)

        outer += torch.log(torch.exp(inner-torch.max(inner)).sum()) + torch.max(inner)
    
    #likelihood = sum(torch.log(torch.tensor([sum([ pi[j]* pdf(x,mu[:,j],kappa[j],p) for j in range(K)]) for x in X.T])))

    return outer

# Compiling code 
#M.code
log_c.code
log_pdf.code
log_likelihood.code