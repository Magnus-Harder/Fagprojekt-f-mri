#%%
import torch
import numpy as np
import torch


#%%
#%%
#torch.set_default_dtype(torch.float64)
Softmax = torch.nn.Softmax(0)
Softplus = torch.nn.Softplus()

def M(a,c,k):
    
    M0 = 1
    Madd = 1

    for j in range(1,100000):
        Madd = Madd * (a+j-1)/(c+j-1) * k/j
        M0 += Madd
        if Madd < 1e-10:
            break
    return M0

def M_log(a,c,k):
    
    M0 = torch.ones(len(k))
    Madd = torch.ones(len(k))
    M0_log = torch.zeros(len(k))


    for j in range(1,100000):
        Madd = Madd * (a+j-1)/(c+j-1) * k/j
        Madd_log = torch.log(1+Madd/M0)
        M0 += Madd
        M0_log += Madd_log
        if all(Madd_log < 1e-10) :
            break
    return M0_log


#%%


#%%

#@torch.jit.script
def log_c(p,k,LogM=False):
        return torch.lgamma(torch.tensor([p/2])) - torch.log(torch.tensor(2 * np.pi**(p/2))) - M_log(1/2,p/2,k)
        #return torch.lgamma(torch.tensor([p/2])) - torch.log(torch.tensor(2 * np.pi**(p/2))) - torch.log(M(1/2,p/2,k))

#@torch.jit.script
def log_pdf(X,mu,kappa,p):
        Wp = log_c(p,kappa) + kappa * (mu.T @ X).T**2
        return Wp


#@torch.jit.script
def log_likelihood(X,pi,kappa,mu,p=90,K=7):
    p = 90
    K = 7
    # Constraining Parameters:
    pi_con = Softmax(pi)
    kappa_con = Softplus(kappa)
    mu_con = torch.zeros((p,K))
    mu_con = mu /torch.diagonal(torch.sqrt(mu.T @ mu))
         
    inner = (torch.log(pi_con) + log_pdf(X,mu_con,kappa_con,p)).T

    Max = torch.max(inner,axis=0).values
    outer = (torch.log(torch.exp(inner-Max).sum()) + Max).sum()

    return outer

def Optimizationloop(X,Parameters,lose,Optimizer,n_iters : int):
        for epoch in range(n_iters):
                Error = -lose(X,*Parameters)
                Error.backward()

                # Using optimzer
                Optimizer.step()
                Optimizer.zero_grad()

                if epoch % 10 == 0:
                        print(f"epoch {epoch+1}; Log-Likelihood = {Error}")
        return Parameters

def Initialize(p,K):
        mus = np.random.rand(p,K)
        for j in range(K):
                mus[:,j] = mus[:,j]/np.sqrt(mus[:,j].T @ mus[:,j]) 
        #print(mus[:,j].T @ mus[:,j])

        # Intialize pi,mu and kappa
        grad = True
        pi = torch.tensor([1/K for _ in range(K)],requires_grad=grad)
        kappa = torch.tensor([1. for _ in range(K)],requires_grad=grad)
        mu = torch.from_numpy(mus)   
        mu.requires_grad = grad

        return pi,kappa,mu