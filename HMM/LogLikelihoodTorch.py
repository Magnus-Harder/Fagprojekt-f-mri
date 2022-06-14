#%%
import torch
import numpy as np
import torch

#torch.set_default_dtype(torch.float64)
Softmax = torch.nn.Softmax(0)
Softplus = torch.nn.Softplus()

def lsumMatrix(X):
    Max = X.max(1).values
    #return  Maxes + torch.log(torch.exp(x-Maxes).sum())
    return torch.log(torch.exp(X-Max).sum(1)) + Max

def lsum(x):
    #return  Maxes + torch.log(torch.exp(x-Maxes).sum())
    return x.max() + torch.log(torch.exp(x-x.max()).sum())


def M_log(a,c,k):
    
    M0 = torch.ones(len(k))
    Madd = torch.ones(len(k))
    M0_log = torch.zeros(len(k))


    for j in range(1,10000):
        Madd = Madd * (a+j-1)/(c+j-1) * k/j
        M0 += Madd
        Madd_log = torch.log(1+Madd.clone()/M0.clone())
        M0_log += Madd_log
        if all(Madd_log < 1e-10) :
            break
    return M0_log

def log_c(p,k):
        return torch.lgamma(torch.tensor([p/2])) - torch.log(torch.tensor(2 * np.pi**(p/2))) - M_log(1/2,p/2,k)
        #return torch.lgamma(torch.tensor([p/2])) - torch.log(torch.tensor(2 * np.pi**(p/2))) - torch.log(M(1/2,p/2,k))

def log_pdf(X,mu,kappa,p):
        Wp = log_c(p,kappa) + kappa * (mu.T @ X).T**2
        return Wp

def log_likelihood_eval(X,pi,kappa,mu,p=90,K=7):
         
    inner = (torch.log(pi) + log_pdf(X,mu,kappa,p)).T

    Max = torch.max(inner,axis=0).values
    outer=(torch.log(torch.exp(inner-Max).sum()) + Max).sum()

    return outer


def log_likelihood(X,pi,kappa,mu,p=90,K=7):
    # Constraining Parameters:
    pi_con = Softmax(pi)
    #kappa_con = Softplus(kappa)
    kappa_con = torch.minimum(Softplus(kappa),torch.tensor([800]))# min(af kappa_con,800)
    mu_con = mu /torch.sqrt((mu * mu).sum(axis=0))
         
    inner = (torch.log(pi_con) + log_pdf(X,mu_con,kappa_con,p)).T

    Max = torch.max(inner,axis=0).values
    outer=(torch.log(torch.exp(inner-Max).sum()) + Max).sum()

    return outer

def Optimizationloop(X,Parameters,lose,Optimizer,n_iters : int,K =7):
        for epoch in range(n_iters):
                Error = -lose(X,*Parameters,K=K)
                Error.backward()

                if torch.isnan(Error):
                    print("Optimizationloop has Converged")
                    break

                # Using optimzer
                Optimizer.step()
                Optimizer.zero_grad()

                if epoch % 100 == 0:
                        print(f"epoch {epoch+1}; Log-Likelihood = {Error}")
        return Parameters

def OptimizationTraj(X,Parameters,lose,Optimizer,n_iters : int,K =7):
        Trajectory = np.zeros(n_iters)
        for epoch in range(n_iters):
                Error = -lose(X,*Parameters,K=K)
                Error.backward()

                if torch.isnan(Error):
                    print("Optimizationloop has Converged")
                    break

                # Using optimzer
                Optimizer.step()
                Optimizer.zero_grad()

                if epoch % 1000 == 0 or epoch == n_iters:
                        print(f"epoch {epoch+1}; Log-Likelihood = {Error}")

                Trajectory[epoch] = Error
        return Trajectory


def Initialize(p,K):
        mu = torch.zeros((p,K))
        for j in range(K):
                val = 1 if j % 2 == 0 else -1
                mu[(j*int(p/K)),j] = val
                #mus[:,j] = mus[:,j]/np.sqrt(mus[:,j].T @ mus[:,j]) 
        #print(mus[:,j].T @ mus[:,j])

        # Intialize pi,mu and kappa
        grad = True
        pi = torch.tensor([1/K for _ in range(K)],requires_grad=grad)
        kappa = torch.tensor([1. for _ in range(K)],requires_grad=grad) 
        mu.requires_grad = grad

        return pi,kappa,mu