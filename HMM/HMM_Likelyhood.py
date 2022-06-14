import torch
import numpy as np
import torch

#%%
#torch.set_default_dtype(torch.float64)
Softmax = torch.nn.Softmax(0)
Softplus = torch.nn.Softplus()

#%%
def lsumMatrix(X):
    Max = X.max(1).values
    #return  Maxes + torch.log(torch.exp(x-Maxes).sum())
    return torch.log(torch.exp(X-Max).sum(1)) + Max

def lsum(x):
    #return  Maxes + torch.log(torch.exp(x-Maxes).sum())
    return x.max() + torch.log(torch.exp(x-x.max()).sum())



#%%

def InitializeParameters(n,p,K):
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
    Tk = torch.ones((n,K,K),requires_grad=grad)
    Pinit = torch.ones((K,n),requires_grad=grad)

    return kappa,mu,Tk,Pinit


def M(a,c,k):

    M0 = torch.ones(len(k))
    Madd = torch.ones(len(k))

    for j in range(1,10000):
        Madd = Madd * (a+j-1)/(c+j-1) * k/j
        M0 += Madd
        if all(Madd < 1e-10):
            break
    return M0

def M_log(a,c,k):
    
    M0 = torch.ones(len(k))
    Madd = torch.ones(len(k))
    M0_log = torch.zeros(len(k))


    for j in range(1,100000):
        Madd = Madd * (a+j-1)/(c+j-1) * k/j
        M0 += Madd
        Madd_log = torch.log(1+Madd.clone()/M0.clone())
        M0_log += Madd_log
        if all(Madd_log < 1e-10) :
            break
    return M0_log

#@torch.jit.script
def log_c(p,kappa):
        return torch.lgamma(torch.tensor([p/2])) - torch.log(torch.tensor(2 * np.pi**(p/2))) - M_log(1/2,p/2,kappa) #torch.log(M(1/2,p/2,kappa)) M_log(1/2,p/2,kappa) 
        #return torch.lgamma(torch.tensor([p/2])) - torch.log(torch.tensor(2 * np.pi**(p/2))) - torch.log(M(1/2,p/2,k))

#@torch.jit.script
def log_pdf(X,mu,kappa,p):
        Wp = log_c(p,kappa) + kappa * (mu.T @ X).T**2
        return Wp

#@torch.jit.script
def HMM_log_likelihood(X,Pinit,kappa,mu,Tk,p=90,K=7):

    Tlog = torch.log(Tk)    
    Emmision_Prop = log_pdf(X,mu,kappa,p).T

    Prob = torch.log(Pinit) + Emmision_Prop[:,0]
    for n in range(1,330):
        Prob = lsumMatrix(Prob.clone() + Tlog) + Emmision_Prop[:,n]
    return lsum(Prob) 

    # V = torch.zeros((K,330))
    # V[:,0] = torch.log(InitalState) + Emmision_Prop[:,0]
    # for n in range(1,330):
       
    #     V[:,n] = lsumMatrix(V[:,n-1] + Tlog) + Emmision_Prop[:,n]        
    #     #for k in range(K):
    #         V[k,n] = lsum(V[:,n-1] + torch.log(Tk.T[:,k])) + Emmision_Prop[k,n]  # Det her Forstår vi ikke!
    # return  lsum(V[:,329])


def Accumulated_HHM_LL(X,Pinit,kappa,mu,Tk,n,p=90,K=7):

    # Constraining Parameters:
    kappa_con = Softplus(kappa)
    mu_con = mu /torch.sqrt((mu * mu).sum(axis=0))

    Subjectlog_Likelihood = torch.zeros(n)
    
    
    for subject in range(n):
        Subjectlog_Likelihood[subject] = HMM_log_likelihood(X[:,330*subject:330*(subject+1)],Softmax(Pinit[:,subject]),kappa_con,mu_con,Softmax(Tk[subject]),p,K)
    
    return Subjectlog_Likelihood.sum()





def Optimizationloop(X,Parameters,lose,Optimizer,n,n_iters : int,p=90,K =7):
        Error_prev = 0
        for epoch in range(n_iters):
                Error = -lose(X,*Parameters,n,p,K=K)

                if torch.isnan(Error):
                    print("Optimizationloop has Converged")
                    break

                Error.backward()

                # Using optimzer
                Optimizer.step()
                Optimizer.zero_grad()

                #if abs(Error)-abs(Error_prev) < 1e-2:
                #    break

                #Error_prev = Error
                if epoch % 1 == 0:
                        print(f"epoch {epoch+1}; Log-Likelihood = {Error}")
        return Parameters
