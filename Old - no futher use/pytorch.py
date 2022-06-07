# Optimzing Watson Models using pytorch gradients
#%%
import pickle
import numpy as np
with open('.DataPhase.pickle','rb') as f:
    X = pickle.load(f)

#%%
import torch

torch.set_default_dtype(torch.float64)
X_tensor = torch.from_numpy(X)
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


def log_M(a,c,k):
    
    M0 = 1
    Madd = 1

    for j in range(1,100000):
        Madd = Madd * (a+j-1)/(c+j-1) * k/j
        M0 += Madd
        if Madd < 1e-10:
            break
    return M0

def Gamma(n):
        return float(torch.jit._builtins.math.factorial(n-1))

def c(p,k):
        return Gamma(p/2) / (2 * np.pi**(p/2) * M(1/2,p/2,k))

def pdf(x,mu,kappa,p):
        Wp = c(p,kappa) * torch.exp(kappa * (mu.T @ x )**2)
        return Wp

#@torch.jit.script
def log_c(p,k):
        return torch.lgamma(torch.tensor([p/2])) - torch.log(torch.tensor(2 * np.pi**(p/2))) - torch.log(M(1/2,p/2,k))

#@torch.jit.script
def log_pdf(x,mu,kappa,p):
        Wp = log_c(p,kappa) + kappa * (mu.T @ x )**2
        return Wp

#@torch.jit.script
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

#log_likelihood.code
#%%
K = 7
p = 90

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

#%%
learning_rate = 0.2
n_iters = 20
epochs = 10

#torch.autograd.set_detect_anomaly(False)
Parameters = [
    {'params':pi},
    {'params':kappa},
    {'params':mu}
]

Optimzier = torch.optim.Adam(Parameters,lr=learning_rate)

#%%
for epoch in range(n_iters):
    likelihood_output = -log_likelihood(X_tensor[:,0:(330*10+1)],pi,kappa,mu)
    likelihood_output.backward()

    # Using optimzer
    Optimzier.step()
    Optimzier.zero_grad()

    if epoch % 10 == 0:
        print(f"epoch {epoch+1}; Log-Likelihood = {likelihood_output}")


# %%
import plotly.express as px

p=90
K = 7
pi_con = Softmax(pi)
kappa_con = Softplus(kappa)
mu_con = torch.zeros((p,K))
for k in range(K):
    mu_con[:,k] =  mu[:,k] / torch.sqrt(mu[:,k].T @ mu[:,k])

for model in range(7):
    print(f"Model {model}: Pi = {pi_con[model]} , kappa = {kappa_con[model]}")
    fig = px.bar(mu_con[:,model].detach().numpy())
    #fig.update_layout(width=300)
    fig.show()

