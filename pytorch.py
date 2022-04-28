# Optimzing Watson Models using pytorch gradients
#%%
import pickle
import numpy as np
with open('.Data_LEiDA_Representation.pickle','rb') as f:
    X = pickle.load(f)
means = X.mean(axis=1)
X_norm = X

for idx in range(X_norm.shape[1]):
    X_norm[:,idx] -= X_norm[:,idx].mean()
    X_norm[:,idx] = X_norm[:,idx] / np.sqrt(X_norm[:,idx].T@X_norm[:,idx])
#%%
import torch 
torch.set_default_dtype(torch.float64)
X_norm_tensor = torch.from_numpy(X_norm)

def M(a,c,k):
    
    M0 = 1
    Madd = 1

    for j in range(1,60):
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


def log_likelihood(X,pi,kappa,mu,p=90,K=7):
    for idx,x in enumerate(X.T):
        for j in range(K):
            if j == 0:
                inner = pi[j]* pdf(x,mu[:,j],kappa[j],p)
            else:
                inner += pi[j]* pdf(x,mu[:,j],kappa[j],p)
        if idx == 0:
            outer =  torch.log(inner)
        else:
            outer += torch.log(inner)
    
    #likelihood = sum(torch.log(torch.tensor([sum([ pi[j]* pdf(x,mu[:,j],kappa[j],p) for j in range(K)]) for x in X.T])))

    return outer

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



print(pi)


#%%
learning_rate = 0.5

Softmax = torch.nn.Softmax(0)
Softplus = torch.nn.Softplus()

for epochs in range(5):
    likelihood_output = log_likelihood(X_norm_tensor[:,0:(330*5+1)],pi,kappa,mu)

    likelihood_output.backward()

    with torch.no_grad():
        # Updating and ensuring pi sums to one
        pi -=  -learning_rate*pi.grad
        pi = Softmax(pi)

        
        # Updating and ensuring mu is a unit vektore
        mu -=learning_rate*mu.grad
        for k in range(K):
            mu[:,k] =  mu[:,k] / torch.sqrt(mu[:,k].T @ mu[:,k])

        # Updatiing kappa and ensure positive
        kappa  -= learning_rate*kappa.grad
        kappa = Softplus(kappa)

    # # Zeroing gradient
    pi.requires_grad = True
    mu.requires_grad = True
    kappa.requires_grad = True
    # pi.grad.zero_()
    # mu.grad.zero_()
    mu.grad.zero_()

    if epochs % 1 == 0:
        print(f"epoch {epochs+1}; Log-Likelihood = {likelihood_output}")

# %%
import plotly.express as px

for model in range(7):
    print(f"Model {model}: Pi = {pi[model]} , kappa = {kappa[model]}")
    fig = px.bar(mus[:,model])
    #fig.update_layout(width=300)
    fig.show()

