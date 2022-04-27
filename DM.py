#%%
from WatsonDistribution import WatsonDistribution
from CoherenceMap import LeadingEigenVector
from tqdm import tqdm
import numpy as np
import plotly.express as px

#%%
def DC(X, K, p,maxiter=1000):

    n = len(X[0])

    #Initialize mus
    mus = np.random.rand(p,K)
    for j in range(K):
        mus[:,j] = mus[:,j]/np.sqrt(mus[:,j].T @ mus[:,j]) 
        print(mus[:,j].T @ mus[:,j])

    for _ in tqdm(range(maxiter)):
        #E step
        Xj = [[] for _ in range(K)]
        for i in range(n):
            c = np.array([X[:,i].T @ mus[:,k] for k in range(K)])
            Xj[c.argmax()].append(i)
        
        #M step
        for k in range(K):
            Aj = np.zeros((p,p))
            for idx in Xj[k]:
                Aj += np.outer(X[:,idx,X[:,idxk][idx])
                mus[:,j] = Aj@mus[:,j] / (Aj @ mus[:,j])
    
    Assigments = np.zeros(n)
    for i in range(n):
            c = np.array([X[:,i].T @ mus[:,k] for k in range(K)])
            Assigments[i] = c.argmax()

    return mus, Assigments
    
#%%

import pickle
with open('.Data_LEiDA_Representation.pickle','rb') as f:
    X = pickle.load(f)


X_test = X[:,0:(330*20 +1)]

mus,states = DC(X_test,K=7,p=90,maxiter=100)

#%%
for i in range(5):
    fig = px.line(states[(5*i*330):(5*i*330+331)])
    fig.show()

#%%
for model in range(7):
    print(f"Model {model}:")
    fig = px.bar(mus[:,model])
    fig.show()
# %%
