#%%

import numpy as np
import pandas as pd
from Bandpass import butter_bandpass_filter
from Hilbert import Hilbert_transform,phase
from CoherenceMap import CoherenceMap

df = pd.read_csv("AALdata/sub-0001_faces.csv",sep=",",header=None)
X = df.values
N,T = X.shape
X_filtered = np.zeros((N,T))

for idx, x in enumerate(X):
    X_filtered[idx,:] = butter_bandpass_filter(x)


#%%

Theta = np.zeros((N,T))


for idx, x in enumerate(X_filtered):
    s_t,sh_t=Hilbert_transform(x)
    Theta_t = phase(s_t, sh_t)
    Theta[idx,:] = Theta_t

LEiDA_Signal = CoherenceMap(Theta)

#%%

import plotly.express as px


fig = px.line(x=np.arange(len(Theta[0,:]),), y=Theta[0,:])
fig.show()


#%%
import plotly.express as px


fig = px.histogram(x=np.arange(90), y=LEiDA_Signal[:,30],nbins=90)
fig.show()


