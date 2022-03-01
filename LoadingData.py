#%%
import numpy as np
import pandas as pd

df = pd.read_csv("AALdata/sub-0001_faces.csv",sep=",",header=None)
X = df.values