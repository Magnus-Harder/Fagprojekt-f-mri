#%%
from turtle import ycor
import numpy as np
import pandas as pd
from scipy.fftpack import rfftfreq
from scipy.signal import butter,lfilter
df = pd.read_csv("AALdata/sub-0001_faces.csv",sep=",",header=None)
X = df.values


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

#%%



#%%

import plotly.express as px

fs = 3/4 # Hz

lowcut = 0.01 # Hz
highcut = 0.1 # Hz

y_filter = butter_bandpass_filter(X[0],lowcut, highcut, fs, order=2)

fig=px.line(x=[i for i  in range(330)],y=X[0])
fig.show()

fig2 = px.line(x=[i for i in range(330)],y=y_filter)
fig2.show()

#%%

def ControlFunc(t):
    wave1 = 100 * np.sin(2*np.pi*100*t)
    wave2 = 50 * np.sin(2*np.pi*10*t)
    wave3 = 10 * np.sin(2*np.pi*200*t)
    wave4 = 30 * np.sin(2*np.pi*50*t)
    return wave1+wave2+wave3+wave4

def Goalfunc(t):

    return 100 * np.sin(2*np.pi*100*t)

y = [ControlFunc(t/420) for t in range(420*5)]
y_goal = [Goalfunc(t/420) for t in range(420*5)]
fs = 420 # Hz
x=[i/420 for i  in range(420*5)]
lowcut = 90# Hz
highcut = 105 # Hz
y_filter = butter_bandpass_filter(y,lowcut, highcut, fs, order=5)

fig=px.line(x=x,y=y)
fig.show()

# fig2 = px.line(x=[i/420 for i  in range(420*5)],y=y_filter)
# fig2.show()

# fig3 = px.line(x=[i/420 for i  in range(420*5)],y=y_goal)
# fig3.show()


from scipy.fft import rfft, irfft, rfftfreq

Z = rfft(y)
W = rfftfreq(2100,d=1/420)


for idx,w in enumerate(W):
    if not (w>lowcut and w<highcut):
        Z[idx]=0

yt = irfft(Z)

fig = px.line(x=x,y=yt)
fig.show()