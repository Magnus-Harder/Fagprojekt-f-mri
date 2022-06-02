import numpy as np
from scipy.signal import hilbert




def Hilbert_transform(signal):
    A_t = hilbert(signal)
    s_t , sh_t = np.real(A_t),np.imag(A_t)
    return s_t, sh_t

def phase(s_t, sh_t):
    theta_t = np.arctan(sh_t/s_t)
    return theta_t


def phaseExtractions(signal):
    return np.angle(hilbert(signal))
