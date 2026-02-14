import numpy as np
from numpy.fft import fft2, ifft2, fftshift

def gaussfft(pic, t):
    N, M = pic.shape

    y, x = np.meshgrid(np.arange(M), np.arange(N))
    y = y - N * (y > N // 2)
    x = x - M * (x > M // 2)

    G = np.exp(-(x ** 2 + y ** 2) / (2 * t))
    G /= G.sum()

    PicF = fft2(pic)
    Ghat = fft2(G)
    Hhat = PicF * Ghat
    return np.real(ifft2(Hhat))
