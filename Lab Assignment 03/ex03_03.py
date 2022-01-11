import numpy as np
import scipy.fft as fft
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import cmath

x1 = np.zeros(64)
x2 = np.zeros(64)
x3 = np.zeros(64)
x1[0] = 1
x2[1] = 1
x3[2] = 1

def show_fft(x):
    x_ft = fft.fft(x)
    mag_x = np.abs(x_ft)
    ang_x = np.angle(x_ft)

    imag_x = np.imag(x_ft)
    real_x = np.real(x_ft)
    x_len = len(x_ft)

    fig, axs = plt.subplots(2,2)
    axs[0,0].plot(range(x_len), real_x)
    axs[0,0].set_title('Real Part')
    axs[0,1].plot(range(x_len), imag_x)
    axs[0,1].set_title('Imaginary Part')
    axs[1,0].plot(range(x_len), mag_x)
    axs[1,0].set_title('Magnitude')
    axs[1,1].plot(range(x_len), ang_x)
    axs[1,1].set_title('Angle')
    plt.show()

    return mag_x, ang_x, imag_x, real_x

show_fft(x3)

N = 100
n = np.linspace(1, 1000, num=1000)
x4 = np.cos(2*np.pi*n/N)
x5 = np.cos((n+0.2)*2*np.pi/N)

plt.plot(n, x4, label="x4")
plt.plot(n, x5, label="x5")

plt.legend()
plt.show()

show_fft(x4)
show_fft(x5)



