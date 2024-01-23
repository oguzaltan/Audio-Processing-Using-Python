
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
import numpy as np
from scipy.fft import fft


# Simulation parameters
# ==========================================

# General Parameter:

c = 340  # Sound velocity (m/s)
fs = 16e3  # Sampling frequency
n = 8  # Number of sensors
theta = 0  # endfire array
d = 0.0425  # Distance between adjacent sensors

# Fixed Parameters
epsi = 1e-10  # (in order to prevent from problems caused by rounding)
dist = np.arange(0, (n - 1) * d + epsi, d)  # Abstand zum Referenzmikrofon
bf = 128  # Number of discrete frequency points
bf2 = int(np.round(bf / 2))
freq = np.arange(0, bf2) / bf * fs  # Vector containing all the discrete frequency points

# Calculate the gains of DS Beamformers
# =============================================================

# Calculate B(Omega)
coeff = np.zeros((bf, n))
for i in range(n):
    #tau = ...  # todo
    tau = dist[i] / c * np.cos(theta) * fs
    arg = np.arange(-bf2 - tau + 1, bf2 - tau + epsi)
    coeff[:, i] = np.sinc(arg[:bf])  # fractional delay filter

b = 1 / n * fft(coeff, axis=0).T

# Calculate Psi(Omega,theta0)
psi = np.zeros(bf2)

tau_vec = [dist[i] / c * np.cos(0) for i in range(n)]
omega = np.array([2 * np.pi * freq[f] / fs for f in range(bf2)])

for f in range(bf2): # frequency index
    # Calculate array-steering vector with dimension 1xN
    #e = ...  # todo
    #omega = 2 * np.pi * freq[f] / fs
    e = np.array([np.exp(1j * fs * omega[f] * t) for t in tau_vec]).T

    # Calculate directivity pattern (Matrix dimension is Bf2xBw)
    #psi[f] = ...  # todo
    psi[f] = np.abs(np.dot(e, b[:, f])) ** 2
    #psi_db[f, a] = 10 * np.log10(psi[f, a])

arg = (2 * freq * d / c).T
b = np.conj(b).T

s = np.zeros(bf2)
swn = np.zeros(bf2)


for i in range(n):
    swn +=    # todo
    for m in range(n):
        dmn = d*np.abs(m-i)
        s += [np.sinc(omega[f]*fs*dmn/(np.pi*c)) for f in range(bf2)]  # todo


g = 10 * np.log10(...)  # todo: Calculate Array Gain
gwn = ...  # todo: Calculate White Noise Gain

plt.plot(freq, g)
plt.title('Array Gain (Broadside, d = ' + str(d) + 'cm)')
plt.xlabel('Gain [db]')
plt.ylabel('Frequency [hertz]')
plt.grid(True)
plt.show()

plt.plot(freq, gwn)
plt.title('White Noise Gain (Broadside, d = ' + str(d) + 'cm)')
plt.xlabel('Gain [db]')
plt.ylabel('Frequency [hertz]')
plt.grid(True)
plt.show()
