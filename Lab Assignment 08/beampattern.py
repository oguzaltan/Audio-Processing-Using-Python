''' This script serves as a template for experiment 8 to
 calculate the 'Array-Gains' and the 'White-Noise Gains' using Delay-and-Sum Beamformers

 Hint : The places marked by todos should be replaced or completed by your own code
 This m-file is not yet executable!
'''

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import cm
from scipy.fft import fft

plt.ion()
### Simulation parameters ###

# General Parameters

c = 340  # Sound velocity
fs = 16000  # Sampling frequency

# Variable Parameters
n = 8  # Number of sensors
d = 0.0425  # Distance between adjacent sensors
d1 = 0.2
dist = np.linspace(0, (n - 1) * d, n, endpoint=True)  # Distances to reference microphone
theta = 0  # todo configure theta according to experiment

# Parameter for graphic representation of the beamformer

'''Please declare all the variables that you need for the implementation.
The variables Bw and Bf are already given
Please calculate the angles and frequencies with the help of these
variables'''

bw = 200  # number of discrete angles
bw2 = np.round(bw / 2)  # rounded for better legibility

angles = np.linspace(-np.pi, np.pi, bw)  # Set of all angles (in Rad)
angles_deg = 180 / np.pi * angles  # ... in Deg

bf = 256  # Number of all the discrete frequency points
bf2 = int(round(bf / 2))  # rounded for better legibility

freq = np.linspace(0, fs / 2, bf2)  # Set of all discrete angles

th = -25  # db-threshold for plots, for better graphic representation
epsi = 1e-10  # (In order to avoid rounding errors)

# Calculating the directivity pattern
# =============================================================

# Calculating B(Omega)
#tau_vec = np.zeros(n)
coeff = np.zeros((bf, n))
for i in range(n):
    tau = dist[i] / c * np.cos(theta) * fs  # time delay
    #tau_vec[i] = tau
    arg = np.arange(-bf2 - tau + 1, bf2 - tau + epsi)
    coeff[:, i] = np.sinc(arg[:bf])  # fractional delay filter

b = 1 / n * fft(coeff, axis=0).T

# Calculating Psi(Omega, theta)
psi = np.zeros((bf2, bw))
psi_db = np.zeros((bf2, bw))

for a in range(bw):  # angle index
    tau_vec = [dist[i] / c * np.cos(angles[a]) for i in range(n)]
    for f in range(bf2):  # frequency index
        # Calculating array-steering vector with dimension 1xN
        # todo: e = ?
        omega = 2*np.pi*freq[f]/fs
        e = np.array([np.exp(1j*fs*omega*t) for t in tau_vec]).T

        # Calculating directivity pattern (Matrix dimension is Bf2xBw)
        # todo: psi[f, a] =
        # todo: psi_db[f, a] =
        psi[f, a] = np.abs(np.dot(e, b[:, f]))**2
        psi_db[f, a] = 10*np.log10(psi[f, a])

        if psi_db[f, a] < th:
            # for better graphic representation
            psi_db[f, a] = th

# Hint: Loops should generally be avoided and replaced by
# matrix operations. We used loops here only to help you to understand the
# calculation procedure.

# Plot the directivity pattern
# =============================================================

# 3 dimensional representation (surface plot)
fig1 = plt.figure()
ax1 = fig1.gca(projection='3d', title="3D representation d=4.25cm theta=0°", xlabel="Angle", ylabel="Frequency")
# grid_theta, grid_f = ...  # todo: coordinate mesh grid for surface plot
grid_theta, grid_f = np.meshgrid(angles_deg, freq)

plot_args = dict(rcount=bf, ccount=bw, cmap=cm.jet, linewidth=1, antialiased=True)
surf = ax1.plot_surface(grid_theta, grid_f, psi_db, **plot_args)
# todo: set correct title and labels

ax1.view_init(40, 120)
plt.show()


fig2 = plt.figure()
ax2 = fig2.gca(title="2D View d=4.25cm theta=0°", xlabel="Angle", ylabel="Frequency")
# plt.pcolormesh(...)  # todo plot 2D view
# todo: set correct title and labels
plt.pcolormesh(grid_theta, grid_f, psi_db)

plt.show()