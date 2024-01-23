import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import sounddevice as sd
from hoa import *
from time import sleep

########################################## Exercise 10.1 ##########################################
theta1 = np.deg2rad(90)
phi1   = np.deg2rad(45)

theta2 = np.deg2rad(90)
phi2   = np.deg2rad(-45)

direction1 = Direction(theta1, phi1)
direction1.plot()

direction2 = Direction(theta2, phi2)
direction2.plot()

########################################## Exercise 10.2 ##########################################
hoaSig = HoaSignal('scene_HOA_O4_N3D_12k.wav')
# todo code here
sigOmni = hoaSig.sigCoeffTime[:, 0]

# Listen to signal
# How to listen
# sigNoise = np.random.randn(48000)
# sd.play(sigNoise, samplerate=48000)
# sd.play(sigOmni, samplerate=hoaSig.fs)

########################################## Exercise 10.3 ##########################################
#todo code here
direction3 = Direction(np.array([theta1,theta2]), [phi1,phi2])
direction3.plot()
#print(direction3.getThetaInDegrees())

########################################## Exercise 10.4 ##########################################
#todo code here

beamformer = Beamformer.createBeamformerFromHoaSignal(hoaSig)  # Use factory method.
sigBeam = beamformer.beamformSignal(hoaSig, direction3)
plt.plot(sigBeam[0])
plt.show()
#sd.play(sigBeam[0], samplerate=hoaSig.fs)
#sleep(10)
plt.plot(sigBeam[1])
plt.show()
#sd.play(sigBeam[1], samplerate=hoaSig.fs)

########################################## Exercise 10.5 ##########################################

# Steered response power map
numAzim = 160
numIncl = 80
# todo code here
srpMap = SteeredResponsePowerMap(numAzimuths=numAzim, numInclinations=numIncl)

# Iterate over frames, calculate and plot steered response power map
frameLength = 2048
frameAdvance = 1024
nFrames = int(np.floor((hoaSig.numSamples - frameLength) / frameAdvance + 1))

def animate(i):
    sampleRange = i * frameAdvance + np.arange(frameLength)

    # Calculate steered response power map for current sample range
    srpMap.generateSrpMap(hoaSig, sampleRange)
    hplot = srpMap.updatePlot()
    hdot = srpMap.markMaximum()

    # return [hplot, hdot]  #from old version


fig = plt.figure()
# Don't use the scientific mode from PyCharm here!!!
anim = animation.FuncAnimation(fig, animate, frames=nFrames, init_func=srpMap.initPlot, interval=1, repeat=False)
plt.show()

########################################## Exercise 10.6 ##########################################
# Create BinauralRenderer for specified HRIR database
renderer = BinauralRenderer.createBinauralRendererFromHoaSignal(hoaSig, 'hrirs_12k.mat')
sigBinaural = renderer.renderSignal(hoaSig)

# Play binaural signal
a = np.array(sigBinaural)
sd.play(a.T, renderer.fs)
sd.wait()
