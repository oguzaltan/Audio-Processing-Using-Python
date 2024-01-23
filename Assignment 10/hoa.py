import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile, loadmat
from scipy.special import sph_harm
#import librosa
from scipy.signal import oaconvolve


class Direction:

    def __init__(self, theta, phi, direct = []):
        # todo code here
        # if direct:
        #     theta = direct[0]
        #     phi =  direct[1]
        if isinstance(theta, list) or isinstance(theta, np.ndarray):
            self._theta = np.array(theta[:min(len(theta), len(phi))])
            self._phi = np.array(phi[:min(len(theta), len(phi))])
        else:
            self._theta = np.array([theta])
            self._phi = np.array([phi])


    def getThetaInRadians(self):
        return self._theta

    def getPhiInRadians(self):
        return self._phi

    def getThetaInDegrees(self):
        return self._theta * 180 / np.pi

    def getPhiInDegrees(self):
        return self._phi * 180 / np.pi

    def plot(self):
        plt.scatter(self._phi, self._theta)
        plt.show()


class HoaSignal:

    def __init__(self, sFilename=None):
        if sFilename is None:
            sFilename = 'signal_HOA_O4_N3D_12k.wav'

        # Load file
        self.fs, self.sigCoeffTime = wavfile.read(sFilename)
        # Alternatively, one might want to use the librosa API

        # Save properties
        # todo code here
        self.hoaOrder = int(np.sqrt(len(self.sigCoeffTime[0]))-1)  # for n=m=N, we have x = (N+1)^2, so N = sqrt(x)-1
        self.numSamples = len(self.sigCoeffTime)

class Beamformer:

    def __init__(self, hoaOrder=None):
        if hoaOrder is not None:
            self.hoaOrder = hoaOrder

    @classmethod
    def createBeamformerFromHoaSignal(cls, hoaSignal):
        return cls(hoaSignal.hoaOrder)

    def beamformSignal(self, hoaSignal, direction, sample_range=0):
        hoaSig = hoaSignal.sigCoeffTime
        if isinstance(sample_range, int):
            sample_range = np.arange(len(hoaSig))
        hoaSig = hoaSig[sample_range]
        a = self.__createSphericalHarmonicsMatrixFromDirection(direction)
        output = a @ hoaSig.T
        # output = np.matmul(self.__createSphericalHarmonicsMatrixFromDirection(direction), hoaSignal.T)

        return output

    def __createSphericalHarmonicsMatrixFromDirection(self, direction):
        """
        Creates the Spherical Harmonics Matrix for the directions given in the Direction object self.direction
        Parameter handling.
        :return:
        """

        return self.__createSphericalHarmonicsMatrix(direction.getThetaInRadians(),
                                                     direction.getPhiInRadians())

    def __createSphericalHarmonicsMatrix(self, theta, phi):
        """
        creates a spherical harmonics matrix

        :param theta: Column vector of inclination angles in radians
        :param phi: Column vector of azimuth angles in radians
        :return: Y =
                                    n = 0                      n = 1                                    n = N
                                /------------\  /---------------------------------------------\  ... --------------\

                   gamma_1   /  Y_0^0(gamma_1)  Y_1^-1(gamma_1)  Y_1^0(gamma_1)  Y_1^1(gamma_1)  ...  Y_N^N(gamma_1)  \
                   gamma_2   |  Y_0^0(gamma_2)  Y_1^-1(gamma_2)  Y_1^0(gamma_1)  Y_1^1(gamma_1)  ...  Y_N^N(gamma_2)  |
                   gamma_3   |  Y_0^0(gamma_3)  Y_1^-1(gamma_3)  Y_1^0(gamma_1)  Y_1^1(gamma_1)  ...  Y_N^N(gamma_3)  |
                    ...     |     ...             ...              ...             ...          ...     ...          |
                   gamma_Q   \  Y_0^0(gamma_Q)  Y_1^-1(gamma_Q)  Y_1^0(gamma_1)  Y_1^1(gamma_1)  ...  Y_N^N(gamma_Q)  /
        """

        assert(len(theta)==len(phi))
        N = self.hoaOrder
        K = (N+1)**2
        Q = len(theta)

        # Different mapping than usual (https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.sph_harm.html)
        azimRad = phi
        inclRad = theta

        alln = np.array([ i  for i in range(0, N+1) for j in range(0, 2*i+1)])
        allm = np.array([j - i for i in range(0, N+1) for j in range(0, 2*i+1)])

        # repeat orders and degrees for each point q such that all inputs have Q*K elements
        m = np.tile(allm, Q)
        n = np.tile(alln, Q)
        az = np.repeat(azimRad, K)
        incl = np.repeat(inclRad, K)

        Y_cmplx = sph_harm(m, n, az, incl)

        # convert to real SHs
        Y_real = Y_cmplx
        Y_real[m>0] = np.array([-1.0])**m[m>0] * np.sqrt(2) * np.real(Y_real[m>0])
        Y_real[m<0] = np.array([-1.0])**m[m<0] * np.sqrt(2) * np.imag(Y_real[m<0])
        Y_real[np.logical_and(m<0,  (m%2)==0)] = -Y_real[np.logical_and(m<0,  (m%2)==0)]
        Y_real = np.reshape(Y_real, [Q, K])
        Y = np.real(Y_real) # make sure this is really real.

        return Y

# todo code here
class SteeredResponsePowerMap(Beamformer):

    def __init__(self, numAzimuths, numInclinations, hoaOrder = 4, fs = 12e3):
        super(SteeredResponsePowerMap, self).__init__(hoaOrder)

        self.fs = fs

        self.inclinationVec = np.linspace(0, np.pi, numInclinations)
        self.azimuthVec = np.linspace(-np.pi, np.pi, numAzimuths)

        self.numAzimuths = numAzimuths
        self.numInclinations = numInclinations

        self.azimuthGrid, self.inclinationGrid = np.meshgrid(self.azimuthVec, self.inclinationVec)
        self.direction = Direction(self.inclinationGrid.flatten(), self.azimuthGrid.flatten())
        self.numDirection = numInclinations*numAzimuths

        self.sigBeamFramePower = 0

    @classmethod
    def createSteeredResponsePowerMapFromHoaSignal(cls, hoaSignal, numAzimuths, numInclinations):
        return cls(numAzimuths, numInclinations, hoaSignal.hoaOrder, hoaSignal.fs)

    def generateSrpMap(self, hoaSignal, sampleRange=None):
        # Default parameter
        if sampleRange is None:
            sampleRange = np.arange(0, hoaSignal.numSamples)

        self.idxLastSample = sampleRange[-1]
        self.numSamples = hoaSignal.numSamples

        #todo code here
        self.sigBeamFramePower = np.reshape(np.mean(self.beamformSignal(hoaSignal, self.direction, sampleRange)**2, axis=1), (self.numInclinations, self.numAzimuths))

        return 0

    def initPlot(self):
        self.hplot = plt.imshow(np.random.rand(self.numInclinations, self.numAzimuths), interpolation='none', cmap=plt.get_cmap('rainbow'), extent=np.rad2deg([-np.pi, np.pi, np.pi, 0]))

        self.hDot = plt.scatter(0, 0)
        plt.xlabel('azimuth [degree]')
        plt.ylabel('inclination [degree]')


    def updatePlot(self):
        #todo code here
        dataInDb = 10*np.log10(self.sigBeamFramePower)

        self.hplot.set_array(dataInDb)
        self.hplot.set_clim(np.min(dataInDb), np.max(dataInDb))

        plt.title('steered response power map\n time ' + "{:.2f}".format(self.idxLastSample / self.fs) + 's / ' + str(
            self.numSamples / self.fs) + 's')
        # return [self.hplot]  #from old version

    def markMaximum(self):
        #todo code here
        pass
        # return self.hDot  #from old version


class BinauralRenderer(Beamformer):

    def __init__(self, sFilenameHrir='hrirs_12k.mat', hoaOrder=4, fs=12e3):
        super(BinauralRenderer, self).__init__(hoaOrder)

        #todo code here
        everything = loadmat(sFilenameHrir)
        self.theta = everything["theta"]
        self.phi = everything["phi"]
        self.fs = everything["fs"]
        self.hrirs = everything["hrirs"]

    @classmethod
    def createBinauralRendererFromHoaSignal(cls, hoaSignal, sFilenameHrir=None):
        # todo code here
        return cls(sFilenameHrir, hoaSignal.hoaOrder, hoaSignal.fs)

    def renderSignal(self, hoaSignal):
        #todo code here
        direction = Direction(self.theta, self.phi)
        #a = self.beamformSignal(hoaSignal, direction)[:, 0]
        #b = self.hrirs[0, :, 0]
        y_l = np.sum([np.convolve(self.beamformSignal(hoaSignal, direction)[i, :], self.hrirs[i, :, 0]) for i in range(len(self.hrirs))], axis=0)
        y_r = np.sum([np.convolve(self.beamformSignal(hoaSignal, direction)[i, :], self.hrirs[i, :, 1]) for i in range(len(self.hrirs))], axis=0)
        return y_l, y_r
