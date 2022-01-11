import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import loadmat


def nlms4echokomp(x, g, noise, alpha, mh):
    """ The MATLAB function 'nlms4echokomp' simulates a system for acoustic echo compensation using NLMS algorithm
    :param x:       Input speech signal from far speaker
    :param g:       impluse response of the simulated room
    :param noise:   Speech signal from the near speaker and the background noise(s + n)
    :param alpha:   Step size for the NLMS algorithm
    :param mh:      Length of the compensation filter

    :return s_diff:  relative system distance in dB
    :return err:    error signal e(k)
    :return x_hat:  output signal of the compensation filter
    :return x_tilde:acoustic echo of far speakers
    """

    # Initialization of all the variables
    lx = len(x)  # Length of the input sequence
    mg = len(g)  # Length of the room impulse response(RIR)
    if mh > mg:
        mh = mg
        import warnings
        warnings.warn('The compensation filter is shortened to fit the length of RIR!', UserWarning)

    # Vectors are initialized to zero vectors.
    x_tilde = np.zeros(lx - mg)
    x_hat = x_tilde.copy()
    err = x_tilde.copy()
    s_diff = x_tilde.copy()
    h = np.zeros(mh)

    # Realization of NLMS algorithm
    k = 0
    for index in range(mg, lx):
        # Extract the last mg values(including the current value) from the
        # input speech signal x, where x(i) represents the current value.
        # todo your code
        x_block = x[index-mg+1:index+1]
        x_blockh = x[index-mg+1:index+1-mg+mh]
        # Filtering the input speech signal using room impulse response and adaptive filter. Please note that you don't
        # need to implement the complete filtering here. A simple vector manipulation would be enough here
        # todo your code:
        x_tilde[k] = np.dot(g, x_block) + noise[k]
        x_hat[k] = np.dot(h, x_blockh)

        # Calculating the estimated error signal
        # todo your code
        err[k] = x_tilde[k]-x_hat[k]

        # Updating the filter
        # todo your code
        beta = alpha/np.linalg.norm(x_blockh)**2
        h = h + beta*err[k]*x_blockh

        # Calculating the relative system distance
        # todo your code
        s_diff[k] = ((np.linalg.norm(g-np.pad(h, (0, mg-mh))))**2)/(np.linalg.norm(g)**2)
        # h = h_new
        k = k + 1  # time index

    # todo your code
    s_diff = 10 * np.log10(s_diff[:k] / 1).T

    # Calculating the relative system distance in dB
    return s_diff, err, x_hat, x_tilde


# switch between exercises
exercise = 7 # choose between 1-7

f = np.load('echocomp.npz')
# g for exercise 1-6
# g = [f['g1'], f['g1'], f['g1']]
# G for exercise 7
g = [f['g1'], f['g2'], f['g3']]
s = f['s']

# Generation of default values
alpha = 0.1  # Step size for NLMS

ls = len(s)  # Length of the speech signal
n0 = np.sqrt(0.16) * np.random.randn(ls)  # White Noise
s = s / np.sqrt(s.T.dot(s)) * np.sqrt(n0.T.dot(n0))  # Number of curves in each plot (should not be changed)
vn = 3  # number of curves
noise = [np.zeros(ls,) for i in range(vn)]  # no disturbance by noise
alphas = [alpha for i in range(vn)]  # Step size factor for different exercises
mh = len(g[0]) * np.ones(vn, dtype=int)  # Length of the compensation filter


x = [n0.copy() for i in range(vn)]  # white noise as input signal

# In the following part, the matrices and vectors must be adjusted to
# meet the requirement for different exercises
# (Exercise 1 can be simulated using only the initialized values above)

if exercise == 2:
    # Only the value of input speech signal need to be changed. All the other
    # vectors and parameters should not be modified

    x[0] = s  # Speech signal

    # todo your code
    x[1] = np.sqrt(0.16) * np.random.randn(ls)  # White Noise
    x[2] = signal.lfilter([1], [1, -0.5], x[1])     #colorful noise

    leg = ('Speech', 'white noise', 'colorful noise')
    title = 'Different Input Signals'

elif exercise == 3:
    # todo your code
    noise[0] = np.sqrt(0) * np.random.randn(ls)
    noise[1] = np.sqrt(0.001) * np.random.randn(ls)
    noise[2] = np.sqrt(0.01) * np.random.randn(ls)
    leg = ["Variance of 0", "Variance of 0.001", "Variance of 0.01"]
    title = "Relative System Distances for different Variances"

elif exercise == 4:
    # todo your code
    x[0] = s
    x[1] = s
    x[2] = s
    noise[0] = np.sqrt(0) * np.random.randn(ls)
    noise[1] = np.sqrt(0.001) * np.random.randn(ls)
    noise[2] = np.sqrt(0.01) * np.random.randn(ls)
    leg = ["Variance of 0", "Variance of 0.001", "Variance of 0.01"]
    title = "Relative System Distances for different Variances"

elif exercise == 5:
    # todo your code
    #noise = [np.sqrt(0.01) * np.random.randn(len(s)) for i in range(vn)]
    noise[0] = np.sqrt(0.01) * np.random.randn(ls)
    noise[1] = noise[0].copy()
    noise[2] = noise[0].copy()
    alphas[0] = 0.1
    alphas[1] = 0.5
    alphas[2] = 1.0
    leg = ["Stepsize of 0.1", "Stepsize of 0.5", "Stepsize of 1.0"]
    title = "Relative System Distances for Different Stepsizes"

elif exercise == 6:
    # todo your code
    mh[0] = len(g[0])-10
    mh[1] = len(g[1])-30
    mh[2] = len(g[2])-60

    leg = ["m = m'-10", "m = m'-30", "m = m'-60"]
    title = "Relative System Distances for different lengths of m"


elif exercise == 7:
    # todo your code
    mh[0] = len(g[0])
    mh[1] = len(g[1])
    mh[2] = len(g[2])
    leg = ["g1", "g2", "g3"]
    title = "Relative System Distances for Different RIRs"

# There should be appropriate legends and axis labels in each figure!
if exercise == 1:
    s_diff, e, x_h, x_t = nlms4echokomp(n0, g[0], np.zeros(ls), alpha, 200)

    fig, axs = plt.subplots(3)
    # todo your code for ex. 1

    lx = len(n0)
    mg = len(g[0])
    erle = np.zeros(lx)
    for k in range(mg, lx):
        #erle[k] = 10 * np.log10(x_t[k]**2/(x_t[k]-x_h[k])**2)
        erle[k] = 10 * np.log10((np.mean(x_t[-200+k+1:k+1]**2))/(np.mean((x_t[-200+k+1:k+1]-x_h[-200+k+1:k+1])**2)))

    axs[0].plot(x_t, label="Echo Signal")
    axs[0].plot(e, label="Residual Echo")
    axs[0].set_xlabel('k')
    axs[0].legend(loc='lower right')

    axs[1].plot(s_diff, label="Relative System Distance")
    axs[1].set_xlabel('k')
    axs[1].set_ylabel('D(k) [dB]')
    axs[1].legend(loc='lower left')

    axs[2].plot(erle, label="ERLE Measure")
    axs[2].set_xlabel('k')
    axs[2].set_ylabel('ERLE(k) [dB]')
    axs[2].legend(loc='upper left')
    plt.show()
else:
    for i in range(vn):
        # 3 system distances with different parameters are calculated here
        # The input variables of 'nlms4echokomp' must be adapted according
        # to different exercises.

        s_diff, e, x_h, x_t = nlms4echokomp(x[i], g[i], noise[i], alphas[i], mh[i])
        plt.plot(s_diff, label=leg[i])

    plt.title('Exercise ' + str(exercise) + ': ' + title)
    plt.xlabel('k')
    plt.ylabel('D(k) [dB]')
    plt.grid(True)
    plt.legend()
    plt.show()
