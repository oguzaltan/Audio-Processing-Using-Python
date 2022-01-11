import soundfile as sf
from scipy.signal import correlate, correlation_lags
import matplotlib.pyplot as plt
import numpy as np

distorted, sr2 = sf.read("distorted.wav")
print("Duration time is:", len(distorted)/sr2, "s")

f1 = sr2*1200/len(distorted)  # f = fs*peak/sr
f2 = sr2*1250/len(distorted)
f3 = sr2*2500/len(distorted)

plt.plot(np.arange(len(distorted)), distorted)
plt.show()
