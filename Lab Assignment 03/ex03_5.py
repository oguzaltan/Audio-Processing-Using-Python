# import soundfile as sf
# from scipy.signal import spectrogram
# import matplotlib.pyplot as plt
# import numpy as np
#
# voice2, sr3 = sf.read("voice2.wav")
# print("Duration time is:", len(voice2)/sr3, "s")
#
#
# def compare_specs(x, fs, params_default, **kwargs):
#     fig, ax = plt.subplots(1, 2)
#     params = params_default.copy()
#
#     fbin1, tframe1, S1 = spectrogram(x, fs, **params)
#     ax[0].pcolormesh(tframe1, fbin1, 20 * np.log10(S1))
#
#     for key in kwargs:
#         params[key] = kwargs[key]
#
#     fbin2, tframe2, S2 = spectrogram(x, fs, **params)
#     ax[1].pcolormesh(tframe2, fbin2, 20 * np.log10(S2))
#     fig.tight_layout(pad=3.0)
#     plt.show()
#
# spec_dict = dict(nperseg=512, noverlap=400, nfft=2048, window='rect', scaling='spectrum')
# plot_dict = dict(shading='goraud', vmin=-40)
#
# compare_specs(voice2, sr3, spec_dict, window="blackman")

import soundfile as sf
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import numpy as np

voice2, sr3 = sf.read("voice2.wav")
print("Duration time is:", len(voice2)/sr3, "s")


def compare_specs(x, fs, params_def, **kwargs):
    fig, ax = plt.subplots(1, 2)
    params = params_def.copy()

    fbin1, tframe1, S1 = spectrogram(x, fs, **params)
    ax[0].pcolormesh(tframe1, fbin1, 20 * np.log10(S1))
    ax[0].set_xlabel('t in [s]')
    ax[0].set_ylabel('f in [Hz]')
    ax[0].set_title('Default Spectrogram')
    for key in kwargs:
        params[key] = kwargs[key]

    fbin2, tframe2, S2 = spectrogram(x, fs, **params)
    ax[1].pcolormesh(tframe2, fbin2, 20 * np.log10(S2))
    ax[1].set_xlabel('t in [s]')
    ax[1].set_ylabel('f in [Hz]')
    ax[1].set_title('Spectrogram with changed Values')
    fig.tight_layout(pad=4.0)
    plt.show()


params_default = dict(nperseg=512, noverlap=0, nfft=None, window='rect')
plot_dict = dict(shading='goraud', vmin=-40)

compare_specs(voice2, sr3, params_default, window="blackman")
