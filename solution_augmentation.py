import numpy as np
import random
import matplotlib.pyplot as plt

import os
from shutil import rmtree
from tqdm import tqdm
from time import sleep
import soundcard as sc

import librosa
from scipy import signal
from sklearn.mixture import GaussianMixture


class GenSpeech:
    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.files = os.listdir(path)
        self.path_list = []

        for root, directories, files in os.walk(path, topdown=True):
            for filename in files:
                # Join the two strings in order to form the full filepath.
                filepath = os.path.join(root, filename)
                self.path_list.append(filepath)  # Add it to the list.

    def __len__(self):
        return len(self.path_list)

    def __iter__(self):
        return self

    def __next__(self):
        if self.path_list:
            return librosa.load(self.path_list.pop(), sr=16000)
        return


def create_gen_noise(path, len_chunk):
    # todo: your code here

    files = os.listdir(path)
    file_count = len(files)
    indexes = list(np.arange(0, file_count))

    while True:
        rand_index = random.choice(indexes)
        file, len_file = librosa.load(os.path.join(path, files[rand_index]), sr=16000)

        while len(file) > len_chunk:
            chunk = file[:len_chunk]
            file = file[len_chunk:]
            len_file -= len_chunk
            yield chunk
        #if len(file) < len_chunk:
            #pass


def create_gen_rirs(path):
    # todo: your code here

    files = os.listdir(path)
    file_count = len(files)
    indexes = list(np.arange(0, file_count))

    while True:
        rand_index = random.choice(indexes)
        file, len_file = librosa.load(os.path.join(path, files[rand_index]), sr=16000)
        yield file


def mix(clean, noise, snr, rir, biquad):
    # todo: your code here

    clean = signal.convolve(clean, rir[0], mode='same')
    noise = signal.convolve(noise, rir[1], mode='same')

    before_snr = 10 * np.log10(np.sum(np.square(clean)) / np.sum(np.square(noise)))
    a = np.sqrt(np.sum(np.square(clean))/(np.sum(np.square(noise))*10**(snr/10)))
    actual_snr = 10 * np.log10(np.sum(np.square(clean)) / np.sum(np.square(a * noise)))
    #print('We got snr of: ', snr, 'and at the end we have snr of: ', actual_snr, 'at the start snr: ', before_snr)

    mixed = clean + a*noise

    if any(np.abs(mixed)>1):
        clean = clean/np.max(np.abs(mixed))
        mixed = mixed/np.max(np.abs(mixed))

    mixed = signal.sosfilt(biquad, mixed, axis=-1, zi=None)
    clean = signal.sosfilt(biquad, clean, axis=-1, zi=None)

    plt.plot(mixed, label="Mixed")
    plt.plot(clean, label="Clean")
    plt.title("Mixed and Signal")
    plt.legend()
    plt.show()

    # sc.play(mixed/np.max(mixed), samplerate=16000)

    return clean, mixed


def feature_extraction(noisy):
    # todo: your code here:

    mfcc = librosa.feature.mfcc(y=noisy, sr=16000, n_mfcc=40, n_fft=512, hop_length=160, win_length=320, window='hamming', power=1)
    mfcc_delta1 = librosa.feature.delta(mfcc, order=1)[:12]
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)[:6]

    single_matrix = np.concatenate([mfcc, mfcc_delta1, mfcc_delta2])
    return single_matrix

def vad_extraction(clean):
    # todo: your code here

    log_power_clean = librosa.feature.rms(clean, frame_length=320, hop_length=160)
    gmm = GaussianMixture(n_components=2)
    gmm.fit(log_power_clean.T)
    predicted = gmm.predict(log_power_clean.T)
    index = np.argmax(gmm.means_)
    posterior_prob = gmm.predict_proba(log_power_clean.T)[:, index]

    plt.plot(clean/np.max(np.abs(clean)))
    plt.plot(np.repeat(posterior_prob, 160))
    plt.show()

    return predicted, posterior_prob

def run_augmentation():
    # todo: your code here
    path_noise_train = 'D:/RWTH Aachen/Audio Processing Using Python/Lab Assignment 06/Training Data Lab Course 6/Noise/train/'
    path_rir = 'D:/RWTH Aachen/Audio Processing Using Python/Lab Assignment 06/Training Data Lab Course 6/RIR/'
    path_speech = 'D:/RWTH Aachen/Audio Processing Using Python/Lab Assignment 06/Training Data Lab Course 6/speech/'
    sr = 16000
    # chunks_noise_train = [chunk for chunk in create_gen_noise(path_noise_train, 16000)]
    # chunks_rir = [chunk for chunk in create_gen_rirs(path_ri

    # clean = [chunk for chunk in create_gen_noise(path_speech, 16000)]
    # # noise = librosa.load('D:/RWTH Aachen/Audio Processing Using Python/Lab Assignment 06/Training Data Lab Course 6/Noise/test/1-7974-A-49.wav')
    # snr = 5
    # clean1, mixed1 = mix(clean[0], chunks_noise_train[0], snr, chunks_rir[:2])

    gs = GenSpeech(path_speech)
    gn = create_gen_noise(path_noise_train, sr)
    gr = create_gen_rirs(path_rir)
    #gs = ['D:/RWTH Aachen/Audio Processing Using Python/Lab Assignment 06/Training Data Lab Course 6/speech/p225/p225_008.wav']
    noise = []
    i=0
    # Create a noise same length as speech
    for speech_file, speech_sr in gs:
        #speech_file, _ = librosa.load(speech_file)
        needed_chunks = np.ceil(len(speech_file) / sr)
        for i in range(int(needed_chunks)):
            noise.extend(next(gn))
        noise = noise[:len(speech_file)]

        rir1 = next(gr)
        rir2 = next(gr)
        while (np.array_equal(rir1,rir2)):
            rir2 = next(gr)

        a1 = np.random.uniform(-3 / 8, 3 / 8)
        a2 = np.random.uniform(-3 / 8, 3 / 8)
        b1 = np.random.uniform(-3 / 8, 3 / 8)
        b2 = np.random.uniform(-3 / 8, 3 / 8)
        biquad = [1, b1, b2, 1, a1, a2]
        snr = np.random.random() * 12

        clean, mixed = mix(speech_file, noise, snr, [rir1, rir2], biquad)
        single_matrix = feature_extraction(mixed)
        predictions, posterior_prob = vad_extraction(clean)
        np.savez("D:/RWTH Aachen/Audio Processing Using Python/Lab Assignment 06/Training Data Lab Course 6/posterior_prob{}.npz".format(i), posterior_prob)
        np.savez("D:/RWTH Aachen/Audio Processing Using Python/Lab Assignment 06/Training Data Lab Course 6/clean{}.npz".format(i), clean)
        np.savez("D:/RWTH Aachen/Audio Processing Using Python/Lab Assignment 06/Training Data Lab Course 6/mixed{}.npz".format(i), mixed)
        i += 1

if __name__ == '__main__':
    run_augmentation()
