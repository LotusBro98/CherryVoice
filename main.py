from scipy.io.wavfile import read

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import cv2 as cv

import model
from model import Model


sample_rate, track = read("data/temp.wav")
track = track[:,0]
track = track / 2**15


encoder = Model(sample_rate)

spectrum, noise, voice, main_freq = model.encode_voice(encoder, track)

spectrum2 = model.decode_voice(encoder, noise, voice, main_freq)

for data in [np.abs(spectrum), np.abs(voice), np.abs(noise), np.abs(spectrum2)]:
    plt.figure(figsize=(20, 20))
    plt.imshow(data.T[::-1])
    plt.show()

plt.plot(np.abs(spectrum[500]))
plt.plot(np.abs(noise[500]))
plt.plot(np.abs(voice[500]))
plt.show()

plt.plot(np.abs(spectrum[550]))
plt.plot(np.abs(noise[550]))
plt.plot(np.abs(voice[550]))
plt.show()

plt.figure(figsize=(20, 5))
plt.plot(main_freq)
plt.show()


track2 = model.restore(encoder, [spectrum2], len(track))[0]

# spectrum = window_fourier(track, sample_rate)

plt.plot(track[100000:101000])
plt.plot(track2[100000:101000])
# plt.plot(track)
# plt.plot(track2)
plt.show()

print(spectrum.shape)

print(track2.shape)

sd.play(track2, sample_rate, blocking=True)
