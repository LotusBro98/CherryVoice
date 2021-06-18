from scipy.io.wavfile import read
from scipy.signal import medfilt

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

spectrum, noise, voice, main_freq, peaks = model.encode_voice(encoder, track)

# frac = noise / frac
# frac = -(frac - 0.5) * 20
# frac = 1 / (1 + tf.exp(-frac))
#
# voice *= frac
# noise *= (1 - frac)

print(voice.shape)

main_freq *= 1.0

voice1 = np.zeros_like(voice)
voice2 = cv.resize(voice, None, None, 1.0, 1)[:,:voice.shape[-1]]
voice1[:,:voice2.shape[-1]] += voice2
voice = voice1

noise1 = np.zeros_like(noise)
noise2 = cv.resize(noise, None, None, 1.0, 1)[:,:noise.shape[-1]]
noise1[:,:noise2.shape[-1]] += noise2
noise = noise1

track2 = model.decode_voice(encoder, noise, voice, main_freq, sample_rate, len(track))
spectrum2, _, _, _, _ = model.encode_voice(encoder, track2)

f, ax = plt.subplots(4, 1, figsize=(40, 20))
for i, data in enumerate([np.abs(spectrum), np.abs(spectrum2), np.abs(voice), np.abs(noise)]):
    ax[i].imshow(data.T[::-1])
plt.show()

f = plt.figure(figsize=(40, 10))
image = np.stack([voice, noise * 10, np.zeros_like(noise)], axis=-1)
plt.imshow((image / np.max(image)).transpose((1, 0, 2))[::-1])
f.tight_layout()
plt.show()

for i in [415, 500, 550, 870, 1000, 1100]:
    plt.plot(np.abs(spectrum[i]))
    plt.plot(np.abs(noise[i]))
    plt.plot(np.abs(voice[i]))
    plt.plot(np.abs(spectrum2[i]))
    plt.plot(np.abs(peaks[i] * 0.02))
    # plt.plot(np.abs(frac[500]))
    plt.show()

plt.figure(figsize=(20, 5))
plt.plot(main_freq)
plt.show()

plt.plot(track)
plt.plot(track2)
plt.show()


# track2 = model.restore(encoder, [spectrum2], len(track))[0]

# spectrum = window_fourier(track, sample_rate)

plt.plot(track[200000:201000])
plt.plot(track2[200000:201000])
# plt.plot(track)
# plt.plot(track2)
plt.show()

print(spectrum.shape)

print(track2.shape)

sd.play(track2, sample_rate, blocking=True)
