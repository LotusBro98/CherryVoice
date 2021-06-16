from scipy.signal import medfilt
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

TIME_RES = 0.01
FREQ_RES = 10
MAX_FREQ = 15000
MIN_VOICE_FREQ = 200
MIN_VOICE_FREQ_AVG = 400
MAX_PEAKS = 50



def restore(self: tf.keras.Model, x, out_len):

    x = tf.concat([tf.math.real(x), tf.math.imag(x)], axis=-1)
    x = tf.cast(x, tf.float32)

    b, s, c = x.shape
    out_shape = (b, out_len, 1)

    x = tf.nn.conv1d_transpose(x, self.F_T, output_shape=out_shape, strides=self.stride, padding="SAME")
    x = x[:,:,0]

    return x


def get_F_kernel(sample_rate):
    n_samples = int(MAX_FREQ / FREQ_RES) // 2 * 2 + 1
    stride = int(sample_rate * TIME_RES)
    dilations = int(sample_rate / MAX_FREQ)
    n_samples_T = dilations * n_samples // 2 * 2 + 1

    x = np.arange(-(n_samples - 1) // 2, (n_samples - 1) // 2 + 1, dtype=np.float32)
    k = np.arange(1, (n_samples - 1) // 2 + 1, dtype=np.float32)
    kx = np.outer(k, x)
    F = np.exp(-2j * np.pi / n_samples * kx)
    F *= np.sin(np.linspace(0, np.pi, F.shape[-1]))
    F = np.concatenate([np.real(F), np.imag(F)], axis=0)
    F /= np.linalg.norm(F, axis=-1, keepdims=True)
    F = np.expand_dims(F, axis=-1)
    F = np.transpose(F, (1, 2, 0))

    x_T = np.arange(-(n_samples_T - 1) // 2, (n_samples_T - 1) // 2 + 1, dtype=np.float32)
    kx_T = np.outer(k, x_T)
    F_T = np.exp(-2j * np.pi / n_samples_T * kx_T)
    F_T = np.concatenate([np.real(F_T), np.imag(F_T)], axis=0)
    F_T /= np.linalg.norm(F_T, axis=0, keepdims=True)
    F_T = np.expand_dims(F_T, axis=-1)
    F_T = np.transpose(F_T, (1, 2, 0))

    # F *= np.square(np.arange(1, F.shape[-1]+1))
    # F_T /= np.square(np.arange(1, F.shape[-1]+1))

    mask = np.ones((n_samples_T,), dtype=np.float32)
    # window = np.sin(np.linspace(0, np.pi, n_samples_T))
    window = np.ones((n_samples_T,))
    window /= np.average(window)
    for i in range(stride, n_samples_T, stride):
        mask[i:] += window[i:]
        mask[:-i] += window[:-i]

    F_T /= np.expand_dims(mask, axis=(1, 2))

    return F, F_T


def find_peaks(spectrum, k=0.1):
    pool_size = int(MIN_VOICE_FREQ / FREQ_RES)
    avg_pool_size = int(MIN_VOICE_FREQ_AVG / FREQ_RES)

    spec_abs = tf.transpose(tf.abs(spectrum), (0, 2, 1))

    max_pool = tf.keras.layers.MaxPool1D(pool_size, strides=1, padding="same")(spec_abs)

    noise = 1 / tf.keras.layers.AvgPool1D(avg_pool_size, strides=1, padding="same")(1 / spec_abs)
    # noise = tf.keras.layers.AvgPool1D(pool_size, strides=1, padding="same")(noise)

    voice = tf.sqrt(tf.keras.layers.AvgPool1D(avg_pool_size, strides=1, padding="same")(tf.square(max_pool)))
    # voice = tf.keras.layers.AvgPool1D(pool_size, strides=1, padding="same")(voice)

    frac = noise / voice
    frac = (k - frac) * 20
    frac = 1 / (1 + tf.exp(-frac))

    voice *= frac

    mask = tf.cast((spec_abs == max_pool) & (voice > noise), tf.float32)

    voice = tf.transpose(voice, (0, 2, 1))
    noise = tf.transpose(noise, (0, 2, 1))
    mask = tf.transpose(mask, (0, 2, 1))

    peaks = mask

    return noise, voice, peaks


def find_main_freq(peaks):
    freq = np.linspace(0, MAX_FREQ / 2, peaks.shape[-1])
    delta = np.maximum.accumulate(freq * peaks, axis=-1)
    delta = delta - np.roll(delta, 1, axis=-1)
    delta[:, 0] = 0


    delta1 = delta.copy()
    delta1[np.abs(delta) < FREQ_RES] = np.nan
    main_freq = np.nanmedian(delta1, axis=-1)

    delta[np.abs(delta - np.expand_dims(main_freq, -1)) > 5 * FREQ_RES] = 0
    main_freq = np.exp(np.sum(np.log(1 + delta), axis=-1) / (np.sum(delta != 0, axis=-1) + 1e-1)) - 1

    main_freq = medfilt(main_freq, 3)
    return main_freq


def encode_voice(encoder, track):
    spectrum, noise, voice, peaks = encoder.predict(np.expand_dims(track, axis=(0, -1)))

    spectrum = spectrum[0]
    peaks = peaks[0]
    noise = noise[0]
    voice = voice[0]

    main_freq = find_main_freq(peaks)

    return spectrum, noise, voice, main_freq

def decode_voice(encoder, noise, voice, main_freq):
    noise = noise * (np.random.normal(size=noise.shape) + 1j * np.random.normal(size=noise.shape)) / np.sqrt(2)

    peaks = np.zeros_like(voice)
    main_freq = main_freq / FREQ_RES
    indices = np.int32(np.outer(main_freq, np.arange(1, MAX_PEAKS)))
    indices = np.clip(indices, 0, peaks.shape[-1] - 1)
    for i in range(peaks.shape[0]):
        peaks[i, indices[i]] = 1

    peaks = peaks * np.exp(2j * np.pi * np.random.uniform(0, 1, size=peaks.shape))
    voice = voice * peaks

    spectrum = voice + noise
    return spectrum



def Model(sample_rate):
    stride = int(sample_rate * TIME_RES)
    dilations = int(sample_rate / MAX_FREQ)
    F, F_T = get_F_kernel(sample_rate)

    input = tf.keras.layers.Input((None,))
    x = input
    x = tf.expand_dims(x, axis=-1)

    x = tf.nn.conv1d(x, F, stride=stride, padding="SAME", dilations=dilations)

    x, y = tf.split(x, 2,  axis=-1)
    x = tf.cast(tf.complex(x, y), tf.complex64)
    spectrum = x

    noise, voice, peaks = find_peaks(spectrum)

    model = tf.keras.Model(inputs=input, outputs=[spectrum, noise, voice, peaks])
    model.F = F
    model.F_T = F_T
    model.stride = stride
    model.sample_rate = sample_rate

    return model