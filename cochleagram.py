import librosa

y, fs = librosa.load("KU.wav", sr=None) 
print(fs)

target_fs = 16000

if fs != target_fs:
    y = librosa.resample(y, orig_sr=fs, target_sr=target_fs)
    fs = target_fs

y = y / (abs(y).max() + 1e-9)


import numpy as np
from scipy.signal import fftconvolve, hilbert
def erb_hz(f):
    return 24.7 * (4.37 * f / 1000 + 1) # our bready and buttery bandy makery

def erb_scale(f):
    return 21.4 * np.log10(1 + 0.00437 * f) # convert it into cochlear coordinates
def inv_erb(e): 
    # convert back to frequencies
    return (10**(e / 21.4) - 1) / 0.00437
def erb_centers(fmin, fmax, n): # place n filters between fmin and fmax
    e = np.linspace(erb_scale(fmin), erb_scale(fmax), n)
    return inv_erb(e)


# Gammatone filter
def gammatone(fs, fc, n=4, dur=0.08):
    t = np.arange(0, int(dur*fs)) / fs
    b = 1.019 * erb_hz(fc)
    g = t**(n-1) * np.exp(-2*np.pi*b*t) * np.cos(2*np.pi*fc*t)
    return g / np.sqrt(np.sum(g**2))


N = 32
centers = erb_centers(100, 10000, N)

coch = []

target_fc = 1000 #Hz 
band_idx = np.argmin(np.abs(centers - target_fc))

y_filt_example = None

for i, fc in enumerate(centers):
    g = gammatone(fs, fc)
    y_filt = fftconvolve(y, g, mode="same")

    if i == band_idx:
        y_filt_example = y_filt.copy()

    env = np.abs(hilbert(y_filt))
    coch.append(np.log1p(env))

# After the loop we should get something that looks like
# coch = [
#   env_1[t],   # band 1
#   env_2[t],   # band 2...
#   env_32[t]   # band 32
# ]

C = np.vstack(coch) # returns C[band, time]


import matplotlib.pyplot as plt



# t = np.arange(len(y)) / fs  # time axis in seconds
# t_zoom = 0.02  # seconds
# idx = t < t_zoom

# plt.figure(figsize=(10, 3))
# plt.plot(t[idx], y[idx])
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.title("Audio waveform (first 20 ms)")
# plt.tight_layout()
# plt.show()




# plt.figure(figsize=(10,4))
# t0, t1 = 30, 45  # seconds
# i0, i1 = int(t0*fs), int(t1*fs)

# plt.imshow(
#     C[:, i0:i1],
#     aspect="auto",
#     origin="lower",
#     extent=[t0, t1, 0, N-1]   # y-axis is band index
# )
# plt.ylabel("Band index")

# # put y-ticks at actual center freqs
# tick_idx = np.linspace(0, N-1, 6).astype(int)
# plt.yticks(tick_idx, [f"{centers[i]:.0f}" for i in tick_idx])

# plt.xlabel("Time (s)")
# plt.title("Cochleagram (Kali Uchis - Tele)")
# plt.colorbar(label="Response")
# plt.tight_layout()
# plt.show()


# # time axis
# t = np.arange(len(y_filt_example)) / fs

# # zoom window (e.g. first 10 ms)
# t_zoom = 0.01
# idx = t < t_zoom

# plt.figure(figsize=(10, 3))
# plt.plot(t[idx], y_filt_example[idx])
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.title("Raw gammatone filter output (≈1 kHz band)")
# plt.tight_layout()
# plt.show()


# env_example = np.abs(hilbert(y_filt_example))

# plt.figure(figsize=(10, 3))
# plt.plot(t[idx], y_filt_example[idx], label="Filtered signal")
# plt.plot(t[idx], env_example[idx], 'r', linewidth=2, label="Envelope")
# plt.xlabel("Time (s)")
# plt.ylabel("Amplitude")
# plt.title("Filtered signal and Hilbert envelope")
# plt.legend()
# plt.tight_layout()
# plt.show()



t = np.arange(len(y_filt_example)) / fs

t0, t1 = 0.0, 0.005  # 5 ms window
i0, i1 = int(t0*fs), int(t1*fs)

plt.figure(figsize=(10, 3))
plt.plot(t[i0:i1], y_filt_example[i0:i1])
plt.axhline(0, linewidth=1)

# 1 kHz period markers 
period = 1/1000
for k in np.arange(t0, t1 + 1e-12, period):
    plt.axvline(k, linestyle="--", linewidth=1)

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Filtered output ~1 cycle per 1 ms")
plt.tight_layout()
plt.show()
