"""
Load a song, run it through an ear-inspired filterbank, and plot a cochleagram.

Cochleagram idea:
- Split the audio into many overlapping “frequency channels” (low → high).
- For each channel, track the signal strength over time (the envelope).
- Stack all channels into a 2D picture: (frequency band) x (time).
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, hilbert


"""
Audio loading + standardizing:
- y = waveform samples (a long array of numbers)
- fs = sample rate (samples per second)
We resample to 16 kHz so timing/filter settings are consistent across files,
then normalize so amplitudes are in a stable range.
"""
y, fs = librosa.load("KU.wav", sr=None)
print(fs)

target_fs = 16000
if fs != target_fs:
    y = librosa.resample(y, orig_sr=fs, target_sr=target_fs)
    fs = target_fs

y = y / (abs(y).max() + 1e-9)


"""
ERB spacing utilities:
ERB (“Equivalent Rectangular Bandwidth”) is a common way to space filters like the ear:
- bands are more densely packed at low frequencies
- bands spread out and get wider at high frequencies

These functions do two jobs:
1) Convert Hz ↔ ERB coordinates (a warped frequency axis).
2) Choose band center frequencies equally spaced on the ERB axis.
"""
def erb_hz(f):
    """Approx ear bandwidth (in Hz) around frequency f (Hz)."""
    return 24.7 * (4.37 * f / 1000 + 1)

def erb_scale(f):
    """Hz → ERB coordinate."""
    return 21.4 * np.log10(1 + 0.00437 * f)

def inv_erb(e):
    """ERB coordinate → Hz."""
    return (10**(e / 21.4) - 1) / 0.00437

def erb_centers(fmin, fmax, n):
    """Pick n center frequencies between fmin and fmax, evenly spaced on the ERB axis."""
    e = np.linspace(erb_scale(fmin), erb_scale(fmax), n)
    return inv_erb(e)


"""
Gammatone filter:
A gammatone is a standard ear-inspired bandpass filter shape.
Given a center frequency fc, it produces a short “ringing” impulse response.
Convolving audio with this impulse response extracts the content near fc.
"""
def gammatone(fs, fc, n=4, dur=0.08):
    t = np.arange(0, int(dur * fs)) / fs
    b = 1.019 * erb_hz(fc)  # bandwidth term (wider at higher fc)
    g = t**(n - 1) * np.exp(-2 * np.pi * b * t) * np.cos(2 * np.pi * fc * t)
    return g / np.sqrt(np.sum(g**2))  # normalize so bands are comparable


"""
Build the cochleagram:
- Choose N ERB-spaced center frequencies between 100 and 10,000 Hz.
- For each center fc:
    - filter the audio with a gammatone filter
    - compute the amplitude envelope via Hilbert transform
    - log-compress (helps visibility + mimics dynamic range compression)
Result:
- C is a 2D array shaped (band, time)
"""
N = 32
centers = erb_centers(100, 10000, N)

coch = []

target_fc = 1000  # optional: keep one “example” band if we want to inspect it later
band_idx = np.argmin(np.abs(centers - target_fc))
y_filt_example = None

for i, fc in enumerate(centers):
    g = gammatone(fs, fc)
    y_filt = fftconvolve(y, g, mode="same")

    if i == band_idx:
        y_filt_example = y_filt.copy()

    env = np.abs(hilbert(y_filt))
    coch.append(np.log1p(env))

C = np.vstack(coch)  # C[band, time]


"""
Plot a time slice of the cochleagram:
imshow shows C as an image:
- x-axis: time (seconds)
- y-axis: band index (we relabel ticks with the band center frequencies in Hz)
- color: “response” (log(1 + envelope))
"""
t0, t1 = 30, 45
i0, i1 = int(t0 * fs), int(t1 * fs)

plt.figure(figsize=(10, 4))
plt.imshow(
    C[:, i0:i1],
    aspect="auto",
    origin="lower",
    extent=[t0, t1, 0, N - 1]
)

plt.ylabel("Band index")

tick_idx = np.linspace(0, N - 1, 6).astype(int)
plt.yticks(tick_idx, [f"{centers[i]:.0f}" for i in tick_idx])

plt.xlabel("Time (s)")
plt.title("Cochleagram (Kali Uchis - Tele)")
plt.colorbar(label="Response")
plt.tight_layout()
plt.show()
