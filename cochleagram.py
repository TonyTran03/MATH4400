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
# t0, t1 = 30, 45
# i0, i1 = int(t0 * fs), int(t1 * fs)

# plt.figure(figsize=(10, 4))
# plt.imshow(
#     C[:, i0:i1],
#     aspect="auto",
#     origin="lower",
#     extent=[t0, t1, 0, N - 1]
# )

# plt.ylabel("Band index")

# tick_idx = np.linspace(0, N - 1, 6).astype(int)
# plt.yticks(tick_idx, [f"{centers[i]:.0f}" for i in tick_idx])

# plt.xlabel("Time (s)")
# plt.title("Cochleagram (Kali Uchis - Tele)")
# plt.colorbar(label="Response")
# plt.tight_layout()
# plt.show()


##### _______________________ Pairing with energy profile plot:
t0, t1 = 30, 45
i0, i1 = int(t0 * fs), int(t1 * fs)
low_band = C[0, i0:i1]     # band 1 = lowest frequency
print(low_band.mean(), low_band.max())

erb_cent = erb_scale(centers)  # ERB coordinate for each band

plt.figure(figsize=(10, 4))
plt.imshow(
    C[:, i0:i1],
    aspect="auto",
    origin="lower",
    extent=[t0, t1, 1, N],
    vmin=np.percentile(C, 5),
    vmax=np.percentile(C, 95)
)
plt.xlabel("Time (s)")
plt.ylabel("Channel index (ERB-spaced)")
tick_idx = np.linspace(0, N-1, 8).astype(int)
plt.yticks(tick_idx + 1, [f"Ch {i+1}\n({centers[i]:.0f} Hz)" for i in tick_idx])
plt.title("Cochleagram (30-45s) Telepatia - Kali Uchis")
plt.savefig("cochleagram.png", dpi=300, bbox_inches="tight")

Cs = C[:, i0:i1]
E = Cs.mean(axis=1)
E = E / (E.sum() + 1e-12)

plt.figure(figsize=(9,3))
plt.plot(np.arange(1, N+1), E)

plt.xlabel("Channel index (low → high)")
plt.ylabel("Normalized mean energy")
plt.title("Energy profile over channels (30-45s)")
plt.tight_layout()
plt.savefig("energy_profile.png", dpi=300, bbox_inches="tight")
plt.show()


##### _______________________
"""
Add-on: Frequency-bucket "histograms" from your cochleagram.

Goal:
- Collapse C[band, time] over time -> one value per band.
- Plot bars vs band center frequency (Hz).

This is a proof-of-concept visualization:
Baseline vs (simulated) hearing-loss levels via:
  - ERB broadening (bw_mult)
  - fewer channels (N)
"""

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import fftconvolve, hilbert

# # -----------------------------
# # (A) Slightly upgraded gammatone: add bw_mult (ERB broadening)
# # -----------------------------
# def gammatone(fs, fc, n=4, dur=0.08, bw_mult=1.0):
#     t = np.arange(0, int(dur * fs)) / fs
#     b = bw_mult * 1.019 * erb_hz(fc)  # broadened bandwidth
#     g = t**(n - 1) * np.exp(-2 * np.pi * b * t) * np.cos(2 * np.pi * fc * t)
#     return g / np.sqrt(np.sum(g**2) + 1e-12)


# # -----------------------------
# # (B) Build cochleagram given N channels + bw_mult
# # -----------------------------
# def build_cochleagram(y, fs, fmin=100, fmax=10000, N=32, bw_mult=1.0):
#     centers = erb_centers(fmin, fmax, N)
#     coch = []
#     for fc in centers:
#         g = gammatone(fs, fc, bw_mult=bw_mult)
#         y_filt = fftconvolve(y, g, mode="same")
#         env = np.abs(hilbert(y_filt))
#         coch.append(np.log1p(env))
#     C = np.vstack(coch)
#     return C, centers


# # -----------------------------
# # (C) Collapse over time -> "frequency bucket histogram"
# # -----------------------------
# def freq_bucket_energy(C, fs, centers, t0, t1):

#     i0, i1 = int(t0 * fs), int(t1 * fs)
#     Cs = C[:, i0:i1]

#     E = Cs.mean(axis=1)

#     # Normalize → energy distribution
#     P = E / (E.sum() + 1e-12)

#     order = np.argsort(centers)

#     return centers[order], P[order]

# def plot_freq_hist(f, y, ylabel, title):
#     plt.figure(figsize=(9, 3))
#     # width ~ 8% of each center frequency (works OK on log scale)
#     plt.bar(f, y, width=f * 0.08)
#     plt.xscale("log")
#     plt.xlabel("Frequency (Hz)")
#     plt.ylabel(ylabel)
#     plt.title(title)
#     plt.tight_layout()
#     plt.show()


# # -----------------------------
# # (D) Proof-of-concept: 4 levels (paper-style)
# # -----------------------------
# # Suggested mapping based on your screenshot:
# # Normal: 1.0× ERB, 36 channels
# # Mild:   1.5× ERB, 36 channels
# # Mod:    2.0× ERB, 28 channels
# # Severe: 3.0× ERB, 19 channels
# levels = [
#     ("Normal",   36, 1.0),
#     ("Mild",     36, 1.5),
#     ("Moderate", 28, 2.0),
#     ("Severe",   19, 3.0),
# ]

# # Build and plot histograms for the same time window you used
# t0, t1 = 30, 45
# for name, N_lvl, bw_mult in levels:

#     C_lvl, centers_lvl = build_cochleagram(
#         y, fs,
#         fmin=100,
#         fmax=10000,
#         N=N_lvl,
#         bw_mult=bw_mult
#     )

#     f, vals = freq_bucket_energy(C_lvl, fs, centers_lvl, t0, t1)

#     plot_freq_hist(
#         f,
#         vals,
#         ylabel="Normalized band energy",
#         title=f"{name} ({N_lvl} ch, {bw_mult:.1f}×ERB)"
#     )
