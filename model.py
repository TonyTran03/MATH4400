import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, hilbert, chirp

# ---------- ERB utilities ----------
def erb_hz(f_hz: float) -> float:
    # Glasberg & Moore ERB approximation (Hz)
    return 24.7 * (4.37 * (f_hz / 1000.0) + 1.0)

def erb_scale(f_hz: float) -> float:
    # ERB-number (aka ERB-rate) scale
    return 21.4 * np.log10(1.0 + 0.00437 * f_hz)

def inv_erb_scale(e: float) -> float:
    return (10**(e / 21.4) - 1.0) / 0.00437

def erb_spaced_center_freqs(f_min: float, f_max: float, n_filters: int) -> np.ndarray:
    e_min = erb_scale(f_min)
    e_max = erb_scale(f_max)
    e = np.linspace(e_min, e_max, n_filters)
    return inv_erb_scale(e)

# ---------- Gammatone impulse response ----------
def gammatone_ir(fs: int, f_c: float, n: int = 4, dur_s: float = 0.100) -> np.ndarray:
    """
    Real-valued gammatone impulse response.
    g(t) = t^(n-1) * exp(-2*pi*b*t) * cos(2*pi*f_c*t)
    with b = 1.019 * ERB(f_c)
    """
    t = np.arange(0, int(dur_s * fs)) / fs
    b = 1.019 * erb_hz(f_c)  # standard mapping used in many auditory models
    g = (t ** (n - 1)) * np.exp(-2.0 * np.pi * b * t) * np.cos(2.0 * np.pi * f_c * t)

    # normalize to avoid crazy scaling differences across channels
    g = g / (np.sqrt(np.sum(g**2)) + 1e-12)
    return g

# ---------- Demo signal ----------
fs = 16000
T = 2.0
t = np.arange(0, int(T * fs)) / fs

# Chirp is great for demos (it sweeps through frequency bands)
s = chirp(t, f0=100, f1=8000, t1=T, method="logarithmic")
# fade in/out a bit to reduce edge artifacts
fade = int(0.02 * fs)
w = np.ones_like(s)
w[:fade] = np.linspace(0, 1, fade)
w[-fade:] = np.linspace(1, 0, fade)
s = s * w

# ---------- Filter bank ----------
N = 32
f_min, f_max = 100, 8000
centers = erb_spaced_center_freqs(f_min, f_max, N)

# Compute cochleagram-like envelope per band
envs = []
for f_c in centers:
    g = gammatone_ir(fs, f_c, n=4, dur_s=0.08)
    y = fftconvolve(s, g, mode="same")
    # envelope (common “ratemap-ish” step)
    env = np.abs(hilbert(y))
    # mild compression for visualization (optional)
    env = np.log1p(env)
    envs.append(env)

C = np.vstack(envs)  # shape: (bands, time)

# ---------- Plot ----------
plt.figure(figsize=(10, 4))
plt.subplot(2, 1, 1)
plt.plot(t, s, linewidth=0.8)
plt.title("Waveform")
plt.xlim([0, T])

plt.subplot(2, 1, 2)
plt.imshow(
    C,
    aspect="auto",
    origin="lower",
    extent=[0, T, centers[0], centers[-1]],
)
plt.title("Cochleagram (gammatone bank + Hilbert envelope)")
plt.xlabel("Time (s)")
plt.ylabel("Center frequency (Hz)")
plt.tight_layout()
plt.show()
