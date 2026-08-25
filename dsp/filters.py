import numpy as np
from typing import Tuple

def lfilter_pure(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Pure NumPy implementation of Direct Form IIR 1D Digital Filter.
    Supports任意 order transfer function H(z) = B(z) / A(z).
    """
    b = np.asarray(b, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    if a[0] != 1.0:
        b = b / a[0]
        a = a / a[0]

    y = np.zeros_like(x, dtype=np.float64)
    nb = len(b)
    na = len(a)

    for n in range(len(x)):
        acc = 0.0
        for i in range(nb):
            if n - i >= 0:
                acc += b[i] * x[n - i]
        for j in range(1, na):
            if n - j >= 0:
                acc -= a[j] * y[n - j]
        y[n] = acc
    return y.astype(np.float32)

def filtfilt_pure(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Pure NumPy zero-phase forward-backward digital filtering.
    """
    if x.ndim == 1:
        y_fwd = lfilter_pure(b, a, x)
        y_rev = lfilter_pure(b, a, y_fwd[::-1])
        return y_rev[::-1].astype(np.float32)
    elif x.ndim == 2:
        out = np.zeros_like(x, dtype=np.float32)
        for i in range(x.shape[0]):
            y_fwd = lfilter_pure(b, a, x[i])
            y_rev = lfilter_pure(b, a, y_fwd[::-1])
            out[i] = y_rev[::-1]
        return out
    return x.astype(np.float32)

def butter_lowpass_coeffs(cutoff: float, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    2nd-order Butterworth low-pass filter coefficients (Biquad).
    """
    w0 = 2.0 * np.pi * cutoff / fs
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2.0 * 0.7071)  # Q = 1/sqrt(2)

    b0 = (1.0 - cos_w0) / 2.0
    b1 = 1.0 - cos_w0
    b2 = (1.0 - cos_w0) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0
    return b, a

def butter_highpass_coeffs(cutoff: float, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    2nd-order Butterworth high-pass filter coefficients (Biquad).
    """
    w0 = 2.0 * np.pi * cutoff / fs
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / (2.0 * 0.7071)

    b0 = (1.0 + cos_w0) / 2.0
    b1 = -(1.0 + cos_w0)
    b2 = (1.0 + cos_w0) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0
    return b, a

def butter_bandpass_coeffs(lowcut: float, highcut: float, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bandpass filter via cascaded highpass and lowpass biquad.
    """
    b_hp, a_hp = butter_highpass_coeffs(lowcut, fs)
    b_lp, a_lp = butter_lowpass_coeffs(highcut, fs)
    # Convolve numerator and denominator
    b = np.convolve(b_hp, b_lp)
    a = np.convolve(a_hp, a_lp)
    return b, a

def butter_bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 2) -> np.ndarray:
    b, a = butter_bandpass_coeffs(lowcut, highcut, fs)
    return filtfilt_pure(b, a, data)

def butter_highpass_filter(data: np.ndarray, cutoff: float, fs: float, order: int = 2) -> np.ndarray:
    b, a = butter_highpass_coeffs(cutoff, fs)
    return filtfilt_pure(b, a, data)

def extract_envelope(signal_data: np.ndarray, fs: float, lpf_cutoff: float = 50.0) -> np.ndarray:
    """
    Extracts the demodulated amplitude envelope via FFT Hilbert Transform.
    """
    n = len(signal_data)
    fft_x = np.fft.fft(signal_data)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = 1
        h[n // 2] = 1
        h[1:n // 2] = 2
    else:
        h[0] = 1
        h[1:(n + 1) // 2] = 2

    analytic_sig = np.fft.ifft(fft_x * h)
    envelope = np.abs(analytic_sig)
    b, a = butter_lowpass_coeffs(lpf_cutoff, fs)
    return filtfilt_pure(b, a, envelope).astype(np.float32)

def detrend_signal(data: np.ndarray) -> np.ndarray:
    """
    Removes linear trend / DC bias from array using pure numpy.
    """
    if data.ndim == 1:
        n = len(data)
        x = np.arange(n)
        p = np.polyfit(x, data, deg=1)
        trend = np.polyval(p, x)
        return (data - trend).astype(np.float32)
    elif data.ndim == 2:
        out = np.zeros_like(data, dtype=np.float32)
        n = data.shape[-1]
        x = np.arange(n)
        for i in range(data.shape[0]):
            p = np.polyfit(x, data[i], deg=1)
            out[i] = data[i] - np.polyval(p, x)
        return out
    return data.astype(np.float32)
