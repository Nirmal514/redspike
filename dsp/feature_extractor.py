import numpy as np
from typing import Dict, Any, Tuple
from .filters import detrend_signal

class DSPFeatureExtractor:
    """
    Production-grade, pure-NumPy DSP Feature Extractor.
    Extracts 2D Mel-Spectrograms from audio and statistical/spectral feature vectors from 3-axis vibration data.
    """
    def __init__(
        self,
        audio_sr: int = 16000,
        vib_sr: int = 500,
        n_mels: int = 128,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mfcc: int = 13,
        fmin: float = 20.0,
        fmax: float = 8000.0,
        target_time_frames: int = 16
    ):
        self.audio_sr = audio_sr
        self.vib_sr = vib_sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mfcc = n_mfcc
        self.fmin = fmin
        self.fmax = min(fmax, audio_sr / 2.0)
        self.target_time_frames = target_time_frames

        # Precompute Mel Filterbank Matrix
        self.mel_basis = self._create_mel_filterbank(
            sr=self.audio_sr,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )
        self.window = np.hanning(self.n_fft).astype(np.float32)

    @staticmethod
    def _hz_to_mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    @staticmethod
    def _mel_to_hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _create_mel_filterbank(self, sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float) -> np.ndarray:
        """
        Creates triangular Mel filterbank matrix of shape (n_mels, 1 + n_fft // 2).
        """
        min_mel = self._hz_to_mel(fmin)
        max_mel = self._hz_to_mel(fmax)
        mel_points = np.linspace(min_mel, max_mel, n_mels + 2)
        hz_points = self._mel_to_hz(mel_points)
        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

        n_bins = 1 + n_fft // 2
        filterbank = np.zeros((n_mels, n_bins), dtype=np.float32)

        for i in range(1, n_mels + 1):
            left = bin_points[i - 1]
            center = bin_points[i]
            right = bin_points[i + 1]

            if center > left:
                for k in range(left, center):
                    if k < n_bins:
                        filterbank[i - 1, k] = (k - left) / (center - left)
            if right > center:
                for k in range(center, right):
                    if k < n_bins:
                        filterbank[i - 1, k] = (right - k) / (right - center)

        enorm = 2.0 / (hz_points[2:n_mels + 2] - hz_points[:n_mels])
        filterbank *= enorm[:, np.newaxis]
        return filterbank

    def extract_mel_spectrogram(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Extracts log-Mel spectrogram from 1D audio array.
        Output shape: (n_mels, target_time_frames) -> (128, 16)
        """
        if len(audio_chunk) < self.n_fft:
            audio_chunk = np.pad(audio_chunk, (0, self.n_fft - len(audio_chunk)))

        num_frames = max(1, (len(audio_chunk) - self.n_fft) // self.hop_length + 1)
        stft_matrix = np.zeros((1 + self.n_fft // 2, num_frames), dtype=np.float32)

        for i in range(num_frames):
            start = i * self.hop_length
            end = start + self.n_fft
            if end <= len(audio_chunk):
                frame = audio_chunk[start:end] * self.window
                mag = np.abs(np.fft.rfft(frame))
                stft_matrix[:, i] = mag

        mel_spec = np.dot(self.mel_basis, stft_matrix)
        log_mel_spec = np.log10(np.maximum(mel_spec, 1e-6))

        if log_mel_spec.shape[1] < self.target_time_frames:
            pad_width = self.target_time_frames - log_mel_spec.shape[1]
            log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, pad_width)), mode='edge')
        elif log_mel_spec.shape[1] > self.target_time_frames:
            log_mel_spec = log_mel_spec[:, :self.target_time_frames]

        log_mel_spec = (log_mel_spec + 3.0) / 3.0
        return log_mel_spec.astype(np.float32)

    def extract_audio_scalars(self, audio_chunk: np.ndarray) -> Dict[str, float]:
        """
        Extracts scalar audio metrics: RMS, ZCR, Spectral Centroid, Rolloff.
        """
        audio_detrended = detrend_signal(audio_chunk)
        rms = float(np.sqrt(np.mean(audio_detrended ** 2) + 1e-8))
        peak = float(np.max(np.abs(audio_detrended)) + 1e-8)
        crest_factor = float(peak / rms)

        zero_crossings = np.sum(np.diff(np.sign(audio_detrended)) != 0)
        zcr = float(zero_crossings / len(audio_detrended))

        n = len(audio_chunk)
        fft_vals = np.abs(np.fft.rfft(audio_chunk * np.hanning(n)))
        freqs = np.fft.rfftfreq(n, 1.0 / self.audio_sr)

        total_energy = np.sum(fft_vals) + 1e-8
        spectral_centroid = float(np.sum(freqs * fft_vals) / total_energy)

        cum_energy = np.cumsum(fft_vals)
        rolloff_idx = np.where(cum_energy >= 0.85 * total_energy)[0]
        spectral_rolloff = float(freqs[rolloff_idx[0]]) if len(rolloff_idx) > 0 else 0.0

        return {
            "rms": rms,
            "crest_factor": crest_factor,
            "zcr": zcr,
            "spectral_centroid": spectral_centroid,
            "spectral_rolloff": spectral_rolloff
        }

    def extract_vibration_features(self, vib_chunk: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Extracts comprehensive time & frequency domain statistical features for 3-axis vibration.
        """
        if vib_chunk.ndim == 1:
            vib_chunk = vib_chunk.reshape(1, -1)

        num_axes = vib_chunk.shape[0]
        axis_features = []
        metrics_dict = {}
        axis_names = ["x", "y", "z"][:num_axes]

        for idx, axis_name in enumerate(axis_names):
            data = detrend_signal(vib_chunk[idx])
            n = len(data)
            mean = np.mean(data)
            variance = np.var(data) + 1e-8
            std = np.sqrt(variance)
            rms = np.sqrt(np.mean(data ** 2) + 1e-8)
            peak = np.max(np.abs(data)) + 1e-8
            peak_to_peak = float(np.ptp(data))
            crest_factor = peak / rms

            skewness = np.mean(((data - mean) / std) ** 3)
            kurtosis = np.mean(((data - mean) / std) ** 4)

            fft_mag = np.abs(np.fft.rfft(data * np.hanning(n)))
            freqs = np.fft.rfftfreq(n, 1.0 / self.vib_sr)

            dom_idx = np.argmax(fft_mag)
            dom_freq = freqs[dom_idx]

            band1_energy = np.sum(fft_mag[(freqs >= 0) & (freqs < 30)]) / (n + 1e-8)
            band2_energy = np.sum(fft_mag[(freqs >= 30) & (freqs < 100)]) / (n + 1e-8)
            band3_energy = np.sum(fft_mag[(freqs >= 100) & (freqs <= 250)]) / (n + 1e-8)

            vec = [
                rms,
                variance,
                peak_to_peak,
                crest_factor,
                kurtosis,
                skewness,
                dom_freq,
                band1_energy,
                band2_energy,
                band3_energy,
                mean,
                peak
            ]
            axis_features.extend(vec)

            metrics_dict[axis_name] = {
                "rms": float(rms),
                "peak_to_peak": float(peak_to_peak),
                "crest_factor": float(crest_factor),
                "kurtosis": float(kurtosis),
                "skewness": float(skewness),
                "dom_freq_hz": float(dom_freq),
                "band_low_energy": float(band1_energy),
                "band_mid_energy": float(band2_energy),
                "band_high_energy": float(band3_energy)
            }

        feature_vector = np.array(axis_features, dtype=np.float32)
        return feature_vector, metrics_dict

    def process_frame(
        self, audio_chunk: np.ndarray, vib_chunk: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        mel_spec = self.extract_mel_spectrogram(audio_chunk)
        vib_vec, vib_metrics = self.extract_vibration_features(vib_chunk)
        audio_scalars = self.extract_audio_scalars(audio_chunk)

        telemetry = {
            "audio": audio_scalars,
            "vibration": vib_metrics
        }
        return mel_spec, vib_vec, telemetry
