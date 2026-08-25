import numpy as np
from typing import Optional, Tuple
from .fault_injector import FaultInjector, FaultType
from dsp.filters import lfilter_pure

def simple_medfilt_1d(x: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Fast 1D moving median filter in pure numpy."""
    k = kernel_size
    pad = k // 2
    padded = np.pad(x, pad, mode='edge')
    out = np.zeros_like(x)
    for i in range(len(x)):
        out[i] = np.median(padded[i:i + k])
    return out

class AudioSynthesizer:
    """
    Synthesizes continuous acoustic audio signals at target sample rates (e.g. 16kHz or 44.1kHz).
    Simulates baseline motor acoustics and fault injection signatures with high physical fidelity.
    """
    def __init__(self, sample_rate: int = 16000, appliance_type: str = "refrigerator"):
        self.sample_rate = sample_rate
        self.appliance_type = appliance_type
        self._time_offset = 0.0

        # Appliance physical characteristics
        if appliance_type == "refrigerator":
            self.base_freq = 60.0  # 60 Hz electrical grid fundamental
            self.rpm = 1800.0
            self.noise_floor = 0.015
            self.harmonics = [(1, 0.4), (2, 0.25), (3, 0.15), (4, 0.08), (6, 0.05), (10, 0.02)]
        elif appliance_type == "washing_machine":
            self.base_freq = 50.0
            self.rpm = 800.0  # Variable spin cycle
            self.noise_floor = 0.03
            self.harmonics = [(1, 0.5), (2, 0.35), (3, 0.2), (4, 0.1), (8, 0.08)]
        elif appliance_type == "water_pump":
            self.base_freq = 50.0
            self.rpm = 2900.0
            self.noise_floor = 0.035
            self.harmonics = [(1, 0.6), (2, 0.4), (3, 0.25), (4, 0.18), (5, 0.12), (12, 0.05)]
        else:
            self.base_freq = 50.0
            self.rpm = 1500.0
            self.noise_floor = 0.02
            self.harmonics = [(1, 0.5), (2, 0.3), (3, 0.15)]

    def generate_chunk(self, num_samples: int, fault_injector: Optional[FaultInjector] = None) -> np.ndarray:
        """
        Generates a 1D audio sample chunk of length `num_samples`.
        Updates internal time offset to guarantee smooth inter-frame phase continuity.
        """
        duration = num_samples / self.sample_rate
        t = np.linspace(self._time_offset, self._time_offset + duration, num_samples, endpoint=False)
        self._time_offset = (self._time_offset + duration) % 10000.0

        # 1. Base acoustic signal (Harmonic Motor Hum + Low frequency hum)
        motor_rot_freq = self.rpm / 60.0  # e.g., 30 Hz for 1800 RPM
        audio = np.zeros(num_samples, dtype=np.float32)

        # Fundamental and electrical harmonics
        for harm_idx, amp in self.harmonics:
            freq = self.base_freq * harm_idx
            jitter = 0.02 * np.sin(2 * np.pi * 0.3 * t)
            audio += amp * np.sin(2 * np.pi * freq * t + jitter).astype(np.float32)

        # Rotor rotation mechanical harmonics (blade pass / pole pass)
        audio += 0.2 * np.sin(2 * np.pi * motor_rot_freq * t).astype(np.float32)
        audio += 0.15 * np.sin(2 * np.pi * (motor_rot_freq * 2) * t).astype(np.float32)

        # Add background broadband Gaussian & pink noise
        white_noise = np.random.normal(0, self.noise_floor, num_samples).astype(np.float32)
        # Simple pink filter approximation (1st order IIR filter)
        b, a = [1.0], [1.0, -0.95]
        pink_noise = lfilter_pure(b, a, white_noise * 0.5).astype(np.float32)
        audio += pink_noise

        # 2. Dynamic Fault Injection
        if fault_injector:
            state = fault_injector.get_state()
            fault = FaultType(state["fault"])
            intensity = state["intensity"]

            if fault != FaultType.NORMAL and intensity > 0.0:
                fault_audio = self._synthesize_fault_signature(t, num_samples, fault, intensity, motor_rot_freq)
                audio += fault_audio

        # Normalize / Soft clip to prevent numerical clipping beyond [-1.0, 1.0]
        audio = np.tanh(audio * 0.8)
        return audio.astype(np.float32)

    def _synthesize_fault_signature(
        self, t: np.ndarray, num_samples: int, fault: FaultType, intensity: float, motor_rot_freq: float
    ) -> np.ndarray:
        fault_sig = np.zeros(num_samples, dtype=np.float32)

        if fault in (FaultType.COMPRESSOR_BEARING_WEAR, FaultType.MOTOR_BEARING_WEAR):
            bpfo = motor_rot_freq * 3.58
            carrier_freq = 3200.0
            impulse_phase = np.mod(t * bpfo, 1.0)
            impulse_envelope = np.exp(-impulse_phase * 15.0)
            carrier = np.sin(2 * np.pi * carrier_freq * t)
            random_shudder = np.random.normal(0, 0.2, num_samples)
            fault_sig = (impulse_envelope * carrier + random_shudder * impulse_envelope) * (1.2 * intensity)

        elif fault == FaultType.FAN_BLADE_FRICTION:
            blade_pass = motor_rot_freq * 4.0
            rub_envelope = (np.sin(2 * np.pi * blade_pass * t) + 1.0) * 0.5
            high_scrape = np.random.normal(0, 0.6, num_samples)
            high_scrape = simple_medfilt_1d(high_scrape, kernel_size=3)
            squeal = np.sin(2 * np.pi * 4200.0 * t + np.sin(2 * np.pi * 5.0 * t))
            fault_sig = (rub_envelope * high_scrape * 0.8 + squeal * 0.4) * (1.1 * intensity)

        elif fault == FaultType.RELAY_CHATTER:
            burst_mask = (np.sin(2 * np.pi * 120.0 * t) > 0.85).astype(np.float32)
            arc_crackles = np.random.normal(0, 1.0, num_samples) * (np.random.rand(num_samples) > 0.92)
            fault_sig = burst_mask * arc_crackles * (2.0 * intensity)

        elif fault == FaultType.UNBALANCE_TUB:
            thump_envelope = np.maximum(0, np.sin(2 * np.pi * motor_rot_freq * t)) ** 4
            low_thump = np.sin(2 * np.pi * (motor_rot_freq * 2) * t) * thump_envelope
            rumble = np.random.normal(0, 0.4, num_samples) * thump_envelope
            fault_sig = (low_thump * 1.5 + rumble) * (1.4 * intensity)

        elif fault == FaultType.BELT_SLIP:
            mod_freq = 2400.0 + 600.0 * np.sin(2 * np.pi * (motor_rot_freq * 0.7) * t)
            squeal = np.sin(2 * np.pi * mod_freq * t)
            hiss = np.random.normal(0, 0.3, num_samples)
            fault_sig = (squeal * 0.7 + hiss * 0.3) * (1.3 * intensity)

        elif fault in (FaultType.DRAIN_CAVITATION, FaultType.IMPELLER_CAVITATION):
            bubble_rate = 0.08 * intensity
            collapses = (np.random.rand(num_samples) < bubble_rate).astype(np.float32)
            ringing = np.sin(2 * np.pi * 5800.0 * t) * np.random.normal(1.0, 0.3, num_samples)
            broadband_hiss = np.random.normal(0, 0.5, num_samples)
            fault_sig = (collapses * ringing * 2.2 + collapses * broadband_hiss * 1.5) * intensity

        elif fault == FaultType.DRY_RUNNING:
            dry_whine = (
                np.sin(2 * np.pi * (motor_rot_freq * 8) * t) * 0.5 +
                np.sin(2 * np.pi * 2100.0 * t) * 0.4 +
                np.random.normal(0, 0.35, num_samples)
            )
            fault_sig = dry_whine * (1.2 * intensity)

        elif fault == FaultType.PIPE_WATER_HAMMER:
            shock_period = 0.8
            phase_in_period = np.mod(t, shock_period)
            shock_envelope = np.exp(-phase_in_period * 20.0) * (phase_in_period < 0.2)
            ringing = np.sin(2 * np.pi * 180.0 * t)
            fault_sig = shock_envelope * ringing * (2.5 * intensity)

        return fault_sig.astype(np.float32)
