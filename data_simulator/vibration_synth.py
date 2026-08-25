import numpy as np
from typing import Optional, Dict, Tuple
from .fault_injector import FaultInjector, FaultType

class VibrationSynthesizer:
    """
    Synthesizes 3-Axis Accelerometer (X, Y, Z) vibration telemetry in m/s^2 or g-units.
    Simulates operational rotational dynamics (1X, 2X, harmonics) and physical fault modulations.
    """
    def __init__(self, sample_rate: int = 500, appliance_type: str = "refrigerator"):
        self.sample_rate = sample_rate
        self.appliance_type = appliance_type
        self._time_offset = 0.0

        if appliance_type == "refrigerator":
            self.base_rpm = 1800.0
            self.base_vib_amp = 0.05  # g RMS
        elif appliance_type == "washing_machine":
            self.base_rpm = 800.0
            self.base_vib_amp = 0.12
        elif appliance_type == "water_pump":
            self.base_rpm = 2900.0
            self.base_vib_amp = 0.15
        else:
            self.base_rpm = 1500.0
            self.base_vib_amp = 0.08

    def generate_chunk(self, num_samples: int, fault_injector: Optional[FaultInjector] = None) -> np.ndarray:
        """
        Generates a 2D array of shape [3, num_samples] corresponding to [X, Y, Z] vibration acceleration.
        """
        duration = num_samples / self.sample_rate
        t = np.linspace(self._time_offset, self._time_offset + duration, num_samples, endpoint=False)
        self._time_offset = (self._time_offset + duration) % 10000.0

        # Operational RPM dynamics (e.g. slight spin cycle variation)
        if self.appliance_type == "washing_machine":
            # Slow wobble modulation + cycle acceleration
            rpm_mod = 200.0 * np.sin(2 * np.pi * 0.05 * t)
            cur_rpm = self.base_rpm + rpm_mod
        else:
            cur_rpm = self.base_rpm + 10.0 * np.sin(2 * np.pi * 0.1 * t)

        rot_freq = cur_rpm / 60.0  # Hz

        # Baseline Mechanical Vibration
        # X: Radial horizontal
        # Y: Radial vertical (90 deg phase shifted from X)
        # Z: Axial
        x_vib = self.base_vib_amp * np.sin(2 * np.pi * rot_freq * t) + 0.3 * self.base_vib_amp * np.sin(4 * np.pi * rot_freq * t)
        y_vib = self.base_vib_amp * np.cos(2 * np.pi * rot_freq * t) + 0.3 * self.base_vib_amp * np.cos(4 * np.pi * rot_freq * t)
        z_vib = 0.4 * self.base_vib_amp * np.sin(2 * np.pi * (rot_freq * 2) * t)

        # Add Gaussian sensor noise
        x_vib += np.random.normal(0, 0.01, num_samples)
        y_vib += np.random.normal(0, 0.01, num_samples)
        z_vib += np.random.normal(0, 0.01, num_samples)

        # Inject Faults
        if fault_injector:
            state = fault_injector.get_state()
            fault = FaultType(state["fault"])
            intensity = state["intensity"]

            if fault != FaultType.NORMAL and intensity > 0.0:
                fx, fy, fz = self._synthesize_fault_vibration(t, num_samples, fault, intensity, rot_freq)
                x_vib += fx
                y_vib += fy
                z_vib += fz

        # Shape: (3, num_samples)
        return np.vstack([x_vib, y_vib, z_vib]).astype(np.float32)

    def _synthesize_fault_vibration(
        self, t: np.ndarray, num_samples: int, fault: FaultType, intensity: float, rot_freq: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        fx = np.zeros(num_samples, dtype=np.float32)
        fy = np.zeros(num_samples, dtype=np.float32)
        fz = np.zeros(num_samples, dtype=np.float32)

        if fault == FaultType.UNBALANCE_TUB:
            # Huge 1X RPM radial amplitude spike on X and Y with 90° quadrature phase
            unbalance_amp = 1.8 * intensity
            fx = unbalance_amp * np.sin(2 * np.pi * rot_freq * t)
            fy = unbalance_amp * np.cos(2 * np.pi * rot_freq * t)
            fz = 0.3 * unbalance_amp * np.sin(2 * np.pi * rot_freq * t)

        elif fault in (FaultType.COMPRESSOR_BEARING_WEAR, FaultType.MOTOR_BEARING_WEAR):
            # High-kurtosis repetitive shock pulses at BPFO (Ball Pass Frequency Outer ~ 3.58 * rot_freq)
            bpfo = rot_freq * 3.58
            pulse_train = np.exp(-np.mod(t * bpfo, 1.0) * 12.0)
            fx = pulse_train * (0.8 * intensity) * np.sin(2 * np.pi * 180.0 * t)
            fy = pulse_train * (0.7 * intensity) * np.cos(2 * np.pi * 180.0 * t)
            fz = pulse_train * (1.1 * intensity)  # High axial impact

        elif fault in (FaultType.IMPELLER_CAVITATION, FaultType.DRAIN_CAVITATION):
            # High frequency broadband vibration floor rise and random turbulence spikes
            turb_x = np.random.normal(0, 0.4 * intensity, num_samples)
            turb_y = np.random.normal(0, 0.4 * intensity, num_samples)
            turb_z = np.random.normal(0, 0.6 * intensity, num_samples)
            fx = turb_x.astype(np.float32)
            fy = turb_y.astype(np.float32)
            fz = turb_z.astype(np.float32)

        elif fault == FaultType.FAN_BLADE_FRICTION:
            # Friction chatter in radial directions
            chatter_freq = rot_freq * 4.0
            fx = (0.5 * intensity) * np.sin(2 * np.pi * chatter_freq * t) + np.random.normal(0, 0.15 * intensity, num_samples)
            fy = (0.5 * intensity) * np.cos(2 * np.pi * chatter_freq * t) + np.random.normal(0, 0.15 * intensity, num_samples)
            fz = 0.1 * fx

        elif fault == FaultType.BELT_SLIP:
            # Periodic slip-stick vibrations
            slip_freq = rot_freq * 0.8
            slip_env = (np.sin(2 * np.pi * slip_freq * t) > 0.5).astype(np.float32)
            fx = slip_env * (0.6 * intensity) * np.sin(2 * np.pi * 95.0 * t)
            fy = slip_env * (0.4 * intensity) * np.sin(2 * np.pi * 95.0 * t)

        elif fault == FaultType.DRY_RUNNING:
            # High erratic jitter across all 3 axes due to lack of liquid damping
            jitter = np.random.normal(0, 0.45 * intensity, (3, num_samples)).astype(np.float32)
            fx, fy, fz = jitter[0], jitter[1], jitter[2]

        elif fault in (FaultType.PIPE_WATER_HAMMER, FaultType.RELAY_CHATTER):
            # Giant shock peak along axial (Z) and radial (X/Y)
            period = 0.8
            phase = np.mod(t, period)
            shock = np.exp(-phase * 25.0) * (phase < 0.15) * (2.8 * intensity)
            fx = shock * np.sin(2 * np.pi * 45.0 * t)
            fy = shock * 0.8 * np.cos(2 * np.pi * 45.0 * t)
            fz = shock * 1.5

        return fx.astype(np.float32), fy.astype(np.float32), fz.astype(np.float32)
