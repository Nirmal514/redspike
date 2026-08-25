"""
Data Simulator Package for Appliance Health Monitoring.
Simulates realistic acoustic and 3-axis vibration telemetry with dynamic fault injection.
"""
from .audio_synth import AudioSynthesizer
from .vibration_synth import VibrationSynthesizer
from .fault_injector import FaultInjector, FaultType

__all__ = ["AudioSynthesizer", "VibrationSynthesizer", "FaultInjector", "FaultType"]
