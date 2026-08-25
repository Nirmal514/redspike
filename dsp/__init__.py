"""
DSP & Feature Extraction Package for Appliance Health Monitoring.
High performance signal processing routines for acoustic and vibration signals.
"""
from .filters import butter_bandpass_filter, butter_highpass_filter, extract_envelope, detrend_signal
from .feature_extractor import DSPFeatureExtractor

__all__ = [
    "butter_bandpass_filter",
    "butter_highpass_filter",
    "extract_envelope",
    "detrend_signal",
    "DSPFeatureExtractor",
]
