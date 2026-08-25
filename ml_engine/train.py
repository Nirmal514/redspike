import os
import sys
import numpy as np
from typing import Tuple, List, Dict

# Ensure root is on python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_simulator.audio_synth import AudioSynthesizer
from data_simulator.vibration_synth import VibrationSynthesizer
from data_simulator.fault_injector import FaultInjector, FaultType
from dsp.feature_extractor import DSPFeatureExtractor
from ml_engine.model import DualStreamApplianceNet, TORCH_AVAILABLE

FAULT_MAPPING = {
    FaultType.NORMAL: 0,
    FaultType.COMPRESSOR_BEARING_WEAR: 1,
    FaultType.MOTOR_BEARING_WEAR: 1,
    FaultType.IMPELLER_CAVITATION: 2,
    FaultType.DRAIN_CAVITATION: 2,
    FaultType.UNBALANCE_TUB: 3,
    FaultType.FAN_BLADE_FRICTION: 4,
    FaultType.BELT_SLIP: 4,
    FaultType.DRY_RUNNING: 5,
    FaultType.RELAY_CHATTER: 6,
    FaultType.PIPE_WATER_HAMMER: 6,
}

FAULT_NAMES = [
    "normal",
    "bearing_wear",
    "cavitation",
    "unbalance",
    "friction",
    "dry_running",
    "chatter_or_hammer"
]

SEVERITY_NAMES = ["low", "medium", "critical"]

def map_severity(fault: FaultType, intensity: float) -> int:
    if fault == FaultType.NORMAL:
        return 0
    if fault in (FaultType.IMPELLER_CAVITATION, FaultType.UNBALANCE_TUB, FaultType.DRY_RUNNING, FaultType.PIPE_WATER_HAMMER):
        return 2 if intensity >= 0.5 else 1
    if fault in (FaultType.COMPRESSOR_BEARING_WEAR, FaultType.MOTOR_BEARING_WEAR):
        return 2 if intensity >= 0.75 else 1
    return 1 if intensity >= 0.4 else 0

def generate_synthetic_data(samples_per_class: int = 50):
    mel_specs = []
    vib_features = []
    anomaly_targets = []
    fault_targets = []
    severity_targets = []

    appliances = ["refrigerator", "washing_machine", "water_pump"]
    extractor = DSPFeatureExtractor()
    injector = FaultInjector()

    print(f"[Train] Generating synthetic multi-modal dataset ({samples_per_class} samples/class)...")
    all_faults = list(FaultType)

    for fault in all_faults:
        for _ in range(samples_per_class):
            appliance = np.random.choice(appliances)
            audio_synth = AudioSynthesizer(sample_rate=16000, appliance_type=appliance)
            vib_synth = VibrationSynthesizer(sample_rate=500, appliance_type=appliance)

            intensity = 0.0 if fault == FaultType.NORMAL else float(np.random.uniform(0.4, 1.0))
            injector.inject_fault(fault.value, intensity=intensity)

            audio_raw = audio_synth.generate_chunk(4096, fault_injector=injector)
            vib_raw = vib_synth.generate_chunk(128, fault_injector=injector)

            mel, vib_vec, _ = extractor.process_frame(audio_raw, vib_raw)

            fault_idx = FAULT_MAPPING[fault]
            severity_idx = map_severity(fault, intensity)
            is_anomaly = 0.0 if fault == FaultType.NORMAL else 1.0

            mel_specs.append(mel)
            vib_features.append(vib_vec)
            anomaly_targets.append(is_anomaly)
            fault_targets.append(fault_idx)
            severity_targets.append(severity_idx)

    return (
        np.array(mel_specs, dtype=np.float32),
        np.array(vib_features, dtype=np.float32),
        np.array(anomaly_targets, dtype=np.float32),
        np.array(fault_targets, dtype=np.int64),
        np.array(severity_targets, dtype=np.int64)
    )

def train_numpy(mels, vibs, anom_t, fault_t, sev_t, epochs: int = 30, lr: float = 0.02):
    """
    Trains DualStreamApplianceNet using pure NumPy gradient descent.
    """
    model = DualStreamApplianceNet(num_fault_classes=len(FAULT_NAMES), num_severity_classes=len(SEVERITY_NAMES))
    n_samples = len(mels)
    print(f"[Train NumPy] Training model on {n_samples} multi-modal samples...")

    # One-hot encode targets
    y_fault = np.zeros((n_samples, len(FAULT_NAMES)), dtype=np.float32)
    y_fault[np.arange(n_samples), fault_t] = 1.0

    y_sev = np.zeros((n_samples, len(SEVERITY_NAMES)), dtype=np.float32)
    y_sev[np.arange(n_samples), sev_t] = 1.0

    mel_flat = np.mean(mels, axis=2) # (N, 128)

    for epoch in range(1, epochs + 1):
        # Forward pass
        h_audio = np.maximum(0, np.dot(mel_flat, model.w_mel) + model.b_mel)
        h_vib = np.maximum(0, np.dot(vibs, model.w_vib) + model.b_vib)
        fused = np.concatenate([h_audio, h_vib], axis=-1)
        h_fusion = np.maximum(0, np.dot(fused, model.w_fusion) + model.b_fusion)

        # Anomaly prediction
        anom_logit = np.dot(h_fusion, model.w_anom) + model.b_anom
        anom_pred = 1.0 / (1.0 + np.exp(-np.clip(anom_logit, -15, 15)))

        # Fault softmax
        fault_logits = np.dot(h_fusion, model.w_fault) + model.b_fault
        exp_f = np.exp(fault_logits - np.max(fault_logits, axis=-1, keepdims=True))
        fault_probs = exp_f / np.sum(exp_f, axis=-1, keepdims=True)

        # Severity softmax
        sev_logits = np.dot(h_fusion, model.w_sev) + model.b_sev
        exp_s = np.exp(sev_logits - np.max(sev_logits, axis=-1, keepdims=True))
        sev_probs = exp_s / np.sum(exp_s, axis=-1, keepdims=True)

        # Gradients
        d_anom = (anom_pred - anom_t[:, np.newaxis]) / n_samples
        d_fault = (fault_probs - y_fault) / n_samples
        d_sev = (sev_probs - y_sev) / n_samples

        # Weight updates
        model.w_anom -= lr * np.dot(h_fusion.T, d_anom)
        model.b_anom -= lr * np.sum(d_anom, axis=0)

        model.w_fault -= lr * np.dot(h_fusion.T, d_fault)
        model.b_fault -= lr * np.sum(d_fault, axis=0)

        model.w_sev -= lr * np.dot(h_fusion.T, d_sev)
        model.b_sev -= lr * np.sum(d_sev, axis=0)

        # Backprop through fusion
        d_fusion = (
            np.dot(d_anom, model.w_anom.T) +
            np.dot(d_fault, model.w_fault.T) +
            np.dot(d_sev, model.w_sev.T)
        ) * (h_fusion > 0)

        model.w_fusion -= lr * np.dot(fused.T, d_fusion)
        model.b_fusion -= lr * np.sum(d_fusion, axis=0)

        # Accuracy
        preds_f = np.argmax(fault_probs, axis=-1)
        acc = (np.mean(preds_f == fault_t)) * 100.0
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} - Classification Accuracy: {acc:.1f}%")

    save_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(save_dir, exist_ok=True)
    npz_path = os.path.join(save_dir, "appliance_classifier.npz")
    model.save_weights(npz_path)
    print(f"[Train] Pure NumPy weights saved to {npz_path}")
    return model

def train_and_export():
    mels, vibs, anom_t, fault_t, sev_t = generate_synthetic_data(samples_per_class=40)

    if TORCH_AVAILABLE:
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import TensorDataset, DataLoader

            t_mel = torch.tensor(mels).unsqueeze(1)
            t_vib = torch.tensor(vibs)
            t_anom = torch.tensor(anom_t).unsqueeze(-1)
            t_fault = torch.tensor(fault_t)
            t_sev = torch.tensor(sev_t)

            ds = TensorDataset(t_mel, t_vib, t_anom, t_fault, t_sev)
            loader = DataLoader(ds, batch_size=32, shuffle=True)

            model = DualStreamApplianceNet(num_fault_classes=len(FAULT_NAMES), num_severity_classes=len(SEVERITY_NAMES))
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            crit_a = nn.BCELoss()
            crit_f = nn.CrossEntropyLoss()
            crit_s = nn.CrossEntropyLoss()

            print("[Train PyTorch] Starting PyTorch training loop...")
            model.train()
            for epoch in range(1, 11):
                for mel_b, vib_b, a_b, f_b, s_b in loader:
                    opt.zero_grad()
                    oa, of, os_ = model(mel_b, vib_b)
                    loss = crit_a(oa, a_b) + 1.2 * crit_f(of, f_b) + 0.8 * crit_s(os_, s_b)
                    loss.backward()
                    opt.step()

            save_dir = os.path.join(os.path.dirname(__file__), "models")
            os.makedirs(save_dir, exist_ok=True)
            pt_path = os.path.join(save_dir, "appliance_classifier.pt")
            torch.save(model.state_dict(), pt_path)
            print(f"[Train PyTorch] Saved {pt_path}")
            return model
        except Exception as e:
            print(f"[Train] PyTorch train error ({e}), running NumPy engine.")

    return train_numpy(mels, vibs, anom_t, fault_t, sev_t)

if __name__ == "__main__":
    train_and_export()
