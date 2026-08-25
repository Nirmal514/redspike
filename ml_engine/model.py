import os
import numpy as np
from typing import Dict, Tuple, Optional, Any, Union

# Conditional PyTorch Import
TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except (ImportError, Exception):
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class AcousticStream2D(nn.Module):
        """
        Lightweight 2D-CNN for processing 128-band Mel Spectrograms (Shape: [B, 1, 128, T]).
        """
        def __init__(self, in_channels: int = 1, latent_dim: int = 64):
            super().__init__()
            self.conv_block = nn.Sequential(
                nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(4, 16),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(4, 32),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.GroupNorm(4, 64),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((2, 2))
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 2 * 2, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feat = self.conv_block(x)
            return self.fc(feat)

    class VibrationStream1D(nn.Module):
        """
        1D Dense network for processing statistical feature vectors from 3-axis accelerometer (Shape: [B, 36]).
        """
        def __init__(self, in_features: int = 36, latent_dim: int = 32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Linear(64, latent_dim),
                nn.LayerNorm(latent_dim),
                nn.ReLU(),
                nn.Dropout(0.15)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class DualStreamApplianceNet(nn.Module):
        """
        Multi-modal Dual-Stream Fusion Architecture in PyTorch.
        Accepts both torch.Tensor and numpy.ndarray automatically.
        """
        def __init__(
            self,
            num_fault_classes: int = 7,
            num_severity_classes: int = 3,
            audio_latent_dim: int = 64,
            vib_latent_dim: int = 32,
            fusion_dim: int = 64
        ):
            super().__init__()
            self.acoustic_stream = AcousticStream2D(in_channels=1, latent_dim=audio_latent_dim)
            self.vibration_stream = VibrationStream1D(in_features=36, latent_dim=vib_latent_dim)

            self.fusion_fc = nn.Sequential(
                nn.Linear(audio_latent_dim + vib_latent_dim, fusion_dim),
                nn.LayerNorm(fusion_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            )

            self.head_anomaly = nn.Sequential(
                nn.Linear(fusion_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()
            )

            self.head_fault_type = nn.Sequential(
                nn.Linear(fusion_dim, 32),
                nn.ReLU(),
                nn.Linear(32, num_fault_classes)
            )

            self.head_severity = nn.Sequential(
                nn.Linear(fusion_dim, 16),
                nn.ReLU(),
                nn.Linear(16, num_severity_classes)
            )

        def forward(
            self,
            mel_spec: Union[torch.Tensor, np.ndarray],
            vib_features: Union[torch.Tensor, np.ndarray]
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # Convert NumPy inputs to Torch tensors
            if isinstance(mel_spec, np.ndarray):
                mel_spec = torch.from_numpy(mel_spec.astype(np.float32))
            if isinstance(vib_features, np.ndarray):
                vib_features = torch.from_numpy(vib_features.astype(np.float32))

            # Ensure proper shape [B, 1, 128, T]
            if mel_spec.dim() == 2:
                mel_spec = mel_spec.unsqueeze(0).unsqueeze(0)
            elif mel_spec.dim() == 3:
                mel_spec = mel_spec.unsqueeze(1)

            # Ensure proper shape [B, 36]
            if vib_features.dim() == 1:
                vib_features = vib_features.unsqueeze(0)

            audio_feat = self.acoustic_stream(mel_spec)
            vib_feat = self.vibration_stream(vib_features)
            fused = torch.cat([audio_feat, vib_feat], dim=-1)
            fused_latent = self.fusion_fc(fused)

            anomaly_score = self.head_anomaly(fused_latent)
            fault_logits = self.head_fault_type(fused_latent)
            severity_logits = self.head_severity(fused_latent)
            return anomaly_score, fault_logits, severity_logits

        def export_to_onnx(self, file_path: str, time_frames: int = 16):
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            self.eval()
            dummy_mel = torch.randn(1, 1, 128, time_frames, dtype=torch.float32)
            dummy_vib = torch.randn(1, 36, dtype=torch.float32)
            torch.onnx.export(
                self,
                (dummy_mel, dummy_vib),
                file_path,
                input_names=["mel_spectrogram", "vibration_features"],
                output_names=["anomaly_score", "fault_logits", "severity_logits"],
                dynamic_axes={
                    "mel_spectrogram": {0: "batch_size"},
                    "vibration_features": {0: "batch_size"},
                    "anomaly_score": {0: "batch_size"},
                    "fault_logits": {0: "batch_size"},
                    "severity_logits": {0: "batch_size"},
                },
                opset_version=14
            )

else:
    class DualStreamApplianceNet:
        def __init__(self, num_fault_classes: int = 7, num_severity_classes: int = 3):
            self.num_fault_classes = num_fault_classes
            self.num_severity_classes = num_severity_classes
            
            np.random.seed(42)
            self.w_mel = np.random.randn(128, 64).astype(np.float32) * np.sqrt(2.0 / 128)
            self.b_mel = np.zeros(64, dtype=np.float32)

            self.w_vib = np.random.randn(36, 32).astype(np.float32) * np.sqrt(2.0 / 36)
            self.b_vib = np.zeros(32, dtype=np.float32)

            self.w_fusion = np.random.randn(96, 64).astype(np.float32) * np.sqrt(2.0 / 96)
            self.b_fusion = np.zeros(64, dtype=np.float32)

            self.w_anom = np.random.randn(64, 1).astype(np.float32) * np.sqrt(2.0 / 64)
            self.b_anom = np.zeros(1, dtype=np.float32)

            self.w_fault = np.random.randn(64, num_fault_classes).astype(np.float32) * np.sqrt(2.0 / 64)
            self.b_fault = np.zeros(num_fault_classes, dtype=np.float32)

            self.w_sev = np.random.randn(64, num_severity_classes).astype(np.float32) * np.sqrt(2.0 / 64)
            self.b_sev = np.zeros(num_severity_classes, dtype=np.float32)

        def __call__(self, mel_spec: np.ndarray, vib_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            return self.forward(mel_spec, vib_features)

        def eval(self):
            pass

        def forward(self, mel_spec: np.ndarray, vib_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            if mel_spec.ndim == 4:
                mel_flat = np.mean(mel_spec, axis=(1, 3))
            elif mel_spec.ndim == 3:
                mel_flat = np.mean(mel_spec, axis=2)
            elif mel_spec.ndim == 2:
                mel_flat = np.mean(mel_spec, axis=1, keepdims=True).T
            else:
                mel_flat = mel_spec

            if vib_features.ndim == 1:
                vib_input = vib_features[np.newaxis, :]
            else:
                vib_input = vib_features

            h_audio = np.maximum(0, np.dot(mel_flat, self.w_mel) + self.b_mel)
            h_vib = np.maximum(0, np.dot(vib_input, self.w_vib) + self.b_vib)
            fused = np.concatenate([h_audio, h_vib], axis=-1)
            h_fusion = np.maximum(0, np.dot(fused, self.w_fusion) + self.b_fusion)

            anom_logit = np.dot(h_fusion, self.w_anom) + self.b_anom
            anom_score = 1.0 / (1.0 + np.exp(-np.clip(anom_logit, -15, 15)))

            fault_logits = np.dot(h_fusion, self.w_fault) + self.b_fault
            sev_logits = np.dot(h_fusion, self.w_sev) + self.b_sev
            return anom_score, fault_logits, sev_logits

        def save_weights(self, path: str):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            np.savez(
                path,
                w_mel=self.w_mel, b_mel=self.b_mel,
                w_vib=self.w_vib, b_vib=self.b_vib,
                w_fusion=self.w_fusion, b_fusion=self.b_fusion,
                w_anom=self.w_anom, b_anom=self.b_anom,
                w_fault=self.w_fault, b_fault=self.b_fault,
                w_sev=self.w_sev, b_sev=self.b_sev
            )

        def load_weights(self, path: str):
            if os.path.exists(path):
                data = np.load(path)
                self.w_mel = data["w_mel"]
                self.b_mel = data["b_mel"]
                self.w_vib = data["w_vib"]
                self.b_vib = data["b_vib"]
                self.w_fusion = data["w_fusion"]
                self.b_fusion = data["b_fusion"]
                self.w_anom = data["w_anom"]
                self.b_anom = data["b_anom"]
                self.w_fault = data["w_fault"]
                self.b_fault = data["b_fault"]
                self.w_sev = data["w_sev"]
                self.b_sev = data["b_sev"]
