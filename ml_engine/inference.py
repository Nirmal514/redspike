import os
import sys
import numpy as np
from typing import Dict, Any, Optional

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ml_engine.model import DualStreamApplianceNet, TORCH_AVAILABLE
from ml_engine.train import FAULT_NAMES, SEVERITY_NAMES

class EdgeInferenceEngine:
    """
    Lightweight, low-latency sliding window inference engine.
    Supports ONNX Runtime, PyTorch, and Pure-NumPy hardware execution.
    """
    def __init__(
        self,
        weights_path: Optional[str] = None,
        onnx_path: Optional[str] = None,
        npz_path: Optional[str] = None,
        anomaly_threshold: float = 0.60
    ):
        self.anomaly_threshold = anomaly_threshold
        self.fault_names = FAULT_NAMES
        self.severity_names = SEVERITY_NAMES

        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.pt_path = weights_path or os.path.join(base_dir, "models", "appliance_classifier.pt")
        self.onnx_path = onnx_path or os.path.join(base_dir, "models", "appliance_classifier.onnx")
        self.npz_path = npz_path or os.path.join(base_dir, "models", "appliance_classifier.npz")

        self.session = None
        self.torch_model = None
        self.numpy_model = None
        self._init_runtime()

    def _init_runtime(self):
        # 1. Try ONNX Runtime first
        try:
            import onnxruntime as ort
            if os.path.exists(self.onnx_path):
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                self.session = ort.InferenceSession(self.onnx_path, sess_options=opts, providers=['CPUExecutionProvider'])
                print(f"[Inference] Initialized ONNX Runtime session with {self.onnx_path}")
                return
        except Exception:
            pass

        # 2. Try PyTorch
        if TORCH_AVAILABLE:
            try:
                import torch
                self.torch_model = DualStreamApplianceNet(
                    num_fault_classes=len(self.fault_names),
                    num_severity_classes=len(self.severity_names)
                )
                if os.path.exists(self.pt_path):
                    state_dict = torch.load(self.pt_path, map_location="cpu", weights_only=True)
                    self.torch_model.load_state_dict(state_dict)
                    print(f"[Inference] Loaded PyTorch checkpoint from {self.pt_path}")
                self.torch_model.eval()
                return
            except Exception:
                pass

        # 3. Pure NumPy Engine Fallback
        self.numpy_model = DualStreamApplianceNet(
            num_fault_classes=len(self.fault_names),
            num_severity_classes=len(self.severity_names)
        )
        if os.path.exists(self.npz_path):
            self.numpy_model.load_weights(self.npz_path)
            print(f"[Inference] Loaded Pure NumPy model from {self.npz_path}")
        else:
            print("[Inference] Initialized Pure NumPy Edge Neural Engine.")

    def predict(self, mel_spec: np.ndarray, vib_vector: np.ndarray) -> Dict[str, Any]:
        """
        Executes edge model prediction.
        """
        # Ensure dimensions
        if mel_spec.ndim == 2:
            mel_input = mel_spec[np.newaxis, np.newaxis, :, :].astype(np.float32)
        elif mel_spec.ndim == 3:
            mel_input = mel_spec[np.newaxis, :, :, :].astype(np.float32)
        else:
            mel_input = mel_spec.astype(np.float32)

        if vib_vector.ndim == 1:
            vib_input = vib_vector[np.newaxis, :].astype(np.float32)
        else:
            vib_input = vib_vector.astype(np.float32)

        # 1. ONNX Inference
        if self.session is not None:
            ort_inputs = {
                "mel_spectrogram": mel_input,
                "vibration_features": vib_input
            }
            ort_outs = self.session.run(None, ort_inputs)
            anom_score = float(ort_outs[0][0, 0])
            fault_logits = ort_outs[1][0]
            sev_logits = ort_outs[2][0]

        # 2. PyTorch Inference
        elif self.torch_model is not None and TORCH_AVAILABLE:
            import torch
            with torch.no_grad():
                t_mel = torch.from_numpy(mel_input)
                t_vib = torch.from_numpy(vib_input)
                anom_t, fault_t, sev_t = self.torch_model(t_mel, t_vib)
                anom_score = float(anom_t.squeeze().item())
                fault_logits = fault_t.squeeze().numpy()
                sev_logits = sev_t.squeeze().numpy()

        # 3. Pure NumPy Inference
        elif self.numpy_model is not None:
            anom_t, fault_t, sev_t = self.numpy_model(mel_spec, vib_vector)
            anom_score = float(anom_t.squeeze())
            fault_logits = fault_t.squeeze()
            sev_logits = sev_t.squeeze()

        else:
            anom_score = 0.0
            fault_logits = np.zeros(len(self.fault_names))
            fault_logits[0] = 1.0
            sev_logits = np.array([1.0, 0.0, 0.0])

        # Softmax calculation
        exp_fault = np.exp(fault_logits - np.max(fault_logits))
        fault_probs = exp_fault / np.sum(exp_fault)

        exp_sev = np.exp(sev_logits - np.max(sev_logits))
        sev_probs = exp_sev / np.sum(exp_sev)

        fault_idx = int(np.argmax(fault_probs))
        fault_class = self.fault_names[fault_idx]
        fault_conf = float(fault_probs[fault_idx])

        sev_idx = int(np.argmax(sev_probs))
        sev_class = self.severity_names[sev_idx]
        sev_conf = float(sev_probs[sev_idx])

        is_anom = bool(anom_score >= self.anomaly_threshold or (fault_class != "normal" and fault_conf > 0.55))
        health_score = max(0.0, min(100.0, (1.0 - anom_score) * 100.0))

        return {
            "anomaly_score": round(anom_score, 4),
            "is_anomaly": is_anom,
            "health_score": round(health_score, 1),
            "fault_class": fault_class,
            "fault_confidence": round(fault_conf, 4),
            "severity": sev_class,
            "severity_confidence": round(sev_conf, 4),
            "fault_probabilities": {name: round(float(prob), 4) for name, prob in zip(self.fault_names, fault_probs)}
        }
