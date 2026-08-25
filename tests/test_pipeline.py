import os
import sys
import unittest
import numpy as np

# Ensure root is on sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(BASE_DIR)

from data_simulator.audio_synth import AudioSynthesizer
from data_simulator.vibration_synth import VibrationSynthesizer
from data_simulator.fault_injector import FaultInjector, FaultType
from dsp.feature_extractor import DSPFeatureExtractor
from ml_engine.model import DualStreamApplianceNet, TORCH_AVAILABLE
from ml_engine.inference import EdgeInferenceEngine
from edge_controller.rule_engine import EdgeSafetyRuleEngine, SystemAction
from edge_controller.autopilot_integration import AutopilotManager

class TestApplianceHealthPipeline(unittest.TestCase):
    def setUp(self):
        self.injector = FaultInjector()
        self.audio_synth = AudioSynthesizer(sample_rate=16000, appliance_type="water_pump")
        self.vib_synth = VibrationSynthesizer(sample_rate=500, appliance_type="water_pump")
        self.dsp = DSPFeatureExtractor()
        self.rule_engine = EdgeSafetyRuleEngine(db_path=":memory:", trip_persistence_seconds=0.5)
        self.autopilot = AutopilotManager()

    def test_synthesizer_and_fault_injection(self):
        # 1. Normal chunk
        audio_norm = self.audio_synth.generate_chunk(4096, self.injector)
        vib_norm = self.vib_synth.generate_chunk(128, self.injector)
        self.assertEqual(len(audio_norm), 4096)
        self.assertEqual(vib_norm.shape, (3, 128))

        # 2. Inject Cavitation
        self.injector.inject_fault("impeller_cavitation", intensity=0.9)
        self.assertEqual(self.injector.active_fault, FaultType.IMPELLER_CAVITATION)

        audio_fault = self.audio_synth.generate_chunk(4096, self.injector)
        vib_fault = self.vib_synth.generate_chunk(128, self.injector)
        self.assertEqual(len(audio_fault), 4096)
        self.assertEqual(vib_fault.shape, (3, 128))

    def test_dsp_feature_extraction(self):
        audio_chunk = self.audio_synth.generate_chunk(4096, self.injector)
        vib_chunk = self.vib_synth.generate_chunk(128, self.injector)

        mel_spec, vib_vec, telemetry = self.dsp.process_frame(audio_chunk, vib_chunk)

        # Check shapes
        self.assertEqual(mel_spec.shape, (128, 16))
        self.assertEqual(vib_vec.shape, (36,))
        self.assertIn("audio", telemetry)
        self.assertIn("vibration", telemetry)
        self.assertIn("x", telemetry["vibration"])
        self.assertIn("y", telemetry["vibration"])
        self.assertIn("z", telemetry["vibration"])

    def test_ml_model_architecture(self):
        model = DualStreamApplianceNet(num_fault_classes=7, num_severity_classes=3)
        dummy_mel = np.random.randn(128, 16).astype(np.float32)
        dummy_vib = np.random.randn(36).astype(np.float32)

        anom, fault, sev = model(dummy_mel, dummy_vib)
        self.assertTrue(anom.shape[-1] == 1)
        self.assertTrue(fault.shape[-1] == 7)
        self.assertTrue(sev.shape[-1] == 3)

    def test_inference_engine(self):
        engine = EdgeInferenceEngine()
        dummy_mel = np.random.randn(128, 16).astype(np.float32)
        dummy_vib = np.random.randn(36).astype(np.float32)

        pred = engine.predict(dummy_mel, dummy_vib)
        self.assertIn("anomaly_score", pred)
        self.assertIn("health_score", pred)
        self.assertIn("fault_class", pred)
        self.assertIn("severity", pred)
        self.assertIn("fault_probabilities", pred)

    def test_safety_rule_engine_and_autopilot(self):
        appliance_id = "test_pump_01"
        
        # Nominal evaluation
        pred_normal = {
            "fault_class": "normal",
            "severity": "low",
            "anomaly_score": 0.05,
            "is_anomaly": False,
            "health_score": 95.0
        }
        res = self.rule_engine.evaluate(appliance_id, pred_normal)
        self.assertEqual(res["relay_state"], "CLOSED")
        self.autopilot.update_status(appliance_id, pred_normal, res)
        ap_state = self.autopilot.get_health(appliance_id)
        self.assertTrue(ap_state["allow_operation"])
        self.assertEqual(ap_state["status"], "HEALTHY")

if __name__ == "__main__":
    unittest.main()
