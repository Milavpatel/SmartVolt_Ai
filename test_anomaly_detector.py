"""
SmartVolt AI — Automated Anomaly Detection Test Suite
Validates the hybrid anomaly detector across 7 rigorous testing scenarios:
  1. Stable normal readings (Baseline preservation, no false positives)
  2. Gradually changing temperatures (Ambient drift adaptation, no false alerts)
  3. Sudden unusual temperature spike (Rapid detection with debounced persistence)
  4. Missing, invalid, and disconnected sensor readings (-127°C, 85°C, None)
  5. Insufficient training data (COLLECTING_DATA state verification)
  6. Zero-variance data (Safe MAD floor handling, zero division immunity)
  7. Labeled benchmark dataset evaluation (Precision, Recall, F1-Score report)
"""

import unittest
import numpy as np
from model import SmartVoltHybridAnomalyDetector


class TestSmartVoltAnomalyDetector(unittest.TestCase):

    def setUp(self):
        """Initializes a fresh hybrid detector before each test."""
        self.detector = SmartVoltHybridAnomalyDetector(
            warning_threshold=40.0,
            critical_threshold=50.0,
            rolling_window_size=30,
            min_baseline_samples=25,
            mad_z_threshold=3.5,
            min_mad_floor=0.15,
            min_absolute_diff=0.40,
            persistence_count=2,
            contamination=0.05,
            random_state=42,
            model_save_path=None
        )

    def test_01_insufficient_training_data(self):
        """Test 1: System remains in COLLECTING_DATA mode and does NOT make premature predictions."""
        for i in range(10):
            res = self.detector.predict(25.0 + 0.1 * (i % 3))
            self.assertEqual(res["model_status"], "COLLECTING_DATA")
            self.assertFalse(res["is_baseline_ready"])
            self.assertFalse(res["is_anomaly"])
            self.assertTrue(any("baseline" in r.lower() for r in res["reasons"]))

    def test_02_stable_normal_readings(self):
        """Test 2: Stable normal readings establish baseline and remain NORMAL without false alarms."""
        # Feed 25 baseline samples (25.0°C to 25.4°C)
        for i in range(25):
            val = 25.0 + (i % 4) * 0.1
            self.detector.predict(val)

        self.assertTrue(self.detector.is_baseline_ready)

        # Feed 30 subsequent stable normal readings
        false_alarms = 0
        for i in range(30):
            val = 25.2 + np.sin(i) * 0.15
            res = self.detector.predict(val)
            if res["model_status"] == "ANOMALY":
                false_alarms += 1

        self.assertEqual(false_alarms, 0, f"Expected 0 false alarms on stable data, got {false_alarms}")

    def test_03_gradually_changing_temperatures(self):
        """Test 3: Slow ambient warming (0.05°C per step) adapts via rolling median without false alert."""
        # Establish baseline around 24°C
        for i in range(25):
            self.detector.predict(24.0 + (i % 3) * 0.05)

        # Gradually warm from 24.0°C to 28.0°C over 60 samples
        false_alarms = 0
        current_val = 24.0
        for i in range(60):
            current_val += 0.05  # slow 0.05°C/step drift
            res = self.detector.predict(round(current_val, 2))
            if res["model_status"] == "ANOMALY":
                false_alarms += 1

        self.assertEqual(false_alarms, 0, f"Gradual ambient drift triggered {false_alarms} false alarms.")

    def test_04_sudden_unusual_temperature_spike(self):
        """Test 4: Abrupt temperature jump (+12°C) is flagged with transient indication and persistent alert."""
        # Establish baseline around 25°C
        for _ in range(25):
            self.detector.predict(25.0)

        # Transient spike at reading 1 (37.0°C)
        res_spike1 = self.detector.predict(37.0)
        self.assertTrue(res_spike1["is_transient_anomaly"])
        self.assertEqual(res_spike1["consecutive_anomalies"], 1)

        # Spike persists at reading 2 (37.2°C) -> reaches persistence threshold (count=2)
        res_spike2 = self.detector.predict(37.2)
        self.assertTrue(res_spike2["is_persistent_anomaly"])
        self.assertEqual(res_spike2["model_status"], "ANOMALY")
        self.assertEqual(res_spike2["consecutive_anomalies"], 2)
        self.assertTrue(any("Statistical Outlier" in r for r in res_spike2["reasons"]))

    def test_05_missing_invalid_and_disconnected_sensor(self):
        """Test 5: Sensor disconnection (-127°C) and None are safely caught as SENSOR_FAULT without crashing."""
        # Establish baseline
        for _ in range(25):
            self.detector.predict(26.0)

        # Disconnected wire (-127.00°C)
        res_disc = self.detector.predict(-127.00)
        self.assertEqual(res_disc["model_status"], "SENSOR_FAULT")
        self.assertTrue(res_disc["is_anomaly"])
        self.assertTrue(any("-127.00°C" in r for r in res_disc["reasons"]))

        # Missing None reading
        res_none = self.detector.predict(None)
        self.assertEqual(res_none["model_status"], "SENSOR_FAULT")
        self.assertTrue(res_none["is_anomaly"])

        # Power-on reset constant (85.00°C)
        res_reset = self.detector.predict(85.00)
        self.assertEqual(res_reset["model_status"], "SENSOR_FAULT")
        self.assertTrue(any("power-on reset" in r for r in res_reset["reasons"]))

    def test_06_zero_variance_data(self):
        """Test 6: Perfectly flat constant readings (MAD=0) handled safely by min_mad_floor without ZeroDivisionError."""
        # Feed 35 identical readings
        for _ in range(35):
            res = self.detector.predict(25.00)

        self.assertTrue(self.detector.is_baseline_ready)
        self.assertEqual(res["rolling_mad"], 0.0)
        self.assertEqual(res["safe_mad"], self.detector.min_mad_floor)
        self.assertEqual(res["modified_z_score"], 0.0)
        self.assertEqual(res["model_status"], "NORMAL")

    def test_07_labeled_benchmark_precision_recall_f1(self):
        """Test 7: Evaluates detector performance on a labeled benchmark sequence and reports metrics."""
        y_true = []
        y_pred = []

        # 30 Normal baseline samples
        for _ in range(30):
            self.detector.predict(26.0)
            y_true.append(0)
            y_pred.append(0)

        # 20 Normal test samples with minor natural noise
        for i in range(20):
            val = 26.0 + 0.1 * ((i % 5) - 2)
            res = self.detector.predict(val)
            y_true.append(0)
            y_pred.append(1 if res["is_anomaly"] else 0)

        # 5 Anomalous high-temperature readings (38°C - 42°C)
        for i in range(5):
            val = 38.0 + i * 0.5
            res = self.detector.predict(val)
            y_true.append(1)
            y_pred.append(1 if res["is_anomaly"] else 0)

        # 10 Recovery samples back to normal
        for _ in range(10):
            res = self.detector.predict(26.0)
            y_true.append(0)
            y_pred.append(1 if res["is_anomaly"] else 0)

        metrics = self.detector.evaluate_labeled_dataset(y_true, y_pred)
        print("\n" + "=" * 60)
        print("BENCHMARK TEST EVALUATION REPORT:")
        print(f"Total Test Samples: {metrics['total_samples']}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"Confusion Matrix: TP={metrics['true_positives']}, FP={metrics['false_positives']}, TN={metrics['true_negatives']}, FN={metrics['false_negatives']}")
        print(f"Disclaimer: {metrics['disclaimer']}")
        print("=" * 60)

        self.assertGreaterEqual(metrics["precision"], 0.80)
        self.assertGreaterEqual(metrics["recall"], 0.80)
        self.assertGreaterEqual(metrics["f1_score"], 0.80)


if __name__ == "__main__":
    unittest.main()

