"""
SmartVolt AI — Comprehensive Automated Test Suite
Covers:
  1. Serial Parsing & Mocked Serial Input
  2. Data Validation & Boundary Filtering
  3. Deterministic Threshold Safety (Warning vs Critical)
  4. Robust Statistical Detection & Zero-Variance MAD Handling
  5. Machine Learning Persistence & Zero-Leakage Benchmark
  6. SQLite Database Persistence, Deduplication & Cooldown
  7. Simulation Labeling & Source Isolation
  8. PDF & CSV Audit Report Generation
"""

import os
import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

import serial_reader
from serial_reader import parse_serial_message, SmartVoltSimulationEngine
import model
from model import SmartVoltHybridAnomalyDetector
import database
import reports


class TestSmartVoltSuite(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_db = os.path.join(self.temp_dir, "test_smartvolt.db")
        self.test_model = os.path.join(self.temp_dir, "test_model.joblib")
        database.init_db(self.test_db)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # 1. Serial Parsing & Mocked Serial Input
    # -------------------------------------------------------------------------
    def test_01_serial_parsing_standard_and_mock(self):
        mock_inputs = [
            ("Temperature: 27.50 C", True, 27.50, None),
            ("Temperature:27.50C", True, 27.50, None),
            ("Temperature: -127.00 C", False, -127.00, "SENSOR_DISCONNECTED"),
            ("Temperature: 85.00 C", False, 85.00, "POWER_ON_RESET"),
            ("Temp: 32.4", True, 32.4, None),
            ("T: 28.1", True, 28.1, None),
            ("44.8", True, 44.8, None),
            ("Corrupted Noise Line #$@!", False, None, "INVALID_FORMAT"),
            ("", False, None, None)
        ]
        for line, exp_valid, exp_temp, exp_fault in mock_inputs:
            res = parse_serial_message(line)
            self.assertEqual(res["is_valid"], exp_valid, f"Failed validity for: {line}")
            if exp_temp is not None:
                self.assertAlmostEqual(res["temperature"], exp_temp, places=2)
            if exp_fault is not None:
                self.assertEqual(res["fault_type"], exp_fault)

    # -------------------------------------------------------------------------
    # 2. Data Validation & Boundary Filtering
    # -------------------------------------------------------------------------
    def test_02_data_validation(self):
        det = SmartVoltHybridAnomalyDetector(model_save_path=self.test_model)
        
        # Test physical bounds
        res_high = det.predict(135.0)
        self.assertEqual(res_high["model_status"], "SENSOR_FAULT")
        self.assertEqual(res_high["alert_severity"], "CRITICAL")

        res_none = det.predict(None)
        self.assertEqual(res_none["model_status"], "SENSOR_FAULT")

        res_disc = det.predict(-127.0)
        self.assertEqual(res_disc["model_status"], "SENSOR_FAULT")

        res_reset = det.predict(85.0)
        self.assertEqual(res_reset["model_status"], "SENSOR_FAULT")

    # -------------------------------------------------------------------------
    # 3. Deterministic Engineering Limits (Warning vs Critical)
    # -------------------------------------------------------------------------
    def test_03_threshold_detection(self):
        det = SmartVoltHybridAnomalyDetector(
            warning_threshold=40.0,
            critical_threshold=50.0,
            model_save_path=self.test_model
        )
        # Normal
        status, breached, _ = det.evaluate_thresholds(35.0)
        self.assertEqual(status, "NORMAL")
        self.assertFalse(breached)

        # Warning
        status, breached, reasons = det.evaluate_thresholds(42.5)
        self.assertEqual(status, "WARNING")
        self.assertTrue(breached)
        self.assertTrue(any("Warning" in r for r in reasons))

        # Critical
        status, breached, reasons = det.evaluate_thresholds(52.0)
        self.assertEqual(status, "CRITICAL")
        self.assertTrue(breached)
        self.assertTrue(any("Critical" in r for r in reasons))

    # -------------------------------------------------------------------------
    # 4. Statistical Detection, Safe MAD Floor & Zero-Variance Handling
    # -------------------------------------------------------------------------
    def test_04_statistical_detection_and_safe_mad(self):
        det = SmartVoltHybridAnomalyDetector(
            min_baseline_samples=15,
            persistence_count=2,
            min_mad_floor=0.15,
            model_save_path=self.test_model
        )
        # Feed 30 identical zero-variance readings
        for _ in range(30):
            res = det.predict(25.0)

        self.assertTrue(det.is_baseline_ready)
        self.assertEqual(res["rolling_mad"], 0.0)
        self.assertEqual(res["safe_mad"], 0.15)
        self.assertEqual(res["modified_z_score"], 0.0)
        self.assertEqual(res["model_status"], "NORMAL")

        # Transient spike at reading 1
        res_sp1 = det.predict(38.0)
        self.assertTrue(res_sp1["is_transient_anomaly"])
        self.assertEqual(res_sp1["model_status"], "NORMAL")  # Debounced

        # Persistent spike at reading 2
        res_sp2 = det.predict(38.5)
        self.assertTrue(res_sp2["is_persistent_anomaly"])
        self.assertEqual(res_sp2["model_status"], "ANOMALY")
        self.assertEqual(res_sp2["consecutive_anomalies"], 2)

    # -------------------------------------------------------------------------
    # 5. Machine Learning Persistence & Comparative Benchmark
    # -------------------------------------------------------------------------
    def test_05_ml_persistence_and_evaluation(self):
        det = SmartVoltHybridAnomalyDetector(
            min_baseline_samples=20,
            model_save_path=self.test_model
        )
        train_data = [26.0 + 0.1 * (i % 3) for i in range(25)]
        ok, msg = det.train_on_history(train_data)
        self.assertTrue(ok)
        self.assertTrue(os.path.exists(self.test_model))

        # Re-load from disk
        det2 = SmartVoltHybridAnomalyDetector(model_save_path=self.test_model)
        self.assertTrue(det2.is_baseline_ready)
        self.assertIsNotNone(det2.model)

        # Comparative benchmark execution
        bm = SmartVoltHybridAnomalyDetector.run_comparative_benchmark()
        self.assertIn("hybrid", bm)
        self.assertIn("rule_based", bm)
        self.assertIn("isolation_forest", bm)
        self.assertGreaterEqual(bm["hybrid"]["recall"], 0.80)

    # -------------------------------------------------------------------------
    # 6. SQLite Database Persistence, Deduplication & Cooldown
    # -------------------------------------------------------------------------
    def test_06_database_persistence_and_cooldown(self):
        # Insert telemetry reading
        read_id = database.insert_reading({
            "timestamp": "2026-10-28 20:00:00",
            "temperature": 27.5,
            "voltage": None,
            "source": "HARDWARE",
            "device_status": "CONNECTED",
            "threshold_status": "NORMAL",
            "model_status": "NORMAL",
            "is_anomaly": 0
        }, db_path=self.test_db)
        self.assertGreater(read_id, 0)

        # Insert anomaly event
        ev1 = {
            "timestamp": "2026-10-28 20:00:01",
            "temperature": 45.0,
            "source": "HARDWARE",
            "severity": "CRITICAL",
            "detection_method": "RULE_BASED",
            "reason": "Overheat",
            "suggested_action": "Check cooling"
        }
        ok1 = database.insert_anomaly_event(ev1, cooldown_seconds=5, db_path=self.test_db)
        self.assertTrue(ok1)

        # Immediate duplicate event within 1s should be throttled by cooldown
        ev2 = {
            "timestamp": "2026-10-28 20:00:02",
            "temperature": 45.5,
            "source": "HARDWARE",
            "severity": "CRITICAL",
            "detection_method": "RULE_BASED",
            "reason": "Overheat",
            "suggested_action": "Check cooling"
        }
        ok2 = database.insert_anomaly_event(ev2, cooldown_seconds=5, db_path=self.test_db)
        self.assertFalse(ok2, "Duplicate event within cooldown should be throttled.")

        # Verify query filters
        df_readings = database.get_recent_readings(limit=10, source="HARDWARE", db_path=self.test_db)
        self.assertEqual(len(df_readings), 1)
        self.assertEqual(df_readings.iloc[0]["source"], "HARDWARE")

    # -------------------------------------------------------------------------
    # 7. Simulation Labeling & Source Isolation
    # -------------------------------------------------------------------------
    def test_07_simulation_labeling_and_isolation(self):
        sim = SmartVoltSimulationEngine()
        sim.set_scenario("NORMAL")
        t1 = sim.tick()
        self.assertEqual(t1["source"], "SIMULATED")
        self.assertIn("[SIMULATED]", t1["raw"])

        sim.set_scenario("OVERHEATING")
        t2 = sim.tick()
        self.assertEqual(t2["source"], "SIMULATED")
        self.assertIn("[SIMULATED]", t2["raw"])

    # -------------------------------------------------------------------------
    # 8. PDF & CSV Audit Report Generation
    # -------------------------------------------------------------------------
    def test_08_reports_generation(self):
        db_stats = database.get_database_stats(self.test_db)
        readings = pd.DataFrame([{
            "timestamp": "2026-10-28 20:00:00",
            "temperature": 27.5,
            "voltage": None,
            "source": "HARDWARE"
        }])
        events = pd.DataFrame([{
            "timestamp": "2026-10-28 20:00:01",
            "temperature": 45.0,
            "voltage": None,
            "source": "HARDWARE",
            "severity": "CRITICAL",
            "detection_method": "HYBRID",
            "reason": "Overheat",
            "suggested_action": "Inspect"
        }])
        bm = SmartVoltHybridAnomalyDetector.run_comparative_benchmark()

        pdf_path = os.path.join(self.temp_dir, "test_report.pdf")
        csv_readings_path = os.path.join(self.temp_dir, "test_readings.csv")
        csv_events_path = os.path.join(self.temp_dir, "test_events.csv")

        # Test PDF generation
        reports.generate_pdf_report(db_stats, readings, events, bm, pdf_path)
        self.assertTrue(os.path.exists(pdf_path))
        self.assertGreater(os.path.getsize(pdf_path), 1000)

        # Test CSV export
        reports.export_readings_csv(readings, csv_readings_path)
        self.assertTrue(os.path.exists(csv_readings_path))

        reports.export_events_csv(events, csv_events_path)
        self.assertTrue(os.path.exists(csv_events_path))


if __name__ == "__main__":
    unittest.main()
