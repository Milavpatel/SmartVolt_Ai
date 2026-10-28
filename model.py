"""
SmartVolt AI — Robust Hybrid Anomaly Detection Engine
Integrates:
  1. Two-Stage Deterministic Thresholds (Warning & Critical)
  2. Robust Statistical Outlier Detection (Rolling Median & Safe MAD with Debouncing)
  3. Machine Learning Anomaly Detection (Isolation Forest with Joblib Persistence)
  4. Explicit Cold-Start Lifecycle (COLLECTING_DATA -> NORMAL / ANOMALY)
  5. Multi-Detector Comparative Benchmarking (Zero-Leakage Train/Test Evaluation)
"""

import os
import joblib
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

DEFAULT_MODEL_DIR = os.path.join("data", "models")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "isolation_forest.joblib")


class SmartVoltHybridAnomalyDetector:
    """
    Industrial-grade hybrid anomaly detector for IoT temperature telemetry.
    Combines deterministic engineering limits, robust statistics (MAD),
    and Scikit-Learn Isolation Forest with model persistence and debouncing.
    """

    def __init__(
        self,
        warning_threshold: float = 40.0,
        critical_threshold: float = 50.0,
        rolling_window_size: int = 30,
        min_baseline_samples: int = 30,
        mad_z_threshold: float = 3.5,
        min_mad_floor: float = 0.15,
        min_absolute_diff: float = 0.40,
        persistence_count: int = 2,
        contamination: float = 0.05,
        n_estimators: int = 100,
        random_state: int = 42,
        model_save_path: str = DEFAULT_MODEL_PATH
    ):
        # 1. Deterministic Threshold Parameters
        self.warning_threshold = float(warning_threshold)
        self.critical_threshold = float(critical_threshold)

        # 2. Statistical Parameters (Rolling Median & MAD)
        self.rolling_window_size = int(rolling_window_size)
        self.min_baseline_samples = int(min_baseline_samples)
        self.mad_z_threshold = float(mad_z_threshold)
        self.min_mad_floor = float(min_mad_floor)
        self.min_absolute_diff = float(min_absolute_diff)
        self.persistence_count = int(persistence_count)

        # 3. Machine Learning Parameters (Isolation Forest)
        self.contamination = float(contamination)
        self.n_estimators = int(n_estimators)
        self.random_state = int(random_state)
        self.model_save_path = model_save_path

        # Internal State Management
        self.window = deque(maxlen=self.rolling_window_size)
        self.baseline_buffer: List[float] = []

        self.is_baseline_ready: bool = False
        self.model: Optional[IsolationForest] = None
        self.model_trained_at: Optional[str] = None
        self.consecutive_anomalies: int = 0
        self.total_samples_ingested: int = 0

        # Attempt to load saved model from disk if available and path provided
        if self.model_save_path:
            self._try_load_model()

    def _try_load_model(self) -> bool:
        """Loads a previously saved Isolation Forest model from disk if valid."""
        if self.model_save_path and os.path.exists(self.model_save_path):
            try:
                bundle = joblib.load(self.model_save_path)
                if isinstance(bundle, dict) and "model" in bundle:
                    self.model = bundle["model"]
                    self.model_trained_at = bundle.get("trained_at", "Unknown")
                    self.is_baseline_ready = True
                    return True
                elif isinstance(bundle, IsolationForest):
                    self.model = bundle
                    self.model_trained_at = "Loaded from file"
                    self.is_baseline_ready = True
                    return True
            except Exception:
                self.model = None
                self.is_baseline_ready = False
        return False

    def save_model(self) -> bool:
        """Saves current Isolation Forest model and metadata to disk."""
        if self.model is None:
            return False
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self.model_save_path)), exist_ok=True)
            bundle = {
                "model": self.model,
                "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "sample_count": len(self.baseline_buffer),
                "contamination": self.contamination,
                "n_estimators": self.n_estimators
            }
            joblib.dump(bundle, self.model_save_path)
            self.model_trained_at = bundle["trained_at"]
            return True
        except Exception:
            return False

    def reset_baseline(self):
        """Clears baseline buffer and resets model to COLLECTING_DATA status."""
        self.window.clear()
        self.baseline_buffer.clear()
        self.is_baseline_ready = False
        self.model = None
        self.model_trained_at = None
        self.consecutive_anomalies = 0
        self.total_samples_ingested = 0

        # Remove stale model file if requested
        if os.path.exists(self.model_save_path):
            try:
                os.remove(self.model_save_path)
            except Exception:
                pass

    def set_thresholds(self, warning: float, critical: float):
        """Updates warning and critical thresholds safely."""
        self.warning_threshold = float(warning)
        self.critical_threshold = max(float(critical), float(warning) + 0.5)

    def evaluate_thresholds(self, temp: float) -> Tuple[str, bool, List[str]]:
        """
        Deterministic two-stage threshold evaluation.
        Returns: threshold_status ('NORMAL' | 'WARNING' | 'CRITICAL'), is_breached, reasons
        """
        reasons = []
        if temp >= self.critical_threshold:
            reasons.append(
                f"Critical Threshold Breached: {temp:.2f}°C >= {self.critical_threshold:.1f}°C"
            )
            return "CRITICAL", True, reasons
        elif temp >= self.warning_threshold:
            reasons.append(
                f"Warning Threshold Breached: {temp:.2f}°C >= {self.warning_threshold:.1f}°C"
            )
            return "WARNING", True, reasons
        return "NORMAL", False, reasons

    def compute_statistical_metrics(self, current_temp: float) -> Dict[str, Optional[float]]:
        """
        Computes rolling median and safe Median Absolute Deviation (MAD).
        Safe MAD floor prevents division by zero on flatline data.
        """
        if len(self.window) < 5:
            return {
                "rolling_median": None,
                "rolling_mad": None,
                "safe_mad": None,
                "upper_bound": None,
                "lower_bound": None,
                "modified_z_score": 0.0,
                "abs_diff": 0.0
            }

        arr = np.array(self.window, dtype=float)
        median_val = float(np.median(arr))
        abs_diffs = np.abs(arr - median_val)
        raw_mad = float(np.median(abs_diffs))

        safe_mad = max(raw_mad, self.min_mad_floor)
        diff = abs(current_temp - median_val)
        mod_z = float(0.6745 * diff / safe_mad)

        envelope = (self.mad_z_threshold * safe_mad) / 0.6745
        upper_bound = round(median_val + envelope, 2)
        lower_bound = round(median_val - envelope, 2)

        return {
            "rolling_median": round(median_val, 2),
            "rolling_mad": round(raw_mad, 3),
            "safe_mad": round(safe_mad, 3),
            "upper_bound": upper_bound,
            "lower_bound": lower_bound,
            "modified_z_score": round(mod_z, 2),
            "abs_diff": round(diff, 2)
        }

    def train_isolation_forest(self, baseline_temps: List[float]) -> Tuple[bool, str]:
        """
        Fits Scikit-Learn IsolationForest on verified baseline samples and saves to disk.
        Features: [Deviation from median, Rate of Change]
        """
        if len(baseline_temps) < self.min_baseline_samples:
            return False, (
                f"Insufficient baseline samples: Received {len(baseline_temps)}/{self.min_baseline_samples}."
            )

        arr = np.array(baseline_temps, dtype=float)
        med = np.median(arr)
        devs = arr - med
        rates = np.zeros_like(arr)
        if len(arr) > 1:
            rates[1:] = np.diff(arr)

        X = np.column_stack([devs, rates])

        try:
            self.model = IsolationForest(
                n_estimators=self.n_estimators,
                contamination=self.contamination,
                random_state=self.random_state
            )
            self.model.fit(X)
            self.is_baseline_ready = True
            self.save_model()
            return True, f"Isolation Forest successfully trained on {len(baseline_temps)} verified samples."
        except Exception as ex:
            return False, f"Model training error: {str(ex)}"

    def predict(
        self,
        current_temp: Optional[float],
        previous_temp: Optional[float] = None
    ) -> Dict:
        """
        Full hybrid detection pipeline:
        Rule-Based + Rolling MAD + Isolation Forest + Debounced Persistence.
        """
        res = {
            "current_temp": current_temp,
            "threshold_status": "NORMAL",
            "threshold_alert": False,
            "model_status": "COLLECTING_DATA",
            "statistical_status": "COLLECTING_DATA",
            "detection_methods": [],
            "alert_severity": "NORMAL",
            "is_transient_anomaly": False,
            "is_persistent_anomaly": False,
            "is_anomaly": False,
            "consecutive_anomalies": self.consecutive_anomalies,
            "persistence_count": self.persistence_count,
            "rolling_median": None,
            "rolling_mad": None,
            "safe_mad": None,
            "upper_bound": None,
            "lower_bound": None,
            "modified_z_score": 0.0,
            "rate_of_change": 0.0,
            "iforest_prediction": 1,
            "iforest_score": 0.0,
            "anomaly_score_pct": 0.0,
            "baseline_progress_pct": 0.0,
            "samples_collected": len(self.baseline_buffer),
            "min_samples_required": self.min_baseline_samples,
            "is_baseline_ready": self.is_baseline_ready,
            "reasons": [],
            "baseline_involved": "None",
            "suggested_action": "Nominal operation: Continue automated monitoring."
        }

        # 1. Hardware Sensor Fault / Out-of-Range Validation
        is_sensor_fault = (
            current_temp is None
            or current_temp <= -100.0
            or current_temp >= 125.0
            or (current_temp is not None and abs(current_temp - 85.0) < 0.05)
        )
        if is_sensor_fault:
            res["model_status"] = "SENSOR_FAULT"
            res["statistical_status"] = "SENSOR_FAULT"
            res["alert_severity"] = "CRITICAL"
            res["is_anomaly"] = True
            res["detection_methods"].append("SENSOR_FAULT")
            if current_temp is not None and abs(current_temp - (-127.0)) < 0.1:
                res["reasons"].append("DS18B20 1-Wire bus disconnected / missing 4.7kΩ pull-up (-127.00°C)")
                res["suggested_action"] = "Inspect DS18B20 wiring: Verify 4.7kΩ pull-up between D4 and 3V3 rail."
            elif current_temp is not None and abs(current_temp - 85.0) < 0.05:
                res["reasons"].append("DS18B20 power-on reset state (85.00°C conversion incomplete)")
                res["suggested_action"] = "Allow sensor conversion delay; check power supply stability."
            else:
                res["reasons"].append(f"Invalid reading outside physical sensor range: {current_temp}")
                res["suggested_action"] = "Verify sensor integrity and check for open circuit."
            return res

        # 2. Rate of Change
        rate = 0.0
        if previous_temp is not None and -50.0 <= previous_temp <= 125.0:
            rate = round(current_temp - previous_temp, 3)
        res["rate_of_change"] = rate

        # 3. Deterministic Engineering Limits
        thresh_status, thresh_breached, thresh_reasons = self.evaluate_thresholds(current_temp)
        res["threshold_status"] = thresh_status
        res["threshold_alert"] = thresh_breached
        if thresh_breached:
            res["detection_methods"].append("RULE_BASED")
            res["reasons"].extend(thresh_reasons)
            res["alert_severity"] = thresh_status
            res["suggested_action"] = (
                f"Verify component cooling: Temperature {current_temp:.1f}°C exceeds {thresh_status.lower()} threshold."
            )

        # 4. Baseline Lifecycle Management
        self.total_samples_ingested += 1
        if not self.is_baseline_ready:
            self.baseline_buffer.append(current_temp)
            self.window.append(current_temp)
            samples_got = len(self.baseline_buffer)
            res["samples_collected"] = samples_got
            res["baseline_progress_pct"] = round(min(100.0, (samples_got / self.min_baseline_samples) * 100.0), 1)

            if samples_got < self.min_baseline_samples:
                res["model_status"] = "COLLECTING_DATA"
                res["statistical_status"] = "COLLECTING_DATA"
                res["reasons"].append(
                    f"Acquiring ambient room baseline ({samples_got}/{self.min_baseline_samples} samples)..."
                )
                if not thresh_breached:
                    res["suggested_action"] = "Allow system to establish operating baseline without disturbances."
                return res
            else:
                self.train_isolation_forest(self.baseline_buffer)

        # Update statistical rolling window
        self.window.append(current_temp)
        res["samples_collected"] = len(self.baseline_buffer)
        res["baseline_progress_pct"] = 100.0
        res["is_baseline_ready"] = True

        # 5. Statistical Anomaly Detection (Rolling Median & MAD)
        stat = self.compute_statistical_metrics(current_temp)
        res.update(stat)

        if stat["rolling_median"] is not None:
            res["baseline_involved"] = (
                f"Median: {stat['rolling_median']:.2f}°C, "
                f"MAD: ±{stat['safe_mad']:.2f}°C, "
                f"Bound: [{stat['lower_bound']:.1f}°C, {stat['upper_bound']:.1f}°C]"
            )

        is_stat_outlier = False
        if stat["modified_z_score"] is not None:
            if stat["modified_z_score"] >= self.mad_z_threshold and stat["abs_diff"] >= self.min_absolute_diff:
                is_stat_outlier = True
                res["detection_methods"].append("STATISTICAL_MAD")

        res["is_transient_anomaly"] = is_stat_outlier

        # Debounced Persistence Filter
        if is_stat_outlier:
            self.consecutive_anomalies += 1
        else:
            self.consecutive_anomalies = 0

        res["consecutive_anomalies"] = self.consecutive_anomalies
        res["is_persistent_anomaly"] = (self.consecutive_anomalies >= self.persistence_count)

        # 6. Isolation Forest Inference
        iforest_outlier = False
        if self.model is not None and stat["rolling_median"] is not None:
            dev = current_temp - stat["rolling_median"]
            sample_feats = np.array([[dev, rate]])
            pred = int(self.model.predict(sample_feats)[0])
            dec = float(self.model.decision_function(sample_feats)[0])

            res["iforest_prediction"] = pred
            res["iforest_score"] = round(dec, 4)
            if pred == -1:
                iforest_outlier = True
                res["detection_methods"].append("ISOLATION_FOREST")

            severity = (0.15 - dec) / 0.40
            res["anomaly_score_pct"] = round(max(0.0, min(100.0, float(severity * 100.0))), 1)

        # 7. Final Consolidated State
        if res["is_persistent_anomaly"] or (iforest_outlier and self.consecutive_anomalies >= 1):
            res["model_status"] = "ANOMALY"
            res["statistical_status"] = "ANOMALY"
            res["is_anomaly"] = True
            if res["alert_severity"] == "NORMAL":
                res["alert_severity"] = "WARNING"
            res["reasons"].append(
                f"Statistical Outlier: Z-score {res['modified_z_score']:.1f} >= {self.mad_z_threshold} "
                f"(Consecutive: {self.consecutive_anomalies}/{self.persistence_count})"
            )
            if iforest_outlier:
                res["reasons"].append(f"Isolation Forest flagged pattern (score: {res['iforest_score']})")
            res["suggested_action"] = "Inspect device: Abnormal heat dissipation or abrupt thermal rate-of-change detected."

        elif res["is_transient_anomaly"]:
            res["model_status"] = "NORMAL"
            res["statistical_status"] = "TRANSIENT_SPIKE"
            res["reasons"].append(
                f"Transient reading detected (Z-score {res['modified_z_score']:.1f}), debouncing 1/{self.persistence_count}"
            )
        else:
            res["model_status"] = "NORMAL"
            res["statistical_status"] = "NORMAL"

        return res

    # Backward compatibility method
    def predict_single(
        self,
        current_temp: Optional[float],
        previous_temp: Optional[float] = None,
        threshold_c: Optional[float] = None
    ) -> Dict:
        if threshold_c is not None:
            self.warning_threshold = float(threshold_c)
        return self.predict(current_temp=current_temp, previous_temp=previous_temp)

    def train_on_history(
        self,
        temperatures: List[float],
        contamination: Optional[float] = None
    ) -> Tuple[bool, str]:
        if contamination is not None:
            self.contamination = max(0.01, min(0.25, float(contamination)))
        valid_temps = [t for t in temperatures if t is not None and -50.0 <= t <= 120.0]
        if len(valid_temps) < self.min_baseline_samples:
            return False, (
                f"Insufficient training samples: Received {len(valid_temps)} valid readings. "
                f"A minimum of {self.min_baseline_samples} baseline samples are required."
            )
        self.baseline_buffer = list(valid_temps)
        self.window = deque(valid_temps[-self.rolling_window_size:], maxlen=self.rolling_window_size)
        return self.train_isolation_forest(valid_temps)

    @classmethod
    def run_comparative_benchmark(cls) -> Dict[str, Dict]:
        """
        Executes a zero-leakage comparative evaluation between:
          1. Rule-Based Thresholds
          2. Rolling Statistical MAD
          3. Isolation Forest
          4. Combined Hybrid System
        
        Uses an explicit train/test split:
          - Train set (100% normal baseline) fits the Isolation Forest.
          - Test set (held-out independent samples with ground-truth labels) evaluates all detectors.
        """
        rng = np.random.RandomState(42)

        # 1. Independent Training Set (100% Normal Baseline - Zero Contamination)
        train_normal = 26.5 + rng.normal(0, 0.2, 50)

        # 2. Independent Test Set (Held-Out)
        # 40 Normal test points with natural diurnal wander
        test_normal = 26.5 + np.sin(np.linspace(0, 3, 40)) * 0.3 + rng.normal(0, 0.15, 40)
        y_test_normal = [0] * len(test_normal)

        # 10 Slow Gradual Warming points (27°C to 29.5°C over 10 steps) -> Normal operational drift
        test_drift = np.linspace(26.8, 29.2, 10)
        y_test_drift = [0] * len(test_drift)

        # 8 Anomalous Temperature Spikes (36°C to 44°C) -> True Anomalies
        test_spikes = 36.0 + rng.uniform(1.0, 8.0, 8)
        y_test_spikes = [1] * len(test_spikes)

        # 4 Extreme Overheating points (52°C to 58°C) -> True Critical Anomalies
        test_overheat = 52.0 + rng.uniform(1.0, 6.0, 4)
        y_test_overheat = [1] * len(test_overheat)

        # Combine test sequence
        test_sequence = np.concatenate([test_normal, test_drift, test_spikes, test_overheat])
        y_true = y_test_normal + y_test_drift + y_test_spikes + y_test_overheat

        # Initialize and train fresh detector on TRAIN set
        detector = cls(
            warning_threshold=40.0,
            critical_threshold=50.0,
            min_baseline_samples=30,
            persistence_count=2,
            random_state=42
        )
        detector.train_isolation_forest(train_normal.tolist())

        # Arrays to collect individual detector predictions
        preds_rule = []
        preds_mad = []
        preds_iforest = []
        preds_hybrid = []

        prev = None
        for val in test_sequence:
            res = detector.predict(float(val), previous_temp=prev)
            prev = float(val)

            # Rule based prediction: 1 if exceeds warning threshold
            preds_rule.append(1 if val >= detector.warning_threshold else 0)

            # Raw MAD statistical prediction
            preds_mad.append(1 if res["is_transient_anomaly"] else 0)

            # Isolation Forest prediction: -1 is anomaly
            preds_iforest.append(1 if res["iforest_prediction"] == -1 else 0)

            # Hybrid consolidated prediction (with debouncing)
            preds_hybrid.append(1 if res["is_anomaly"] else 0)

        def calc_metrics(y_t, y_p, name):
            prec = float(precision_score(y_t, y_p, zero_division=0))
            rec = float(recall_score(y_t, y_p, zero_division=0))
            f1 = float(f1_score(y_t, y_p, zero_division=0))
            tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
            return {
                "name": name,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1_score": round(f1, 4),
                "tp": int(tp),
                "fp": int(fp),
                "tn": int(tn),
                "fn": int(fn)
            }

        return {
            "dataset_info": {
                "train_samples": len(train_normal),
                "test_samples": len(test_sequence),
                "normal_samples": y_true.count(0),
                "anomaly_samples": y_true.count(1),
                "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "rule_based": calc_metrics(y_true, preds_rule, "Rule-Based Threshold (40°C)"),
            "statistical_mad": calc_metrics(y_true, preds_mad, "Rolling MAD (Raw Z >= 3.5)"),
            "isolation_forest": calc_metrics(y_true, preds_iforest, "Isolation Forest (X=[T-med, dT])"),
            "hybrid": calc_metrics(y_true, preds_hybrid, "SmartVolt AI Hybrid System")
        }

    @staticmethod
    def evaluate_labeled_dataset(y_true: List[int], y_pred: List[int]) -> Dict:
        """Computes evaluation metrics on a labeled sequence."""
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return {
            "total_samples": len(y_true),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "disclaimer": "Benchmark performance on controlled test sequence."
        }


# Maintain backward compatibility
SmartVoltAnomalyDetector = SmartVoltHybridAnomalyDetector
