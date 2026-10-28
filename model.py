"""
SmartVolt AI — Robust Hybrid Anomaly Detection Engine
Integrates:
  1. Two-Stage Deterministic Thresholds (Warning & Critical)
  2. Robust Statistical Outlier Detection (Rolling Median & Safe MAD with Debouncing)
  3. Machine Learning Anomaly Detection (Isolation Forest on Baseline Features)
  4. Explicit Data Acquisition Lifecycle (COLLECTING_DATA -> NORMAL / ANOMALY)
"""

import numpy as np
import pandas as pd
from collections import deque
from typing import List, Dict, Optional, Tuple, Union
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


class SmartVoltHybridAnomalyDetector:
    """
    Industrial-grade hybrid anomaly detector for IoT temperature telemetry.
    Combines deterministic engineering limits, robust statistics (MAD),
    and Scikit-Learn Isolation Forest with debouncing to eliminate false alarms.
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
        random_state: int = 42
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

        # Internal State Management
        # Rolling window of valid temperature measurements
        self.window = deque(maxlen=self.rolling_window_size)
        # Dedicated baseline buffer collected during baseline phase
        self.baseline_buffer: List[float] = []

        # State flags
        self.is_baseline_ready: bool = False
        self.model: Optional[IsolationForest] = None
        self.consecutive_anomalies: int = 0
        self.total_samples_ingested: int = 0
        self.anomaly_history: List[Dict] = []

    def reset_baseline(self):
        """Clears baseline buffer and resets model to COLLECTING_DATA status."""
        self.window.clear()
        self.baseline_buffer.clear()
        self.is_baseline_ready = False
        self.model = None
        self.consecutive_anomalies = 0
        self.total_samples_ingested = 0

    def set_thresholds(self, warning: float, critical: float):
        """Updates warning and critical thresholds safely."""
        self.warning_threshold = float(warning)
        self.critical_threshold = max(float(critical), float(warning))

    def evaluate_thresholds(self, temp: float) -> Tuple[str, bool, List[str]]:
        """
        Deterministic two-stage threshold evaluation.
        Returns:
            threshold_status: 'NORMAL' | 'WARNING' | 'CRITICAL'
            is_breached: bool
            reasons: list of trigger descriptions
        """
        reasons = []
        if temp >= self.critical_threshold:
            reasons.append(
                f"CRITICAL OVERHEAT: {temp:.2f}°C exceeds critical safety limit ({self.critical_threshold:.1f}°C)"
            )
            return "CRITICAL", True, reasons
        elif temp >= self.warning_threshold:
            reasons.append(
                f"THERMAL WARNING: {temp:.2f}°C exceeds warning threshold ({self.warning_threshold:.1f}°C)"
            )
            return "WARNING", True, reasons
        return "NORMAL", False, reasons

    def compute_statistical_metrics(
        self, 
        current_temp: float
    ) -> Dict[str, Optional[float]]:
        """
        Computes rolling median and Median Absolute Deviation (MAD).
        Safely handles zero-variance data (constant readings).
        Formula:
            MAD = median(|x_i - median(X)|)
            Modified Z-Score = 0.6745 * |current_temp - median| / max(MAD, min_mad_floor)
        """
        if len(self.window) < 5:
            return {
                "rolling_median": None,
                "rolling_mad": None,
                "upper_bound": None,
                "lower_bound": None,
                "modified_z_score": 0.0,
                "abs_diff": 0.0
            }

        arr = np.array(self.window, dtype=float)
        median_val = float(np.median(arr))
        abs_diffs = np.abs(arr - median_val)
        raw_mad = float(np.median(abs_diffs))

        # Safe MAD floor prevents division by zero when data is completely constant
        safe_mad = max(raw_mad, self.min_mad_floor)
        diff = abs(current_temp - median_val)

        # Standard Boris Iglewicz and David Hoaglin modified z-score
        mod_z = float(0.6745 * diff / safe_mad)

        # Compute dynamic statistical envelope for visualization: median ± 3.5 * safe_mad / 0.6745
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
        Fits Scikit-Learn IsolationForest on verified baseline samples.
        Features used:
          - Feature 0: Deviation from rolling median (scale-independent)
          - Feature 1: First difference (rate of change dT/dt)
        """
        if len(baseline_temps) < self.min_baseline_samples:
            return False, (
                f"Insufficient baseline samples: Received {len(baseline_temps)}/{self.min_baseline_samples}."
            )

        # Build feature matrix
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
            return True, f"Isolation Forest successfully trained on {len(baseline_temps)} verified samples."
        except Exception as ex:
            return False, f"Model training error: {str(ex)}"

    def predict(
        self,
        current_temp: Optional[float],
        previous_temp: Optional[float] = None
    ) -> Dict:
        """
        Full hybrid evaluation pipeline:
          1. Sensor validity check (None, -127°C, 85°C)
          2. Deterministic threshold evaluation (Warning vs Critical)
          3. Baseline accumulation check (COLLECTING_DATA vs READY)
          4. Statistical MAD outlier detection + Debounced Persistence
          5. Isolation Forest inference
        """
        res = {
            "current_temp": current_temp,
            "threshold_status": "NORMAL",
            "threshold_alert": False,
            "model_status": "COLLECTING_DATA",
            "statistical_status": "COLLECTING_DATA",
            "is_transient_anomaly": False,
            "is_persistent_anomaly": False,
            "is_anomaly": False,
            "consecutive_anomalies": self.consecutive_anomalies,
            "persistence_count": self.persistence_count,
            "rolling_median": None,
            "rolling_mad": None,
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
            "reasons": []
        }

        # 1. Hardware Sensor Fault / Out-of-Range Check
        is_sensor_fault = (
            current_temp is None
            or current_temp <= -100.0
            or current_temp >= 125.0
            or (current_temp is not None and abs(current_temp - 85.0) < 0.05)
        )
        if is_sensor_fault:
            res["model_status"] = "SENSOR_FAULT"
            res["statistical_status"] = "SENSOR_FAULT"
            res["is_anomaly"] = True
            res["anomaly_score_pct"] = 100.0
            if current_temp is not None and abs(current_temp - (-127.0)) < 0.1:
                res["reasons"].append("DS18B20 1-Wire bus disconnected / missing pull-up (-127.00°C)")
            elif current_temp is not None and abs(current_temp - 85.0) < 0.05:
                res["reasons"].append("DS18B20 power-on reset state (85.00°C conversion incomplete)")
            else:
                res["reasons"].append(f"Invalid reading outside physical sensor range: {current_temp}")
            return res

        # Compute rate of change
        rate = 0.0
        if previous_temp is not None and -50.0 <= previous_temp <= 125.0:
            rate = round(current_temp - previous_temp, 3)
        res["rate_of_change"] = rate

        # 2. Independent Deterministic Threshold Evaluation
        thresh_status, thresh_breached, thresh_reasons = self.evaluate_thresholds(current_temp)
        res["threshold_status"] = thresh_status
        res["threshold_alert"] = thresh_breached
        res["reasons"].extend(thresh_reasons)

        # 3. Baseline Data Ingestion
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
                    f"Collecting initial room temperature baseline ({samples_got}/{self.min_baseline_samples} samples)..."
                )
                return res
            else:
                # Train the model on the acquired baseline
                self.train_isolation_forest(self.baseline_buffer)

        # Update rolling statistical window
        self.window.append(current_temp)
        res["samples_collected"] = len(self.baseline_buffer)
        res["baseline_progress_pct"] = 100.0
        res["is_baseline_ready"] = True

        # 4. Statistical Anomaly Detection (Rolling Median & MAD)
        stat_metrics = self.compute_statistical_metrics(current_temp)
        res.update(stat_metrics)

        is_stat_outlier = False
        if stat_metrics["modified_z_score"] is not None:
            # Anomaly condition: Z-Score exceeds threshold AND absolute difference is above noise floor
            if (
                stat_metrics["modified_z_score"] >= self.mad_z_threshold 
                and stat_metrics["abs_diff"] >= self.min_absolute_diff
            ):
                is_stat_outlier = True

        res["is_transient_anomaly"] = is_stat_outlier

        # 5. Debounce / Persistence Filtering
        if is_stat_outlier:
            self.consecutive_anomalies += 1
        else:
            self.consecutive_anomalies = 0

        res["consecutive_anomalies"] = self.consecutive_anomalies
        res["is_persistent_anomaly"] = (self.consecutive_anomalies >= self.persistence_count)

        # 6. Isolation Forest Inference
        iforest_outlier = False
        if self.model is not None and stat_metrics["rolling_median"] is not None:
            dev = current_temp - stat_metrics["rolling_median"]
            sample_feats = np.array([[dev, rate]])
            pred = int(self.model.predict(sample_feats)[0])  # 1 for inlier, -1 for outlier
            dec = float(self.model.decision_function(sample_feats)[0])

            res["iforest_prediction"] = pred
            res["iforest_score"] = round(dec, 4)
            if pred == -1:
                iforest_outlier = True

            # Anomaly severity score mapped to 0-100%
            severity = (0.15 - dec) / 0.40
            res["anomaly_score_pct"] = round(max(0.0, min(100.0, float(severity * 100.0))), 1)

        # Determine Final Consolidated ML / Statistical Status
        # Persistent statistical outlier OR confirmed ML anomaly with persistence
        if res["is_persistent_anomaly"] or (iforest_outlier and self.consecutive_anomalies >= 1):
            res["model_status"] = "ANOMALY"
            res["statistical_status"] = "ANOMALY"
            res["is_anomaly"] = True
            res["reasons"].append(
                f"Statistical Outlier (Z-score: {res['modified_z_score']} > {self.mad_z_threshold}, "
                f"consecutive count: {self.consecutive_anomalies}/{self.persistence_count})"
            )
            if iforest_outlier:
                res["reasons"].append(f"Isolation Forest flagged pattern as out-of-distribution (score: {res['iforest_score']})")
        elif res["is_transient_anomaly"]:
            res["model_status"] = "NORMAL"
            res["statistical_status"] = "TRANSIENT_SPIKE"
            res["reasons"].append(
                f"Transient reading detected (Z-score {res['modified_z_score']}), debouncing 1/{self.persistence_count}"
            )
        else:
            res["model_status"] = "NORMAL"
            res["statistical_status"] = "NORMAL"

        return res

    # Backward compatibility method for existing calls
    def predict_single(
        self,
        current_temp: Optional[float],
        previous_temp: Optional[float] = None,
        threshold_c: Optional[float] = None
    ) -> Dict:
        """Compatibility wrapper that routes to predict()."""
        if threshold_c is not None:
            self.warning_threshold = float(threshold_c)
        return self.predict(current_temp=current_temp, previous_temp=previous_temp)

    def train_on_history(
        self,
        temperatures: List[float],
        contamination: Optional[float] = None
    ) -> Tuple[bool, str]:
        """Manually retrains the Isolation Forest on a provided sequence of temperatures."""
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

    @staticmethod
    def evaluate_labeled_dataset(
        y_true: List[int],
        y_pred: List[int]
    ) -> Dict[str, Union[float, int, str]]:
        """
        Calculates precision, recall, and F1-score on a labeled dataset.
        0: Normal, 1: Anomaly
        Includes explicit academic disclaimers.
        """
        if len(y_true) == 0 or len(y_true) != len(y_pred):
            return {"error": "Invalid or mismatched labeled dataset."}

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
            "disclaimer": (
                "NOTE: Evaluation metrics are computed strictly against the specified benchmark test dataset. "
                "Simulated benchmark performance demonstrates algorithmic mechanics under controlled test distributions "
                "and does not constitute certified validation on unmeasured real-world hardware environments."
            )
        }


# Maintain backward compatibility alias for existing imports
SmartVoltAnomalyDetector = SmartVoltHybridAnomalyDetector
