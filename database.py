"""
SmartVolt AI — SQLite Persistence Layer
Manages persistent storage for telemetry streams and anomaly audit events.
Ensures strict separation between real hardware data and simulated test runs.
"""

import os
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Union
import pandas as pd

DEFAULT_DB_PATH = os.path.join("data", "smartvolt.db")
_db_lock = threading.Lock()


def get_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Creates a thread-safe connection with foreign keys and WAL mode enabled."""
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def init_db(db_path: str = DEFAULT_DB_PATH):
    """Initializes the database schema and indexes if they do not exist."""
    with _db_lock:
        conn = get_connection(db_path)
        cursor = conn.cursor()

        # 1. Telemetry readings table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            temperature REAL,
            voltage REAL,
            source TEXT NOT NULL,          -- 'HARDWARE' or 'SIMULATED'
            device_status TEXT NOT NULL,   -- 'CONNECTED', 'DISCONNECTED', 'SIMULATING', 'ERROR'
            threshold_status TEXT NOT NULL,-- 'NORMAL', 'WARNING', 'CRITICAL'
            model_status TEXT NOT NULL,    -- 'COLLECTING_DATA', 'NORMAL', 'ANOMALY', 'SENSOR_FAULT'
            is_anomaly INTEGER NOT NULL,   -- 0 or 1
            modified_z_score REAL,
            rate_of_change REAL,
            scenario_id TEXT               -- NULL for hardware, scenario name for simulation
        );
        """)

        # 2. Anomaly events table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS anomaly_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            temperature REAL,
            voltage REAL,
            source TEXT NOT NULL,          -- 'HARDWARE' or 'SIMULATED'
            severity TEXT NOT NULL,        -- 'WARNING', 'CRITICAL'
            detection_method TEXT NOT NULL,-- 'RULE_BASED', 'STATISTICAL_MAD', 'ISOLATION_FOREST', 'SENSOR_FAULT'
            reason TEXT NOT NULL,
            suggested_action TEXT NOT NULL,
            scenario_id TEXT
        );
        """)

        # Indexes for efficient historical queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_ts ON telemetry_readings(timestamp);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_source ON telemetry_readings(source);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_anomaly ON telemetry_readings(is_anomaly);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON anomaly_events(timestamp);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_severity ON anomaly_events(severity);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_source ON anomaly_events(source);")

        conn.commit()
        conn.close()


def insert_reading(reading: Dict, db_path: str = DEFAULT_DB_PATH) -> int:
    """Inserts a single telemetry record into SQLite."""
    with _db_lock:
        conn = get_connection(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO telemetry_readings (
                timestamp, temperature, voltage, source, device_status,
                threshold_status, model_status, is_anomaly, modified_z_score,
                rate_of_change, scenario_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                reading.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                reading.get("temperature"),
                reading.get("voltage"),
                reading.get("source", "SIMULATED"),
                reading.get("device_status", "UNKNOWN"),
                reading.get("threshold_status", "NORMAL"),
                reading.get("model_status", "NORMAL"),
                1 if reading.get("is_anomaly") else 0,
                reading.get("modified_z_score", 0.0),
                reading.get("rate_of_change", 0.0),
                reading.get("scenario_id")
            ))
            row_id = cursor.lastrowid
            conn.commit()
            return row_id
        finally:
            conn.close()


def insert_anomaly_event(
    event: Dict, 
    cooldown_seconds: int = 5, 
    db_path: str = DEFAULT_DB_PATH
) -> bool:
    """
    Inserts an anomaly event with debouncing cooldown.
    Suppresses duplicate logging if a matching event occurred within cooldown_seconds.
    """
    ts_str = event.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    try:
        current_dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        current_dt = datetime.now()

    with _db_lock:
        conn = get_connection(db_path)
        try:
            cursor = conn.cursor()

            # Check for duplicate event within cooldown period
            cutoff = (current_dt - timedelta(seconds=cooldown_seconds)).strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute("""
            SELECT id FROM anomaly_events 
            WHERE source = ? AND severity = ? AND timestamp >= ? 
            ORDER BY id DESC LIMIT 1
            """, (event.get("source", "SIMULATED"), event.get("severity", "WARNING"), cutoff))

            existing = cursor.fetchone()
            if existing:
                return False  # Cooldown active, throttled duplicate

            cursor.execute("""
            INSERT INTO anomaly_events (
                timestamp, temperature, voltage, source, severity,
                detection_method, reason, suggested_action, scenario_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                ts_str,
                event.get("temperature"),
                event.get("voltage"),
                event.get("source", "SIMULATED"),
                event.get("severity", "WARNING"),
                event.get("detection_method", "HYBRID"),
                event.get("reason", "Outlier detected"),
                event.get("suggested_action", "Inspect physical device"),
                event.get("scenario_id")
            ))
            conn.commit()
            return True
        finally:
            conn.close()


def get_recent_readings(
    limit: int = 100, 
    source: Optional[str] = None, 
    db_path: str = DEFAULT_DB_PATH
) -> pd.DataFrame:
    """Retrieves the most recent N telemetry records."""
    conn = get_connection(db_path)
    try:
        query = "SELECT * FROM telemetry_readings"
        params = []
        if source:
            query += " WHERE source = ?"
            params.append(source)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        df = pd.read_sql_query(query, conn, params=params)
        if not df.empty:
            df = df.sort_values(by="id", ascending=True).reset_index(drop=True)
        return df
    finally:
        conn.close()


def get_filtered_readings(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    source: Optional[str] = None,
    anomalies_only: bool = False,
    limit: int = 1000,
    db_path: str = DEFAULT_DB_PATH
) -> pd.DataFrame:
    """Retrieves telemetry readings matching historical filters."""
    conn = get_connection(db_path)
    try:
        conditions = []
        params = []

        if start_date:
            conditions.append("timestamp >= ?")
            params.append(f"{start_date} 00:00:00")
        if end_date:
            conditions.append("timestamp <= ?")
            params.append(f"{end_date} 23:59:59")
        if source and source != "ALL":
            conditions.append("source = ?")
            params.append(source)
        if anomalies_only:
            conditions.append("is_anomaly = 1")

        query = "SELECT * FROM telemetry_readings"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        df = pd.read_sql_query(query, conn, params=params)
        if not df.empty:
            df = df.sort_values(by="id", ascending=True).reset_index(drop=True)
        return df
    finally:
        conn.close()


def get_anomaly_events(
    limit: int = 100, 
    source: Optional[str] = None,
    severity: Optional[str] = None,
    db_path: str = DEFAULT_DB_PATH
) -> pd.DataFrame:
    """Retrieves logged anomaly events."""
    conn = get_connection(db_path)
    try:
        conditions = []
        params = []

        if source and source != "ALL":
            conditions.append("source = ?")
            params.append(source)
        if severity and severity != "ALL":
            conditions.append("severity = ?")
            params.append(severity)

        query = "SELECT * FROM anomaly_events"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        df = pd.read_sql_query(query, conn, params=params)
        return df
    finally:
        conn.close()


def get_database_stats(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Union[int, float, str]]:
    """Returns database size, record counts, and health statistics."""
    if not os.path.exists(db_path):
        init_db(db_path)

    file_size_kb = round(os.path.getsize(db_path) / 1024.0, 1)
    conn = get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM telemetry_readings;")
        total_readings = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM telemetry_readings WHERE source = 'HARDWARE';")
        hardware_readings = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM telemetry_readings WHERE source = 'SIMULATED';")
        simulated_readings = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM anomaly_events;")
        total_events = cursor.fetchone()[0]

        cursor.execute("SELECT timestamp FROM telemetry_readings ORDER BY id DESC LIMIT 1;")
        last_row = cursor.fetchone()
        last_timestamp = last_row[0] if last_row else "None"

        return {
            "db_path": db_path,
            "file_size_kb": file_size_kb,
            "total_readings": total_readings,
            "hardware_readings": hardware_readings,
            "simulated_readings": simulated_readings,
            "total_events": total_events,
            "last_timestamp": last_timestamp
        }
    finally:
        conn.close()


def clear_database(db_path: str = DEFAULT_DB_PATH):
    """Safely clears stored readings and events."""
    with _db_lock:
        conn = get_connection(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM telemetry_readings;")
            cursor.execute("DELETE FROM anomaly_events;")
            cursor.execute("VACUUM;")
            conn.commit()
        finally:
            conn.close()


# Initialize database automatically on module import
init_db()

