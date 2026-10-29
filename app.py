"""
SmartVolt AI — Industrial IoT Telemetry & Anomaly Detection System
Upgraded Multi-Page Industrial Dashboard with SQLite Persistence,
Hybrid Machine Learning, Simulation Lab, Model Evaluation & PDF Audits.
"""

import os
import sys
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Local modules
import database
import model
import reports
import serial_reader
import thingspeak_client
from serial_reader import SmartVoltSerialManager, SmartVoltSimulationEngine, list_available_ports
from model import SmartVoltHybridAnomalyDetector
from thingspeak_client import ThingSpeakBackgroundWorker, ThingSpeakClient, mask_key, load_credentials_from_environment


# -----------------------------------------------------------------------------
# 1. Page Configuration & Custom CSS Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartVolt AI — Industrial IoT Anomaly Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Industrial SCADA / High-Contrast Theme
st.markdown("""
<style>
    /* Metric Card styling */
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px 16px;
        color: #f8fafc;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        margin-bottom: 10px;
    }
    .metric-title {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 3px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        line-height: 1.1;
        color: #38bdf8;
    }
    .metric-subtitle {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 3px;
    }
    .metric-badge-sim {
        background-color: #f59e0b;
        color: #0f172a;
        font-size: 0.65rem;
        font-weight: 800;
        padding: 2px 6px;
        border-radius: 4px;
        letter-spacing: 0.05em;
        display: inline-block;
        margin-bottom: 3px;
    }
    .metric-badge-real {
        background-color: #10b981;
        color: #ffffff;
        font-size: 0.65rem;
        font-weight: 800;
        padding: 2px 6px;
        border-radius: 4px;
        letter-spacing: 0.05em;
        display: inline-block;
        margin-bottom: 3px;
    }

    /* Status Badges */
    .status-normal {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid #10b981;
        padding: 3px 8px;
        border-radius: 5px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    .status-collecting {
        background-color: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid #38bdf8;
        padding: 3px 8px;
        border-radius: 5px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    .status-warning {
        background-color: rgba(245, 158, 11, 0.2);
        color: #f59e0b;
        border: 1px solid #f59e0b;
        padding: 3px 8px;
        border-radius: 5px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    .status-critical {
        background-color: rgba(239, 68, 68, 0.25);
        color: #ef4444;
        border: 1px solid #ef4444;
        padding: 3px 8px;
        border-radius: 5px;
        font-weight: 700;
        font-size: 0.8rem;
        display: inline-block;
    }
    .safety-banner {
        background-color: #451a03;
        border: 1px solid #b45309;
        border-radius: 6px;
        padding: 10px 14px;
        color: #fef3c7;
        margin-bottom: 12px;
        font-size: 0.85rem;
    }
    .sim-banner {
        background-color: #172554;
        border: 1px solid #3b82f6;
        border-radius: 6px;
        padding: 8px 14px;
        color: #dbeafe;
        margin-bottom: 12px;
        font-size: 0.85rem;
    }
    .terminal-box {
        background-color: #020617;
        border: 1px solid #1e293b;
        border-radius: 6px;
        padding: 8px 12px;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 0.78rem;
        color: #22c55e;
        max-height: 140px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 2. State & Resource Initialization
# -----------------------------------------------------------------------------
@st.cache_resource
def get_serial_manager() -> SmartVoltSerialManager:
    return SmartVoltSerialManager(max_history=2000)

@st.cache_resource
def get_simulation_engine() -> SmartVoltSimulationEngine:
    return SmartVoltSimulationEngine(max_history=2000)

@st.cache_resource
def get_anomaly_detector() -> SmartVoltHybridAnomalyDetector:
    return SmartVoltHybridAnomalyDetector(
        warning_threshold=40.0,
        critical_threshold=50.0,
        rolling_window_size=30,
        min_baseline_samples=30,
        mad_z_threshold=3.5,
        min_mad_floor=0.15,
        min_absolute_diff=0.40,
        persistence_count=2,
        contamination=0.05,
        n_estimators=100,
        random_state=42
    )

@st.cache_resource
def get_thingspeak_worker() -> ThingSpeakBackgroundWorker:
    return ThingSpeakBackgroundWorker()

# Initialize Session State
if "telemetry_buffer" not in st.session_state:
    st.session_state.telemetry_buffer = []
if "active_mode" not in st.session_state:
    st.session_state.active_mode = "Simulation Lab (Demo)"
if "navigation_page" not in st.session_state:
    st.session_state.navigation_page = "📊 Live Telemetry & Overview"

serial_mgr = get_serial_manager()
sim_engine = get_simulation_engine()
detector = get_anomaly_detector()
ts_worker = get_thingspeak_worker()


# -----------------------------------------------------------------------------
# 3. Sidebar: Navigation & Primary Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("⚡ SmartVolt AI")
    st.caption("Industrial IoT Telemetry & Hybrid Anomaly Detection")
    st.markdown("---")

    # Primary Navigation Selector
    nav_options = [
        "📊 Live Telemetry & Overview",
        "🧪 Simulation & Test Lab",
        "📈 Historical Trends & Analytics",
        "🤖 Model Evaluation & Comparison",
        "📑 Reports & Audit Export",
        "☁️ ThingSpeak Cloud Telemetry",
        "⚙️ Diagnostics & Device Health"
    ]
    cur_nav = st.session_state.navigation_page
    nav_idx = nav_options.index(cur_nav) if cur_nav in nav_options else 0

    page = st.radio(
        "Navigation Menu",
        options=nav_options,
        index=nav_idx
    )
    st.session_state.navigation_page = page

    st.markdown("---")
    st.subheader("⚙️ Operating Mode")
    mode = st.radio(
        "Data Source Selector",
        options=["Real Hardware (USB Serial)", "Simulation Lab (Demo)"],
        index=0 if st.session_state.active_mode == "Real Hardware (USB Serial)" else 1,
        help="Real Hardware connects to ESP8266 COM port; Simulation Lab runs offline test scenarios."
    )
    st.session_state.active_mode = mode

    # Real Hardware Serial Connection
    if mode == "Real Hardware (USB Serial)":
        st.subheader("🔌 Serial COM Port")
        ports = list_available_ports()
        port_devs = [p["device"] for p in ports]
        port_labels = {p["device"]: f"{p['device']} ({p['description'][:24]})" for p in ports}

        col_p1, col_p2 = st.columns([3, 1])
        with col_p1:
            if port_devs:
                selected_port = st.selectbox("Device Port", options=port_devs, format_func=lambda x: port_labels.get(x, x), index=0)
            else:
                st.warning("No COM ports found.")
                selected_port = None
        with col_p2:
            st.write("")
            st.write("")
            if st.button("🔄", help="Scan COM ports"):
                st.rerun()

        baud_rate = st.selectbox("Baud Rate", options=[115200, 9600, 57600], index=0)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            if st.button("🟢 Connect", disabled=serial_mgr.is_connected or not selected_port, use_container_width=True):
                ok, msg = serial_mgr.connect(selected_port, baud_rate)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        with col_c2:
            if st.button("🔴 Disconnect", disabled=not serial_mgr.is_connected, use_container_width=True):
                serial_mgr.disconnect()
                st.rerun()

        if serial_mgr.is_connected:
            st.success(f"Connected: {serial_mgr.port_name}")
        else:
            st.info("Status: Disconnected")

    st.markdown("---")
    st.subheader("🎯 Safety Limits")
    warn_th = st.slider("Warning Limit (°C)", 30.0, 60.0, detector.warning_threshold, 0.5)
    crit_th = st.slider("Critical Limit (°C)", max(35.0, warn_th + 1.0), 80.0, max(detector.critical_threshold, warn_th + 5.0), 0.5)
    detector.set_thresholds(warn_th, crit_th)

    if st.button("🔄 Recalibrate Baseline", use_container_width=True, help="Clears stored baseline and acquires fresh room temperature distribution."):
        detector.reset_baseline()
        st.session_state.telemetry_buffer = []
        st.info("Baseline reset. Now acquiring 30 fresh baseline samples.")
        st.rerun()

    st.markdown("---")
    st.subheader("☁️ ThingSpeak Cloud")
    ts_stat = ts_worker.get_status()
    is_cfg = ts_stat["credentials"]["is_configured"]

    if is_cfg:
        cloud_active = st.checkbox(
            "Enable Cloud Uplink",
            value=ts_worker.is_enabled,
            help="Uploads live readings over HTTPS to ThingSpeak Field 1 (>=15s interval)."
        )
        ts_worker.set_enabled(cloud_active)
        if cloud_active:
            st.markdown('<span class="status-normal">🟢 UPLINK ACTIVE</span>', unsafe_allow_html=True)
            if ts_stat["seconds_since_last_upload"] is not None:
                st.caption(f"Last post: {ts_stat['seconds_since_last_upload']}s ago (ID: {ts_stat['last_successful_entry_id'] or '--'})")
        else:
            st.markdown('<span class="status-warning">⚪ UPLINK PAUSED</span>', unsafe_allow_html=True)
        st.caption(f"Channel: `{ts_stat['credentials']['channel_id']}` | Key: `{ts_stat['credentials']['masked_write_key']}`")
    else:
        st.markdown('<span class="status-warning">⚠️ NOT CONFIGURED</span>', unsafe_allow_html=True)
        st.caption("Configure credentials in the Cloud Telemetry page or secrets.toml.")


# -----------------------------------------------------------------------------
# 4. Ingestion & Persistence Engine
# -----------------------------------------------------------------------------
def ingest_telemetry_stream():
    """Reads packets, evaluates hybrid detection, and persists to SQLite."""
    new_raw = []

    if st.session_state.active_mode == "Real Hardware (USB Serial)":
        if serial_mgr.is_connected:
            items = serial_mgr.get_history(limit=50)
            existing_ts = set(r["dt"] for r in st.session_state.telemetry_buffer[-100:])
            for item in items:
                if item["dt"] not in existing_ts:
                    new_raw.append(item)
    else:
        if sim_engine.is_running:
            item = sim_engine.tick()
            new_raw.append(item)

    for rec in new_raw:
        prev_t = st.session_state.telemetry_buffer[-1]["temperature"] if st.session_state.telemetry_buffer else None
        eval_res = detector.predict(rec["temperature"], previous_temp=prev_t)

        record = {
            "timestamp": rec["timestamp"],
            "dt": rec["dt"],
            "temperature": rec["temperature"],
            "voltage": rec["voltage"],
            "source": rec["source"],
            "device_status": rec.get("device_status", "CONNECTED" if rec["source"] == "HARDWARE" else "SIMULATING"),
            "is_valid": rec["is_valid"],
            "raw": rec["raw"],
            "message": rec["message"],
            "threshold_status": eval_res["threshold_status"],
            "threshold_alert": eval_res["threshold_alert"],
            "model_status": eval_res["model_status"],
            "alert_severity": eval_res["alert_severity"],
            "is_anomaly": eval_res["is_anomaly"],
            "consecutive_anomalies": eval_res["consecutive_anomalies"],
            "rolling_median": eval_res["rolling_median"],
            "rolling_mad": eval_res["rolling_mad"],
            "upper_bound": eval_res["upper_bound"],
            "lower_bound": eval_res["lower_bound"],
            "modified_z_score": eval_res["modified_z_score"],
            "rate_of_change": eval_res["rate_of_change"],
            "reasons": eval_res["reasons"],
            "suggested_action": eval_res["suggested_action"],
            "scenario_id": rec.get("scenario_id")
        }

        # 1. Update In-Memory Stream Buffer
        st.session_state.telemetry_buffer.append(record)

        # 2. Persist to SQLite Database
        database.insert_reading(record)

        # 3. Post to Non-blocking ThingSpeak Cloud Worker
        if rec.get("is_valid") and rec.get("temperature") is not None:
            ts_worker.post_reading(
                temp_c=rec["temperature"],
                voltage=rec.get("voltage"),
                source=rec.get("source", "HARDWARE"),
                device_status=record.get("device_status", "CONNECTED")
            )

        # 4. Log Event if Anomaly or Threshold Alert occurred
        if eval_res["is_anomaly"] or eval_res["threshold_alert"]:
            event = {
                "timestamp": record["timestamp"],
                "temperature": record["temperature"],
                "voltage": record["voltage"],
                "source": record["source"],
                "severity": eval_res["alert_severity"],
                "detection_method": " | ".join(eval_res["detection_methods"]) if eval_res["detection_methods"] else "HYBRID",
                "reason": " | ".join(eval_res["reasons"]) if eval_res["reasons"] else "Outlier",
                "suggested_action": eval_res["suggested_action"],
                "scenario_id": record["scenario_id"]
            }
            database.insert_anomaly_event(event, cooldown_seconds=5)

    if len(st.session_state.telemetry_buffer) > 1000:
        st.session_state.telemetry_buffer = st.session_state.telemetry_buffer[-1000:]


# Always ingest fresh packets on update
ingest_telemetry_stream()


# -----------------------------------------------------------------------------
# 5. Global Permanent Header & Data Source Indicator
# -----------------------------------------------------------------------------
st.title("⚡ SmartVolt AI — IoT Supervisory System")

# Always-Visible Data Source & System State Indicator Bar
col_h1, col_h2, col_h3, col_h4 = st.columns([2, 1, 1, 1])
with col_h1:
    if st.session_state.active_mode == "Real Hardware (USB Serial)":
        if serial_mgr.is_connected:
            if serial_mgr.is_stale():
                st.markdown('<span class="status-warning">⚠️ LINK STALE: Real Hardware (No readings &gt; 4s)</span>', unsafe_allow_html=True)
            else:
                st.markdown(f'<span class="status-normal">🟢 REAL HARDWARE ACTIVE: Port {serial_mgr.port_name} (115200 baud)</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-critical">🔴 HARDWARE DISCONNECTED: Plug in ESP8266 and Connect</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'<span class="status-warning">🧪 SIMULATION LAB ACTIVE: Scenario [{sim_engine.scenario}] — ALL DATA SIMULATED</span>', unsafe_allow_html=True)

with col_h2:
    db_stat = database.get_database_stats()
    st.caption(f"💾 **SQLite Storage:** {db_stat['total_readings']} records ({db_stat['file_size_kb']} KB)")

with col_h3:
    ts_h = ts_worker.get_status()
    if ts_h["credentials"]["is_configured"]:
        if ts_worker.is_enabled:
            st.caption(f"☁️ **ThingSpeak:** Active (Ch {ts_h['credentials']['channel_id']})")
        else:
            st.caption(f"☁️ **ThingSpeak:** Paused (Ch {ts_h['credentials']['channel_id']})")
    else:
        st.caption("☁️ **ThingSpeak:** Unconfigured")

with col_h4:
    st.caption(f"🕒 **Last Synchronized:** {datetime.now().strftime('%H:%M:%S')}")

st.markdown("---")


# =============================================================================
# PAGE 1: 📊 LIVE TELEMETRY & OVERVIEW
# =============================================================================
if st.session_state.navigation_page == "📊 Live Telemetry & Overview":
    @st.fragment(run_every="2s")
    def render_live_overview():
        ingest_telemetry_stream()
        records = st.session_state.telemetry_buffer
        latest = records[-1] if records else None

        # 5 Overview Cards
        c1, c2, c3, c4, c5 = st.columns(5)

        with c1:
            if latest and latest["temperature"] is not None and latest["is_valid"]:
                badge = '<span class="metric-badge-sim">SIMULATED</span>' if latest["source"] == "SIMULATED" else '<span class="metric-badge-real">HARDWARE DS18B20</span>'
                st.markdown(f"""
                <div class="metric-card">
                    {badge}
                    <div class="metric-title">Temperature</div>
                    <div class="metric-value">{latest['temperature']:.2f} °C</div>
                    <div class="metric-subtitle">Rate: {latest['rate_of_change']:+.2f} °C/s</div>
                </div>
                """, unsafe_allow_html=True)
            elif latest and not latest["is_valid"]:
                st.markdown(f"""
                <div class="metric-card" style="border-color: #ef4444;">
                    <div class="metric-title" style="color: #ef4444;">Sensor Fault</div>
                    <div class="metric-value" style="color: #ef4444;">{latest['temperature']} °C</div>
                    <div class="metric-subtitle">{latest['message']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-card"><div class="metric-title">Temperature</div><div class="metric-value">--.- °C</div><div class="metric-subtitle">Awaiting stream...</div></div>', unsafe_allow_html=True)

        with c2:
            if latest:
                ts = latest.get("threshold_status", "NORMAL")
                badge_t = '<span class="status-normal">🟢 NORMAL</span>' if ts == "NORMAL" else ('<span class="status-warning">⚠️ WARNING</span>' if ts == "WARNING" else '<span class="status-critical">🚨 CRITICAL</span>')
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Threshold Limits</div>
                    <div style="margin: 4px 0;">{badge_t}</div>
                    <div class="metric-subtitle">Warn: {detector.warning_threshold:.1f}°C | Crit: {detector.critical_threshold:.1f}°C</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-card"><div class="metric-title">Threshold Limits</div><div class="metric-value">--</div></div>', unsafe_allow_html=True)

        with c3:
            if latest:
                ms = latest.get("model_status", "COLLECTING_DATA")
                if ms == "COLLECTING_DATA":
                    got = len(detector.baseline_buffer)
                    req = detector.min_baseline_samples
                    badge_m = f'<span class="status-collecting">⏳ BASELINE ({got}/{req})</span>'
                    sub_m = "Gathering room baseline"
                elif ms == "NORMAL":
                    badge_m = '<span class="status-normal">🟢 INLIER NORMAL</span>'
                    sub_m = f"MAD Z: {latest['modified_z_score']:.1f} &lt; {detector.mad_z_threshold}"
                elif ms == "ANOMALY":
                    badge_m = '<span class="status-critical">⚠️ ANOMALY ACTIVE</span>'
                    sub_m = f"Consecutive: {latest['consecutive_anomalies']}/{detector.persistence_count}"
                else:
                    badge_m = '<span class="status-critical">❌ FAULT</span>'
                    sub_m = "Probe disconnected"

                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Hybrid ML State</div>
                    <div style="margin: 4px 0;">{badge_m}</div>
                    <div class="metric-subtitle">{sub_m}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="metric-card"><div class="metric-title">Hybrid ML State</div><div class="metric-value">--</div></div>', unsafe_allow_html=True)

        with c4:
            # Voltage Status Panel
            if st.session_state.active_mode == "Simulation Lab (Demo)":
                volt_v = latest["voltage"] if latest and latest["voltage"] is not None else 12.08
                st.markdown(f"""
                <div class="metric-card">
                    <span class="metric-badge-sim">SIMULATED BUS</span>
                    <div class="metric-title">Voltage Status</div>
                    <div class="metric-value" style="color: #a78bfa;">{volt_v:.2f} V</div>
                    <div class="metric-subtitle">12V DC System Model [SIMULATED]</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card" style="border-color: #64748b;">
                    <div class="metric-title">Hardware Voltage (A0)</div>
                    <div class="metric-value" style="color: #94a3b8; font-size: 1.35rem;">UNAVAILABLE 🔒</div>
                    <div class="metric-subtitle">ADC Interlock: Hardware unverified</div>
                </div>
                """, unsafe_allow_html=True)

        with c5:
            # Anomaly Count & Last Update
            db_stats_now = database.get_database_stats()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Audit Count & Time</div>
                <div class="metric-value" style="color: #f59e0b;">{db_stats_now['total_events']} Events</div>
                <div class="metric-subtitle">Last: {latest['timestamp'][-8:] if latest else '--:--:--'}</div>
            </div>
            """, unsafe_allow_html=True)

        # Real-time Chart Section with Time-Range Selector
        st.subheader("📈 Live Telemetry Stream with Rolling Statistical Envelope")
        col_ctrl1, col_ctrl2 = st.columns([3, 1])
        with col_ctrl2:
            time_range = st.selectbox("View Range", ["Last 50 Samples", "Last 100 Samples", "Last 200 Samples", "All Buffer"], index=1)

        range_map = {"Last 50 Samples": 50, "Last 100 Samples": 100, "Last 200 Samples": 200, "All Buffer": 1000}
        n_samples = range_map[time_range]

        if len(records) > 1:
            df = pd.DataFrame(records).tail(n_samples)
            fig = go.Figure()

            # Dynamic Statistical Envelope (MAD Upper / Lower Bounds)
            valid_bounds = df.dropna(subset=["upper_bound", "lower_bound"])
            if not valid_bounds.empty:
                fig.add_trace(go.Scatter(
                    x=valid_bounds["timestamp"], y=valid_bounds["upper_bound"],
                    mode="lines", name="MAD Upper Bound", line=dict(color="rgba(167, 139, 250, 0.35)", width=1, dash="dot"), showlegend=True
                ))
                fig.add_trace(go.Scatter(
                    x=valid_bounds["timestamp"], y=valid_bounds["lower_bound"],
                    mode="lines", name="MAD Lower Bound", line=dict(color="rgba(167, 139, 250, 0.35)", width=1, dash="dot"),
                    fill="tonexty", fillcolor="rgba(167, 139, 250, 0.08)", showlegend=True
                ))

            # Rolling Median Baseline
            valid_med = df.dropna(subset=["rolling_median"])
            if not valid_med.empty:
                fig.add_trace(go.Scatter(
                    x=valid_med["timestamp"], y=valid_med["rolling_median"],
                    mode="lines", name="Rolling Median (MAD)", line=dict(color="#a78bfa", width=2, dash="dash"),
                    hovertemplate="Baseline: %{y:.2f} °C<extra></extra>"
                ))

            # Measured Temperature Line
            fig.add_trace(go.Scatter(
                x=df["timestamp"], y=df["temperature"],
                mode="lines", name="Measured Temperature (°C)", line=dict(color="#38bdf8", width=2.5),
                hovertemplate="Time: %{x}<br>Temp: %{y:.2f} °C<extra></extra>"
            ))

            # Highlight Anomaly Points
            anom_df = df[df["is_anomaly"] == True]
            if not anom_df.empty:
                fig.add_trace(go.Scatter(
                    x=anom_df["timestamp"], y=anom_df["temperature"],
                    mode="markers", name="Detected Anomaly",
                    marker=dict(color="#ef4444", size=10, symbol="diamond", line=dict(color="#ffffff", width=1.5)),
                    hovertemplate="⚠️ ANOMALY<br>Temp: %{y:.2f} °C<br>Severity: %{customdata[0]}<br>Reason: %{customdata[1]}<extra></extra>",
                    customdata=np.column_stack([anom_df["alert_severity"], anom_df["suggested_action"]])
                ))

            # Threshold Reference Lines
            fig.add_hline(y=detector.warning_threshold, line_dash="dash", line_color="#f59e0b", annotation_text=f"Warning ({detector.warning_threshold:.1f}°C)")
            fig.add_hline(y=detector.critical_threshold, line_dash="dash", line_color="#ef4444", annotation_text=f"Critical ({detector.critical_threshold:.1f}°C)")

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(15, 23, 42, 0.6)",
                plot_bgcolor="rgba(15, 23, 42, 0.8)",
                margin=dict(l=40, r=40, t=30, b=40),
                height=380,
                xaxis=dict(title="Timestamp", showgrid=True, gridcolor="#334155"),
                yaxis=dict(title="Temperature (°C)", showgrid=True, gridcolor="#334155"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Accumulating telemetry stream data points...")

        # Diagnostic Terminal Feed
        col_t1, col_t2 = st.columns([2, 1])
        with col_t1:
            st.markdown("**Live Raw Telemetry Feed**")
            recent_logs = records[-6:] if records else []
            term_content = ""
            for r in reversed(recent_logs):
                tag = f"[{r['source']}]"
                alert = f" [ALERT: {r['alert_severity']}]" if r["is_anomaly"] else ""
                term_content += f"{r['timestamp']} {tag} {r['raw']}{alert}\n"
            if not term_content:
                term_content = "Awaiting telemetry stream..."
            st.markdown(f'<div class="terminal-box">{term_content.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)

        with col_t2:
            st.markdown("**Active Alert Status**")
            if latest and latest["is_anomaly"]:
                st.error(f"**Severity:** {latest['alert_severity']}\n\n**Reasons:** {', '.join(latest['reasons'])}\n\n**Action:** {latest['suggested_action']}")
            elif latest and latest["model_status"] == "COLLECTING_DATA":
                st.info("System is acquiring nominal baseline samples. Normal operation expected.")
            else:
                st.success("All parameters operating within nominal statistical distribution.")

    render_live_overview()


# =============================================================================
# PAGE 2: 🧪 SIMULATION & TEST LAB
# =============================================================================
elif st.session_state.navigation_page == "🧪 Simulation & Test Lab":
    st.header("🧪 IoT Simulation & Anomaly Testing Laboratory")
    st.markdown(
        "Generate controlled, reproducible operational scenarios to evaluate detection performance "
        "before deploying to physical high-voltage or uncalibrated hardware."
    )

    st.markdown("""
    <div class="sim-banner">
        <strong>⚠️ ACADEMIC INTEGRITY & SIMULATION NOTICE:</strong><br>
        All telemetry generated in this laboratory is purely synthetic and explicitly tagged <code>SIMULATED</code>.
        Simulated voltage curves represent a 12V DC system model and must <b>never be represented as physical hardware measurements</b>.
        Hardware voltage monitoring on ESP8266 pin A0 remains locked until benchtop calibration is completed.
    </div>
    """, unsafe_allow_html=True)

    col_l1, col_l2 = st.columns([1, 2])
    with col_l1:
        st.subheader("1. Scenario Selection & Controls")
        chosen_scen = st.selectbox(
            "Select Scenario",
            options=list(SmartVoltSimulationEngine.SCENARIO_INFO.keys()),
            format_func=lambda k: f"{SmartVoltSimulationEngine.SCENARIO_INFO[k]['title']}",
            index=list(SmartVoltSimulationEngine.SCENARIO_INFO.keys()).index(sim_engine.scenario)
        )
        sim_engine.set_scenario(chosen_scen)

        scen_info = SmartVoltSimulationEngine.SCENARIO_INFO[chosen_scen]
        st.info(f"**Description:** {scen_info['description']}")

        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            if st.button("▶ Start", use_container_width=True, help="Run simulation stream"):
                sim_engine.start()
                st.rerun()
        with col_b2:
            if st.button("⏸ Pause", use_container_width=True, help="Pause simulation"):
                sim_engine.stop()
                st.rerun()
        with col_b3:
            if st.button("🔄 Reset", use_container_width=True, help="Reset scenario to baseline"):
                sim_engine.reset()
                detector.reset_baseline()
                st.rerun()

    with col_l2:
        st.subheader("2. Scenario Analysis: Expected vs. Actual Detection")
        st.markdown(f"""
        - **Active Scenario:** `{scen_info['title']}`
        - **Expected System Response:** {scen_info['expected_behavior']}
        - **Target Severity Level:** `{scen_info['expected_severity']}`
        """)

        records = st.session_state.telemetry_buffer
        latest = records[-1] if records else None
        if latest:
            st.markdown(f"""
            - **Current Measured Value:** `{latest['temperature']} °C` | Voltage: `{latest['voltage']} V [SIMULATED]`
            - **Actual System Severity:** `{latest['alert_severity']}`
            - **Actual Model Status:** `{latest['model_status']}`
            - **Active Detection Method:** `{', '.join(latest['reasons']) if latest['reasons'] else 'None (Inlier)'}`
            """)
            if latest['alert_severity'] == scen_info['expected_severity'] or (scen_info['expected_severity'] == 'NORMAL' and latest['alert_severity'] == 'NORMAL'):
                st.success("✅ Actual detector response aligns with expected scenario behavior.")
            else:
                st.warning(f"⚠️ Transitional state: Current response [{latest['alert_severity']}], Expected [{scen_info['expected_severity']}].")
        else:
            st.info("Simulation engine initializing...")

    st.markdown("---")
    st.subheader("3. Simulated Multi-Variable Telemetry Trajectory [SIMULATED DATA]")
    if len(st.session_state.telemetry_buffer) > 1:
        df_sim = pd.DataFrame(st.session_state.telemetry_buffer).tail(80)

        fig_sim = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["Simulated Temperature (°C)", "Simulated 12V DC Bus Voltage (V) [SIMULATED]"])
        fig_sim.add_trace(go.Scatter(x=df_sim["timestamp"], y=df_sim["temperature"], line=dict(color="#38bdf8", width=2), name="Simulated Temp"), row=1, col=1)
        fig_sim.add_trace(go.Scatter(x=df_sim["timestamp"], y=df_sim["voltage"], line=dict(color="#a78bfa", width=2), name="Simulated Voltage"), row=2, col=1)

        fig_sim.add_hline(y=detector.warning_threshold, line_dash="dash", line_color="#f59e0b", row=1, col=1)
        fig_sim.add_hline(y=10.0, line_dash="dash", line_color="#ef4444", annotation_text="Undervoltage (10V)", row=2, col=1)
        fig_sim.add_hline(y=15.0, line_dash="dash", line_color="#f59e0b", annotation_text="Overvoltage (15V)", row=2, col=1)

        fig_sim.update_layout(template="plotly_dark", height=420, paper_bgcolor="rgba(15, 23, 42, 0.6)", plot_bgcolor="rgba(15, 23, 42, 0.8)")
        st.plotly_chart(fig_sim, use_container_width=True)


# =============================================================================
# PAGE 3: 📈 HISTORICAL TRENDS & ANALYTICS
# =============================================================================
elif st.session_state.navigation_page == "📈 Historical Trends & Analytics":
    st.header("📈 Historical Telemetry & Event Analytics")
    st.caption("Query persistent SQLite records, filter by date and data source, and analyze multi-hour operational trends.")

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        f_start = st.date_input("Start Date", value=datetime.now().date() - timedelta(days=1))
    with col_f2:
        f_end = st.date_input("End Date", value=datetime.now().date())
    with col_f3:
        f_source = st.selectbox("Data Source", ["ALL", "HARDWARE", "SIMULATED"], index=0)
    with col_f4:
        f_anom_only = st.checkbox("Anomalies Only", value=False)

    df_hist = database.get_filtered_readings(
        start_date=f_start.strftime("%Y-%m-%d"),
        end_date=f_end.strftime("%Y-%m-%d"),
        source=f_source,
        anomalies_only=f_anom_only,
        limit=1000
    )

    if not df_hist.empty:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("Queried Samples", len(df_hist))
        with col_m2:
            st.metric("Anomalies Found", int(df_hist["is_anomaly"].sum()))
        with col_m3:
            avg_t = df_hist["temperature"].dropna().mean()
            st.metric("Average Temperature", f"{avg_t:.2f} °C" if not np.isnan(avg_t) else "--")
        with col_m4:
            max_t = df_hist["temperature"].dropna().max()
            st.metric("Peak Temperature", f"{max_t:.2f} °C" if not np.isnan(max_t) else "--")

        fig_h = go.Figure()
        fig_h.add_trace(go.Scatter(
            x=df_hist["timestamp"], y=df_hist["temperature"],
            mode="lines", name="Temperature (°C)", line=dict(color="#38bdf8", width=2)
        ))
        anom_h = df_hist[df_hist["is_anomaly"] == 1]
        if not anom_h.empty:
            fig_h.add_trace(go.Scatter(
                x=anom_h["timestamp"], y=anom_h["temperature"],
                mode="markers", name="Anomalies", marker=dict(color="#ef4444", size=9, symbol="diamond")
            ))
        fig_h.add_hline(y=detector.warning_threshold, line_dash="dash", line_color="#f59e0b")
        fig_h.add_hline(y=detector.critical_threshold, line_dash="dash", line_color="#ef4444")
        fig_h.update_layout(template="plotly_dark", height=380, title="Historical Temperature Trajectory", paper_bgcolor="rgba(15, 23, 42, 0.6)")
        st.plotly_chart(fig_h, use_container_width=True)

        st.subheader("Historical Data Records")
        st.dataframe(df_hist[["id", "timestamp", "temperature", "voltage", "source", "threshold_status", "model_status", "is_anomaly", "scenario_id"]].tail(50), use_container_width=True)
    else:
        st.info("No telemetry records found matching the specified query filters.")


# =============================================================================
# PAGE 4: 🤖 MODEL EVALUATION & COMPARISON
# =============================================================================
elif st.session_state.navigation_page == "🤖 Model Evaluation & Comparison":
    st.header("🤖 Machine Learning Model Evaluation & Comparative Benchmark")
    st.markdown(
        "Objective evaluation of detection techniques against an explicitly labeled zero-leakage benchmark test dataset. "
        "Compares Rule-Based Limits, Rolling MAD, Isolation Forest, and the Combined Hybrid System."
    )

    if st.button("🚀 Re-Run Comparative Benchmark", use_container_width=True):
        st.session_state.benchmark_cache = SmartVoltHybridAnomalyDetector.run_comparative_benchmark()
        st.success("Comparative benchmark successfully executed!")

    if "benchmark_cache" not in st.session_state:
        st.session_state.benchmark_cache = SmartVoltHybridAnomalyDetector.run_comparative_benchmark()

    res_bm = st.session_state.benchmark_cache
    info = res_bm["dataset_info"]

    st.markdown(f"""
    <div style="background-color: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 10px 16px; margin-bottom: 15px;">
        <b>Dataset Information:</b> Split: <code>{info['train_samples']} Train Samples (100% Normal Baseline)</code> vs. 
        <code>{info['test_samples']} Independent Held-Out Test Samples</code> (Normal: {info['normal_samples']}, Labeled Anomalies: {info['anomaly_samples']}) | Evaluated: {info['evaluation_date']}
    </div>
    """, unsafe_allow_html=True)

    # Comparison Table
    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
    approaches = [
        ("Rule-Based Thresholds", res_bm["rule_based"], col_k1),
        ("Rolling Statistical MAD", res_bm["statistical_mad"], col_k2),
        ("Isolation Forest (Unsupervised)", res_bm["isolation_forest"], col_k3),
        ("SmartVolt AI Hybrid", res_bm["hybrid"], col_k4)
    ]

    for title, met, col in approaches:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">{title}</div>
                <div class="metric-value" style="font-size: 1.6rem;">F1: {met['f1_score']:.3f}</div>
                <div class="metric-subtitle">
                    Prec: {met['precision']*100:.1f}% | Rec: {met['recall']*100:.1f}%<br>
                    TP: {met['tp']} | FP: {met['fp']} | FN: {met['fn']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.subheader("Comparative Detection Metrics Table")
    comp_rows = []
    for k in ["rule_based", "statistical_mad", "isolation_forest", "hybrid"]:
        m = res_bm[k]
        comp_rows.append({
            "Detection Technique": m["name"],
            "Precision": f"{m['precision']*100:.1f}%",
            "Recall": f"{m['recall']*100:.1f}%",
            "F1-Score": f"{m['f1_score']:.4f}",
            "True Positives (TP)": m["tp"],
            "False Positives (FP)": m["fp"],
            "False Negatives (FN)": m["fn"],
            "True Negatives (TN)": m["tn"]
        })
    st.table(pd.DataFrame(comp_rows))

    st.markdown("""
    ### 📌 Viva Voce Metric Explanations for Faculty Presentation:
    - **Precision (TP / [TP + FP]):** Measures alarm accuracy. High precision means when an alert fires, it is almost certainly a genuine equipment fault (avoids false alarms).
    - **Recall (TP / [TP + FN]):** Measures fault coverage. High recall means the system misses very few real overheating events (critical for preventing physical equipment damage).
    - **F1-Score:** The harmonic mean of Precision and Recall. Balances false alarms against missed detections.
    - **Why Hybrid Wins:** Rule-based detection has 0 false positives during normal drift but misses early surges before 40°C. Raw MAD catches surges but can flag single-sample noise without debouncing. The **Hybrid System** achieves high recall while suppressing false positives through persistence filtering!
    """)


# =============================================================================
# PAGE 5: 📑 REPORTS & AUDIT EXPORT
# =============================================================================
elif st.session_state.navigation_page == "📑 Reports & Audit Export":
    st.header("📑 Industrial Compliance Reports & Audit Export")
    st.caption("Generate official PDF engineering inspection reports and export verified CSV datasets for offline analysis.")

    db_stats = database.get_database_stats()
    recent_r = database.get_recent_readings(limit=100)
    recent_e = database.get_anomaly_events(limit=50)

    col_rp1, col_rp2 = st.columns(2)
    with col_rp1:
        st.subheader("1. Official PDF Supervisory Report")
        st.write("Generates a comprehensive multi-page engineering inspection document including summary statistics, benchmark metrics, recent anomaly events, and safety notices.")

        pdf_path = os.path.join("data", "smartvolt_audit_report.pdf")
        if st.button("📄 Generate PDF Audit Report", use_container_width=True):
            bm = SmartVoltHybridAnomalyDetector.run_comparative_benchmark()
            reports.generate_pdf_report(db_stats, recent_r, recent_e, bm, pdf_path)
            st.success("PDF Report generated successfully!")

        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="📥 Download Official PDF Report",
                    data=f.read(),
                    file_name="SmartVolt_AI_Audit_Report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

    with col_rp2:
        st.subheader("2. CSV Telemetry & Event Export")
        st.write("Export verified raw telemetry samples and logged anomaly events directly to standard CSV format.")

        if not recent_r.empty:
            csv_r = recent_r.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Export Telemetry Readings (CSV)", data=csv_r, file_name="smartvolt_telemetry.csv", mime="text/csv", use_container_width=True)

        if not recent_e.empty:
            csv_e = recent_e.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Export Anomaly Audit Events (CSV)", data=csv_e, file_name="smartvolt_events.csv", mime="text/csv", use_container_width=True)


# =============================================================================
# PAGE 6: ☁️ THINGSPEAK CLOUD TELEMETRY
# =============================================================================
elif st.session_state.navigation_page == "☁️ ThingSpeak Cloud Telemetry":
    st.header("☁️ ThingSpeak Cloud Telemetry & Remote Monitoring")
    st.caption("Secure HTTPS telemetry uplink (Option B: Python Gateway). Uploads DS18B20 live temperature data to ThingSpeak Channel Field 1. Field 2 (Voltage) remains reserved and disabled until calibrated.")

    st.markdown("""
    <div style="background-color: #0c4a6e; border: 1px solid #0284c7; border-radius: 6px; padding: 12px 16px; margin-bottom: 16px; color: #f0f9ff; font-size: 0.85rem;">
        <strong>📡 THINSPEAK CLOUD ARCHITECTURE (OPTION B — PYTHON GATEWAY):</strong><br>
        Physical telemetry from the ESP8266 + DS18B20 probe is ingested via USB serial (115200 baud) into the local Python runtime, 
        validated by statistical sanity checks, and uploaded over outbound TLS/HTTPS to <code>https://api.thingspeak.com/update</code>.<br>
        • <strong>Field 1:</strong> Physical Temperature (°C) from DS18B20 probe.<br>
        • <strong>Field 2:</strong> Reserved for Voltage (V). <em>Strictly locked and omitted from requests until sensor calibration is benchtop-verified.</em><br>
        • <strong>Rate-Limiting:</strong> Enforces &ge; 15s interval between cloud posts to strictly comply with ThingSpeak Free tier constraints.<br>
        • <strong>Credential Privacy:</strong> API keys are loaded securely from <code>.streamlit/secrets.toml</code> or environment variables and masked across all displays.
    </div>
    """, unsafe_allow_html=True)

    ts_stat = ts_worker.get_status()
    creds = ts_stat["credentials"]
    is_cfg = creds["is_configured"]

    # 4 Status Metric Cards
    col_c1, col_c2, col_c3, col_c4 = st.columns(4)
    with col_c1:
        cfg_badge = '<span class="status-normal">🟢 CONFIGURED</span>' if is_cfg else '<span class="status-warning">⚠️ NOT CONFIGURED</span>'
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Channel Configuration</div>
            <div style="margin: 4px 0;">{cfg_badge}</div>
            <div class="metric-subtitle">ID: {creds['channel_id'] or 'None'} | Key: {creds['masked_write_key']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_c2:
        if not is_cfg:
            up_badge = '<span class="status-warning">⚪ STANDBY (NO KEYS)</span>'
        elif ts_worker.is_enabled:
            up_badge = '<span class="status-normal">🟢 UPLINK ACTIVE</span>'
        else:
            up_badge = '<span class="status-warning">⏸ PAUSED</span>'

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Uplink Gateway</div>
            <div style="margin: 4px 0;">{up_badge}</div>
            <div class="metric-subtitle">Worker: Background Async Daemon</div>
        </div>
        """, unsafe_allow_html=True)

    with col_c3:
        sec_since = ts_stat["seconds_since_last_upload"]
        since_str = f"{sec_since:.1f}s ago" if sec_since is not None else "No uploads yet"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Update Interval</div>
            <div class="metric-value" style="font-size: 1.6rem; color: #38bdf8;">{creds['upload_interval']}s</div>
            <div class="metric-subtitle">Last post: {since_str}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Cloud Counters</div>
            <div class="metric-value" style="font-size: 1.6rem; color: #10b981;">{ts_stat['success_count']} <span style="font-size: 0.9rem; color: #94a3b8;">/ {ts_stat['error_count']} err</span></div>
            <div class="metric-subtitle">Last Entry ID: {ts_stat['last_successful_entry_id'] or '--'}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Configuration Form & Uplink Controls
    col_cfg1, col_cfg2 = st.columns([1, 1])

    with col_cfg1:
        st.subheader("1. Channel Credentials & Settings")
        st.caption("Credentials loaded from environment or `.streamlit/secrets.toml`. You can also enter or override them below for this session:")

        with st.form("thingspeak_credentials_form"):
            in_channel_id = st.text_input("ThingSpeak Channel ID", value=creds["channel_id"], help="Your numeric ThingSpeak Channel ID (e.g., 2865432)")
            in_write_key = st.text_input("Write API Key", type="password", placeholder=creds["masked_write_key"], help="16-character ThingSpeak Write API Key (never logged or exposed)")
            in_read_key = st.text_input("Read API Key (Optional)", type="password", placeholder=creds["masked_read_key"], help="Required if your channel is private and you want to view cloud history")
            in_interval = st.slider("Upload Interval (seconds)", min_value=15, max_value=120, value=creds["upload_interval"], step=5, help="Free accounts require at least 15 seconds between updates")

            btn_save = st.form_submit_button("💾 Save Credentials & Settings", use_container_width=True)
            if btn_save:
                kw = {"upload_interval": in_interval}
                if in_channel_id.strip():
                    kw["channel_id"] = in_channel_id.strip()
                if in_write_key.strip():
                    kw["write_api_key"] = in_write_key.strip()
                if in_read_key.strip():
                    kw["read_api_key"] = in_read_key.strip()

                ts_worker.client.update_credentials(**kw)
                st.success("Credentials and interval updated successfully for current session!")
                st.rerun()

        st.info("💡 **Persistent Setup:** Create `.streamlit/secrets.toml` with your credentials so you don't need to re-enter them every time. See `README.md` or `.streamlit/secrets.toml.example`.")

    with col_cfg2:
        st.subheader("2. Uplink Controls & Operations")

        # Uplink switch
        enable_toggle = st.checkbox(
            "🟢 Enable Automatic Cloud Uplink",
            value=ts_worker.is_enabled,
            help="When checked, incoming valid readings are periodically sent to ThingSpeak in the background."
        )
        ts_worker.set_enabled(enable_toggle)

        # Simulation mode permission switch
        allow_sim_toggle = st.checkbox(
            "🧪 Allow Simulated Telemetry Upload (Test Mode Only)",
            value=ts_worker.allow_simulated_upload,
            help="ACADEMIC INTEGRITY SAFETY: Prevents simulated test runs from polluting your production ThingSpeak channel. Only enable if testing a dedicated sandbox channel."
        )
        ts_worker.set_allow_simulated(allow_sim_toggle)
        if allow_sim_toggle:
            st.warning("⚠️ Test Mode enabled: Simulated values may be posted to ThingSpeak Field 1 with '[SIMULATED]' tag.")

        st.markdown("---")
        st.markdown("**Cloud Operations & Verification:**")

        col_act1, col_act2 = st.columns(2)
        with col_act1:
            if st.button("🧪 Test Connection", use_container_width=True, help="Queries Channel metadata to verify connectivity without posting new data (enforces 5s cooldown)."):
                test_res = ts_worker.client.test_connection()
                if test_res["success"]:
                    st.success(f"✅ {test_res['message']}")
                else:
                    st.error(f"❌ {test_res['message']}")

        with col_act2:
            latest_rec = st.session_state.telemetry_buffer[-1] if st.session_state.telemetry_buffer else None
            can_upload = bool(latest_rec and latest_rec.get("temperature") is not None and latest_rec.get("is_valid"))
            if st.button("⚡ Manual Upload Sample", disabled=not can_upload or not is_cfg, use_container_width=True, help="Uploads the latest valid reading immediately to verify cloud writing."):
                res_man = ts_worker.trigger_immediate_upload(
                    temp_c=latest_rec["temperature"],
                    voltage=latest_rec.get("voltage"),
                    is_simulated=(latest_rec["source"] == "SIMULATED")
                )
                if res_man["success"]:
                    st.success(f"✅ {res_man['message']} ({res_man['duration_ms']}ms)")
                else:
                    st.error(f"❌ {res_man['message']}")

        # Display last upload message
        last_res = ts_stat["last_result"]
        if last_res.get("message"):
            st.caption(f"**Latest Gateway Message:** {last_res['message']} (at {last_res.get('timestamp', '--')})")

    st.markdown("---")

    # 3. Recent Cloud Transactions Feed
    st.subheader("3. Cloud Gateway Activity & Upload Log")
    recent_txs = ts_stat.get("history", [])
    if recent_txs:
        df_tx = pd.DataFrame(recent_txs)
        display_cols = ["timestamp", "temp_c", "entry_id", "status_code", "duration_ms", "success", "message", "source"]
        st.dataframe(df_tx[[c for c in display_cols if c in df_tx.columns]], use_container_width=True, height=180)
    else:
        st.info("No cloud upload attempts logged in this session yet. Enable the uplink or trigger a manual test above.")

    st.markdown("---")

    # 4. Cloud History View (ThingSpeak Read API)
    st.subheader("4. Remote Cloud History & Field Visualization (ThingSpeak Read API)")
    st.caption("Directly query stored cloud data points from `api.thingspeak.com/channels/<ID>/feeds.json` to verify cloud persistence.")

    col_q1, col_q2 = st.columns([1, 3])
    with col_q1:
        q_results = st.selectbox("Feed Samples to Fetch", [10, 25, 50, 100, 200], index=2)
        btn_fetch = st.button("🔄 Fetch Live Cloud Feeds", use_container_width=True)

    if btn_fetch or "cloud_feeds_df" in st.session_state:
        if btn_fetch:
            with st.spinner("Contacting ThingSpeak Read API..."):
                ok, msg, df_feeds, ch_meta = ts_worker.client.fetch_channel_feeds(results=q_results)
                if ok:
                    st.session_state.cloud_feeds_df = df_feeds
                    st.session_state.cloud_channel_meta = ch_meta
                    st.success(msg)
                else:
                    st.error(f"Failed to fetch cloud feeds: {msg}")

        if "cloud_feeds_df" in st.session_state and not st.session_state.cloud_feeds_df.empty:
            df_cloud = st.session_state.cloud_feeds_df
            ch_meta = st.session_state.get("cloud_channel_meta", {})

            st.write(f"**Channel Name:** `{ch_meta.get('name', 'N/A')}` | **Field 1 Name:** `{ch_meta.get('field1', 'Field 1')}` | **Last Entry ID:** `{ch_meta.get('last_entry_id', 'N/A')}`")

            # Plotly Chart of Cloud Temperature
            fig_cloud = go.Figure()
            valid_cloud_temp = df_cloud.dropna(subset=["temperature_c"])
            if not valid_cloud_temp.empty:
                fig_cloud.add_trace(go.Scatter(
                    x=valid_cloud_temp["created_at"],
                    y=valid_cloud_temp["temperature_c"],
                    mode="lines+markers",
                    name="ThingSpeak Field 1 (Temp °C)",
                    line=dict(color="#38bdf8", width=2),
                    marker=dict(size=6, color="#0284c7")
                ))

            fig_cloud.update_layout(
                title=f"Cloud Telemetry Trajectory (Channel {ts_worker.client.channel_id})",
                template="plotly_dark",
                paper_bgcolor="rgba(15, 23, 42, 0.6)",
                plot_bgcolor="rgba(15, 23, 42, 0.8)",
                height=350,
                xaxis=dict(title="Cloud Timestamp (UTC)", showgrid=True, gridcolor="#334155"),
                yaxis=dict(title="Temperature (°C)", showgrid=True, gridcolor="#334155")
            )
            st.plotly_chart(fig_cloud, use_container_width=True)

            col_cf1, col_cf2 = st.columns([3, 1])
            with col_cf1:
                st.markdown("**Cloud Feeds Table:**")
                st.dataframe(df_cloud.tail(50), use_container_width=True)
            with col_cf2:
                st.markdown("**Export Cloud Data:**")
                csv_cloud = df_cloud.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Export Cloud Feeds (CSV)",
                    data=csv_cloud,
                    file_name=f"thingspeak_channel_{ts_worker.client.channel_id}_feeds.csv",
                    mime="text/csv",
                    use_container_width=True
                )


# =============================================================================
# PAGE 7: ⚙️ DIAGNOSTICS & DEVICE HEALTH
# =============================================================================
elif st.session_state.navigation_page == "⚙️ Diagnostics & Device Health":
    st.header("⚙️ Diagnostics & Physical Hardware Link Status")
    st.caption("Inspect physical UART serial link integrity, SQLite storage health, and machine learning model persistence.")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.subheader("1. Serial Link Diagnostics (ESP8266)")
        diag = serial_mgr.get_diagnostics()
        st.write(f"- **Link State:** `{diag['status']}`")
        st.write(f"- **Connected Port:** `{diag['port'] or 'None'}`")
        st.write(f"- **Baud Rate:** `{diag['baudrate']}`")
        st.write(f"- **Packets Ingested:** `{diag['packets_received']}`")
        st.write(f"- **Packet Error Count:** `{diag['error_count']}`")
        st.write(f"- **Last Packet Timestamp:** `{diag['last_packet_time']}`")
        st.write(f"- **Stale Communication Flag:** `{'YES (Stale)' if diag['is_stale'] else 'NO (Live)'}`")
        if diag["last_error"]:
            st.error(f"Last Error: {diag['last_error']}")

    with col_d2:
        st.subheader("2. Storage & Machine Learning Persistence")
        db_s = database.get_database_stats()
        st.write(f"- **Database Path:** `{db_s['db_path']}`")
        st.write(f"- **Database File Size:** `{db_s['file_size_kb']} KB`")
        st.write(f"- **Total Stored Telemetry Rows:** `{db_s['total_readings']}`")
        st.write(f"- **Real Hardware Rows:** `{db_s['hardware_readings']}`")
        st.write(f"- **Simulated Test Rows:** `{db_s['simulated_readings']}`")
        st.write(f"- **Logged Anomaly Events:** `{db_s['total_events']}`")
        st.write(f"- **Saved Model State:** `{'Present (Joblib)' if os.path.exists(detector.model_save_path) else 'Unsaved'}`")
        st.write(f"- **Model Trained At:** `{detector.model_trained_at or 'Not yet saved'}`")

    st.markdown("---")
    st.subheader("3. ThingSpeak Cloud Gateway Diagnostics")
    col_ts1, col_ts2 = st.columns(2)
    ts_diag = ts_worker.get_status()
    with col_ts1:
        st.write(f"- **Cloud Uplink Active:** `{'YES (Running)' if ts_diag['is_enabled'] else 'NO (Paused)'}`")
        st.write(f"- **Configured Channel ID:** `{ts_diag['credentials']['channel_id'] or 'Not Configured'}`")
        st.write(f"- **Write Key (Redacted):** `{ts_diag['credentials']['masked_write_key']}`")
        st.write(f"- **Read Key (Redacted):** `{ts_diag['credentials']['masked_read_key']}`")
    with col_ts2:
        st.write(f"- **Upload Interval:** `{ts_diag['credentials']['upload_interval']} seconds`")
        st.write(f"- **Successful Uploads:** `{ts_diag['success_count']}`")
        st.write(f"- **Failed Uploads:** `{ts_diag['error_count']}`")
        st.write(f"- **Last Gateway Response:** `{ts_diag['last_result']['message']}`")

    st.markdown("---")
    st.subheader("4. Hardware Safety & ADC Interlock Verification")
    st.markdown("""
    <div class="safety-banner">
        <strong>HARDWARE VOLTAGE SENSOR STATUS: LOCKED</strong><br>
        The blue DC voltage sensor uses a 5:1 resistor divider (30kΩ/7.5kΩ). The ESP8266 bare ADC core input 
        limit is 0 to 1.0V (scaled to 3.3V on NodeMCU pin A0). Because the resistor tolerances and voltage limits 
        have not undergone two-point calibration using a regulated benchtop power supply and digital multimeter, 
        <b>hardware voltage acquisition remains strictly locked</b>. NEVER connect household AC mains to pin A0!
    </div>
    """, unsafe_allow_html=True)
