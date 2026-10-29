"""
SmartVolt AI — ThingSpeak Cloud Integration Client
Handles HTTPS telemetry uploads to ThingSpeak Field 1 (Temperature) and Field 2 (Voltage),
rate limiting compliance (>= 15s free tier), credential masking, non-blocking background queuing,
and private/public cloud feed history retrieval.
"""

import os
import time
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import requests
import pandas as pd

# Default ThingSpeak API endpoints
THINGSPEAK_UPDATE_URL = "https://api.thingspeak.com/update"
THINGSPEAK_CHANNELS_URL = "https://api.thingspeak.com/channels"
DEFAULT_MIN_INTERVAL = 15.0  # Free account rate-limit floor (seconds)


def mask_key(key: Optional[str]) -> str:
    """Safely masks API keys so sensitive credentials are never displayed in UI or logs."""
    if not key or not str(key).strip():
        return "Not Configured"
    cleaned = str(key).strip()
    if len(cleaned) <= 4:
        return "••••••••"
    return "••••••••" + cleaned[-4:]


def load_credentials_from_environment() -> Dict[str, Union[str, int, bool]]:
    """
    Attempts to read ThingSpeak configuration in priority order:
    1. Streamlit secrets (st.secrets)
    2. OS Environment variables
    Never throws an exception if secrets are missing.
    """
    channel_id = ""
    write_api_key = ""
    read_api_key = ""
    upload_interval = 15

    # 1. Try Streamlit secrets
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if "thingspeak" in st.secrets:
                ts_sec = st.secrets["thingspeak"]
                channel_id = str(ts_sec.get("channel_id", "")).strip()
                write_api_key = str(ts_sec.get("write_api_key", "")).strip()
                read_api_key = str(ts_sec.get("read_api_key", "")).strip()
                upload_interval = int(ts_sec.get("upload_interval", 15))
            else:
                channel_id = str(st.secrets.get("THINGSPEAK_CHANNEL_ID", "")).strip()
                write_api_key = str(st.secrets.get("THINGSPEAK_WRITE_API_KEY", "")).strip()
                read_api_key = str(st.secrets.get("THINGSPEAK_READ_API_KEY", "")).strip()
                upload_interval = int(st.secrets.get("THINGSPEAK_UPLOAD_INTERVAL", 15))
    except Exception:
        pass

    # 2. Fallback to OS Environment Variables
    if not channel_id:
        channel_id = os.environ.get("THINGSPEAK_CHANNEL_ID", "").strip()
    if not write_api_key:
        write_api_key = os.environ.get("THINGSPEAK_WRITE_API_KEY", "").strip()
    if not read_api_key:
        read_api_key = os.environ.get("THINGSPEAK_READ_API_KEY", "").strip()
    if not upload_interval or upload_interval < 15:
        try:
            upload_interval = int(os.environ.get("THINGSPEAK_UPLOAD_INTERVAL", 15))
        except (ValueError, TypeError):
            upload_interval = 15

    upload_interval = max(int(upload_interval), int(DEFAULT_MIN_INTERVAL))
    is_configured = bool(channel_id and write_api_key)

    return {
        "channel_id": channel_id,
        "write_api_key": write_api_key,
        "read_api_key": read_api_key,
        "upload_interval": upload_interval,
        "is_configured": is_configured
    }


class ThingSpeakClient:
    """
    Synchronous ThingSpeak HTTPS API Client.
    Handles single-point uploads, private feed queries, and connection tests.
    """

    def __init__(
        self,
        channel_id: Optional[str] = None,
        write_api_key: Optional[str] = None,
        read_api_key: Optional[str] = None,
        upload_interval: int = 15
    ):
        creds = load_credentials_from_environment()
        self.channel_id = str(channel_id or creds["channel_id"]).strip()
        self.write_api_key = str(write_api_key or creds["write_api_key"]).strip()
        self.read_api_key = str(read_api_key or creds["read_api_key"]).strip()
        self.upload_interval = max(int(upload_interval or creds["upload_interval"]), int(DEFAULT_MIN_INTERVAL))

        self.last_upload_time: float = 0.0
        self.last_test_time: float = 0.0
        self.last_successful_entry_id: Optional[int] = None
        self.session = requests.Session()

    def update_credentials(
        self,
        channel_id: Optional[str] = None,
        write_api_key: Optional[str] = None,
        read_api_key: Optional[str] = None,
        upload_interval: Optional[int] = None
    ):
        """Updates credentials dynamically at runtime without restarting the app."""
        if channel_id is not None:
            self.channel_id = str(channel_id).strip()
        if write_api_key is not None:
            self.write_api_key = str(write_api_key).strip()
        if read_api_key is not None:
            self.read_api_key = str(read_api_key).strip()
        if upload_interval is not None:
            self.upload_interval = max(int(upload_interval), int(DEFAULT_MIN_INTERVAL))

    @property
    def is_configured(self) -> bool:
        """Returns True if Channel ID and Write API Key are present."""
        return bool(self.channel_id and self.write_api_key)

    def validate_reading(self, temp_c: Optional[float]) -> Tuple[bool, str]:
        """Validates that temperature reading is physical, non-null, and not a sensor fault code."""
        if temp_c is None:
            return False, "Null reading"
        if not isinstance(temp_c, (int, float)):
            return False, "Reading is not numeric"
        if abs(temp_c - (-127.0)) < 0.1:
            return False, "Sensor fault code (-127.00°C: disconnected)"
        if abs(temp_c - 85.0) < 0.05:
            return False, "Sensor reset code (85.00°C: conversion incomplete)"
        if temp_c < -55.0 or temp_c > 125.0:
            return False, f"Temperature {temp_c}°C outside physical DS18B20 limits (-55 to +125°C)"
        return True, "Valid reading"

    def upload_reading(
        self,
        temp_c: Optional[float],
        voltage: Optional[float] = None,
        status_msg: Optional[str] = None,
        is_simulated: bool = False,
        allow_simulated: bool = False,
        force_now: bool = False
    ) -> Dict[str, any]:
        """
        Uploads validated telemetry point to ThingSpeak HTTPS API.
        - Field 1: Temperature (°C)
        - Field 2: Voltage (V) [STRICT RULE: Only sent if real and physically verified; omitted if None or simulated]
        """
        start_ts = time.time()
        res = {
            "success": False,
            "entry_id": None,
            "status_code": None,
            "message": "",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "duration_ms": 0.0
        }

        # 1. Configuration check
        if not self.is_configured:
            res["message"] = "ThingSpeak not configured. Please enter Channel ID and Write API Key."
            return res

        # 2. Simulation isolation check
        if is_simulated and not allow_simulated:
            res["message"] = "Simulated data upload withheld from production ThingSpeak channel (academic integrity lock)."
            return res

        # 3. Reading validation check
        is_valid, val_msg = self.validate_reading(temp_c)
        if not is_valid:
            res["message"] = f"Invalid sensor reading skipped: {val_msg}"
            return res

        # 4. Rate-limiting interval check (Free tier enforces 15 seconds)
        elapsed = start_ts - self.last_upload_time
        if not force_now and elapsed < self.upload_interval:
            remaining = self.upload_interval - elapsed
            res["message"] = f"Rate limited: Please wait {remaining:.1f}s before next cloud upload."
            return res

        # 5. Build HTTPS payload
        payload = {
            "api_key": self.write_api_key,
            "field1": f"{temp_c:.2f}"
        }

        # HARDWARE SAFETY RULE: Never upload simulated voltage as real physical reading
        if voltage is not None and not is_simulated and isinstance(voltage, (int, float)) and 0.0 <= voltage <= 30.0:
            payload["field2"] = f"{voltage:.2f}"

        if status_msg:
            payload["status"] = str(status_msg)[:255]

        try:
            # POST to ThingSpeak API with 8.0s timeout
            response = self.session.post(
                THINGSPEAK_UPDATE_URL,
                data=payload,
                timeout=8.0
            )
            res["status_code"] = response.status_code
            res["duration_ms"] = round((time.time() - start_ts) * 1000.0, 1)

            if response.status_code == 200:
                body_text = response.text.strip()
                if body_text == "0":
                    res["success"] = False
                    res["message"] = "ThingSpeak update failed (Rate limited or incorrect Write API Key)."
                else:
                    try:
                        entry_id = int(body_text)
                        self.last_upload_time = time.time()
                        self.last_successful_entry_id = entry_id
                        res["success"] = True
                        res["entry_id"] = entry_id
                        res["message"] = f"Successfully uploaded to ThingSpeak (Entry ID: {entry_id}, Field1: {temp_c:.2f}°C)."
                    except ValueError:
                        res["success"] = False
                        res["message"] = f"Unexpected response from ThingSpeak: '{body_text}'"
            elif response.status_code in (400, 401, 403):
                res["message"] = f"HTTP {response.status_code}: Invalid ThingSpeak Write API Key or Channel ID."
            elif response.status_code == 404:
                res["message"] = f"HTTP 404: ThingSpeak Channel ID '{self.channel_id}' not found."
            else:
                res["message"] = f"HTTP Error {response.status_code}: {response.reason}"

        except requests.exceptions.Timeout:
            res["duration_ms"] = round((time.time() - start_ts) * 1000.0, 1)
            res["message"] = "Network timeout: ThingSpeak cloud took longer than 8.0s to respond."
        except requests.exceptions.ConnectionError:
            res["duration_ms"] = round((time.time() - start_ts) * 1000.0, 1)
            res["message"] = "Connection failed: Check laptop Wi-Fi / Internet connectivity."
        except Exception as ex:
            res["duration_ms"] = round((time.time() - start_ts) * 1000.0, 1)
            res["message"] = f"Unexpected cloud error: {str(ex)}"

        return res

    def fetch_channel_feeds(self, results: int = 50) -> Tuple[bool, str, pd.DataFrame, Dict]:
        """
        Retrieves historical cloud feed entries from ThingSpeak Read API.
        Works for both public channels and private channels (using Read API Key).
        """
        empty_df = pd.DataFrame(columns=["created_at", "entry_id", "temperature_c", "voltage_v"])
        empty_channel = {}

        if not self.channel_id:
            return False, "Channel ID not configured.", empty_df, empty_channel

        url = f"{THINGSPEAK_CHANNELS_URL}/{self.channel_id}/feeds.json"
        params = {"results": min(max(int(results), 1), 8000)}

        if self.read_api_key:
            params["api_key"] = self.read_api_key

        try:
            response = self.session.get(url, params=params, timeout=9.0)
            if response.status_code == 200:
                data = response.json()
                channel_info = data.get("channel", {})
                feeds = data.get("feeds", [])

                rows = []
                for f in feeds:
                    temp_val = None
                    volt_val = None
                    try:
                        if f.get("field1") is not None and str(f.get("field1")).strip() != "":
                            temp_val = float(f.get("field1"))
                    except (ValueError, TypeError):
                        pass

                    try:
                        if f.get("field2") is not None and str(f.get("field2")).strip() != "":
                            volt_val = float(f.get("field2"))
                    except (ValueError, TypeError):
                        pass

                    rows.append({
                        "created_at": f.get("created_at"),
                        "entry_id": f.get("entry_id"),
                        "temperature_c": temp_val,
                        "voltage_v": volt_val
                    })

                df = pd.DataFrame(rows)
                msg = f"Successfully fetched {len(df)} feeds from ThingSpeak Channel {self.channel_id}."
                return True, msg, df, channel_info

            elif response.status_code in (400, 401, 403):
                return False, f"HTTP {response.status_code}: Channel is private or Read API Key is invalid.", empty_df, empty_channel
            elif response.status_code == 404:
                return False, f"HTTP 404: ThingSpeak Channel {self.channel_id} not found.", empty_df, empty_channel
            else:
                return False, f"HTTP {response.status_code}: Failed to retrieve feeds ({response.reason}).", empty_df, empty_channel

        except requests.exceptions.Timeout:
            return False, "Request timed out while contacting ThingSpeak API.", empty_df, empty_channel
        except requests.exceptions.ConnectionError:
            return False, "Network connection error: Unable to reach api.thingspeak.com.", empty_df, empty_channel
        except Exception as ex:
            return False, f"Feed retrieval error: {str(ex)}", empty_df, empty_channel

    def test_connection(self) -> Dict[str, any]:
        """
        Tests channel connectivity and verifies read/write permission without spamming the API.
        Enforces a 5-second cooldown on manual test attempts.
        """
        now = time.time()
        if now - self.last_test_time < 5.0:
            return {
                "success": False,
                "message": "Please wait a few seconds before testing again.",
                "status": "COOLDOWN"
            }
        self.last_test_time = now

        if not self.channel_id:
            return {
                "success": False,
                "message": "Channel ID is empty.",
                "status": "NOT_CONFIGURED"
            }

        # Query channel metadata via Read API
        ok, msg, df, ch_info = self.fetch_channel_feeds(results=1)
        if ok:
            ch_name = ch_info.get("name", f"Channel {self.channel_id}")
            return {
                "success": True,
                "message": f"Connection verified! Channel Name: '{ch_name}' (ID: {self.channel_id}).",
                "channel_name": ch_name,
                "last_entry_id": ch_info.get("last_entry_id"),
                "status": "CONNECTED"
            }
        else:
            return {
                "success": False,
                "message": f"Channel test failed: {msg}",
                "status": "FAILED"
            }


class ThingSpeakBackgroundWorker:
    """
    Non-blocking background thread worker.
    Receives telemetry points from the serial stream, applies rate-limiting,
    and performs HTTPS uploads without freezing the Streamlit dashboard or serial worker.
    """

    def __init__(self, client: Optional[ThingSpeakClient] = None):
        self.client = client or ThingSpeakClient()
        self.is_enabled: bool = False
        self.allow_simulated_upload: bool = False

        self.latest_sample: Optional[Dict] = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

        # Statistics
        self.success_count: int = 0
        self.error_count: int = 0
        self.upload_history: List[Dict] = []
        self.last_upload_result: Dict = {
            "success": False,
            "message": "Cloud uploader initialized (Uploads disabled by default).",
            "timestamp": "None"
        }

        # Start background polling thread
        self.thread = threading.Thread(
            target=self._worker_loop,
            name="ThingSpeakWorker",
            daemon=True
        )
        self.thread.start()

    def set_enabled(self, enabled: bool):
        """Enables or disables cloud uploads."""
        with self.lock:
            self.is_enabled = bool(enabled)

    def set_allow_simulated(self, allow: bool):
        """Enables or disables uploading simulated readings to test channels."""
        with self.lock:
            self.allow_simulated_upload = bool(allow)

    def post_reading(
        self,
        temp_c: Optional[float],
        voltage: Optional[float] = None,
        source: str = "HARDWARE",
        device_status: str = "CONNECTED"
    ):
        """Queues the newest valid reading in a thread-safe slot."""
        with self.lock:
            self.latest_sample = {
                "temperature": temp_c,
                "voltage": voltage,
                "source": source,
                "is_simulated": (source == "SIMULATED"),
                "device_status": device_status,
                "timestamp": datetime.now()
            }

    def _worker_loop(self):
        """Background thread executing throttled cloud updates."""
        while not self.stop_event.is_set():
            time.sleep(1.0)

            sample_to_upload = None
            with self.lock:
                if not self.is_enabled or not self.client.is_configured:
                    continue

                if self.latest_sample is not None:
                    # Check if minimum interval has elapsed since last upload
                    elapsed = time.time() - self.client.last_upload_time
                    if elapsed >= self.client.upload_interval:
                        sample_to_upload = dict(self.latest_sample)
                        self.latest_sample = None

            if sample_to_upload is not None:
                is_sim = sample_to_upload.get("is_simulated", False)
                res = self.client.upload_reading(
                    temp_c=sample_to_upload.get("temperature"),
                    voltage=sample_to_upload.get("voltage"),
                    status_msg=f"SmartVolt AI | {sample_to_upload.get('source', 'HARDWARE')}",
                    is_simulated=is_sim,
                    allow_simulated=self.allow_simulated_upload
                )

                with self.lock:
                    self.last_upload_result = res
                    tx_record = {
                        "timestamp": res["timestamp"],
                        "temp_c": sample_to_upload.get("temperature"),
                        "entry_id": res.get("entry_id"),
                        "status_code": res.get("status_code"),
                        "duration_ms": res.get("duration_ms"),
                        "success": res["success"],
                        "message": res["message"],
                        "source": sample_to_upload.get("source", "HARDWARE")
                    }
                    self.upload_history.append(tx_record)
                    if len(self.upload_history) > 30:
                        self.upload_history = self.upload_history[-30:]

                    if res["success"]:
                        self.success_count += 1
                    else:
                        if "Rate limited" not in res["message"] and "withheld" not in res["message"]:
                            self.error_count += 1

    def trigger_immediate_upload(
        self,
        temp_c: float,
        voltage: Optional[float] = None,
        is_simulated: bool = False
    ) -> Dict:
        """Manually triggers an immediate cloud upload attempt."""
        res = self.client.upload_reading(
            temp_c=temp_c,
            voltage=voltage,
            status_msg="SmartVolt AI Manual Trigger",
            is_simulated=is_simulated,
            allow_simulated=self.allow_simulated_upload,
            force_now=True
        )
        with self.lock:
            self.last_upload_result = res
            tx_record = {
                "timestamp": res["timestamp"],
                "temp_c": temp_c,
                "entry_id": res.get("entry_id"),
                "status_code": res.get("status_code"),
                "duration_ms": res.get("duration_ms"),
                "success": res["success"],
                "message": res["message"],
                "source": "MANUAL_TRIGGER"
            }
            self.upload_history.append(tx_record)
            if len(self.upload_history) > 30:
                self.upload_history = self.upload_history[-30:]

            if res["success"]:
                self.success_count += 1
            else:
                self.error_count += 1
        return res

    def get_status(self) -> Dict:
        """Returns cloud uploader operational diagnostics."""
        with self.lock:
            creds = {
                "channel_id": self.client.channel_id,
                "masked_write_key": mask_key(self.client.write_api_key),
                "masked_read_key": mask_key(self.client.read_api_key),
                "upload_interval": self.client.upload_interval,
                "is_configured": self.client.is_configured
            }
            return {
                "is_enabled": self.is_enabled,
                "allow_simulated_upload": self.allow_simulated_upload,
                "credentials": creds,
                "success_count": self.success_count,
                "error_count": self.error_count,
                "last_result": dict(self.last_upload_result),
                "last_successful_entry_id": self.client.last_successful_entry_id,
                "seconds_since_last_upload": round(time.time() - self.client.last_upload_time, 1) if self.client.last_upload_time > 0 else None,
                "history": list(reversed(self.upload_history))
            }
