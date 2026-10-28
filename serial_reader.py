"""
SmartVolt AI — Serial Communication & Data Acquisition Engine
Handles COM port discovery, robust background serial streaming, 
fault-tolerant telemetry parsing, stale data detection, and Simulation Lab.
"""

import time
import re
import threading
import random
from collections import deque
from datetime import datetime
from typing import List, Dict, Optional, Tuple

import serial
import serial.tools.list_ports


def list_available_ports() -> List[Dict[str, str]]:
    """
    Scans the system for active serial COM ports.
    Returns a list of dictionaries containing device name and description.
    """
    ports = serial.tools.list_ports.comports()
    port_list = []
    for port in sorted(ports, key=lambda p: p.device):
        port_list.append({
            "device": port.device,
            "description": port.description,
            "hwid": port.hwid or ""
        })
    return port_list


def parse_serial_message(line: str) -> Dict[str, any]:
    """
    Parses incoming serial lines from the ESP8266.
    Target format: "Temperature: 27.50 C"
    Also handles variations, sensor fault codes (-127°C, 85°C), 
    and arbitrary numeric telemetry.
    """
    result = {
        "raw": line.strip() if line else "",
        "temperature": None,
        "is_valid": False,
        "fault_type": None,
        "message": ""
    }

    if not line:
        result["message"] = "Empty serial packet"
        return result

    cleaned = line.strip()
    if not cleaned:
        result["message"] = "Empty serial packet"
        return result

    # 1. Primary Regex: Match "Temperature: <float> C"
    match = re.search(r"Temperature:\s*([+-]?\d+(?:\.\d+)?)\s*°?C?", cleaned, re.IGNORECASE)
    
    # 2. Secondary Regex fallback: Match "Temp: <float>" or "T: <float>"
    if not match:
        match = re.search(r"(?:Temp(?:erature)?|T)[\s:=]+([+-]?\d+(?:\.\d+)?)", cleaned, re.IGNORECASE)

    # 3. Third fallback: Standalone floating point number
    if not match:
        match = re.match(r"^[+-]?\d+(?:\.\d+)?$", cleaned)

    if match:
        try:
            temp_val = float(match.group(1) if match.lastindex else match.group(0))
            result["temperature"] = temp_val

            # Hardware DS18B20 boundary and fault checks
            if abs(temp_val - (-127.0)) < 0.05:
                result["is_valid"] = False
                result["fault_type"] = "SENSOR_DISCONNECTED"
                result["message"] = "DS18B20 1-Wire disconnected or 4.7kΩ pull-up missing (-127°C)"
            elif abs(temp_val - 85.0) < 0.05:
                result["is_valid"] = False
                result["fault_type"] = "POWER_ON_RESET"
                result["message"] = "DS18B20 power-on reset state (conversion incomplete)"
            elif temp_val < -55.0 or temp_val > 125.0:
                result["is_valid"] = False
                result["fault_type"] = "OUT_OF_RANGE"
                result["message"] = f"Reading {temp_val}°C outside physical sensor range (-55 to +125°C)"
            else:
                result["is_valid"] = True
                result["message"] = f"Valid temperature: {temp_val:.2f} °C"
        except (ValueError, IndexError) as err:
            result["is_valid"] = False
            result["message"] = f"Parse conversion error: {str(err)}"
    else:
        result["is_valid"] = False
        result["fault_type"] = "INVALID_FORMAT"
        result["message"] = f"Unrecognized serial format: '{cleaned}'"

    return result


class SmartVoltSerialManager:
    """
    Non-blocking, thread-safe serial port manager for ESP8266 telemetry.
    Maintains a rolling buffer of telemetry records, isolates Streamlit 
    from I/O blocking, and detects stale or dropped connections.
    """

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.buffer = deque(maxlen=max_history)
        self.lock = threading.Lock()

        # Serial configuration
        self.port_name: Optional[str] = None
        self.baudrate: int = 115200
        self.ser: Optional[serial.Serial] = None

        # Threading state
        self.thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.is_connected = False
        self.status = "DISCONNECTED"
        self.last_error = ""

        # Telemetry metrics
        self.packets_received = 0
        self.error_count = 0
        self.connected_since: Optional[datetime] = None
        self.last_packet_time: Optional[datetime] = None

    def connect(self, port: str, baudrate: int = 115200) -> Tuple[bool, str]:
        """Attempts to open the serial port and spawns the background worker thread."""
        if self.is_connected:
            self.disconnect()

        self.port_name = port
        self.baudrate = baudrate
        self.last_error = ""

        try:
            self.ser = serial.Serial(
                port=port,
                baudrate=baudrate,
                timeout=1.0,
                write_timeout=1.0
            )
            self.ser.reset_input_buffer()
            self.is_connected = True
            self.status = "CONNECTED"
            self.connected_since = datetime.now()
            self.stop_event.clear()

            self.thread = threading.Thread(
                target=self._worker_loop,
                name=f"SmartVoltSerial-{port}",
                daemon=True
            )
            self.thread.start()
            return True, f"Successfully connected to {port} @ {baudrate} baud."

        except serial.SerialException as ex:
            error_msg = str(ex)
            self.is_connected = False
            self.status = "ERROR"
            if "PermissionError" in error_msg or "Access is denied" in error_msg:
                self.last_error = (
                    f"Access Denied on {port}. Please verify that Arduino IDE "
                    "Serial Monitor, PuTTY, or other serial terminals are closed."
                )
            elif "FileNotFoundError" in error_msg or "cannot find" in error_msg:
                self.last_error = f"{port} was disconnected or not found."
            else:
                self.last_error = f"Serial error: {error_msg}"
            return False, self.last_error

        except Exception as ex:
            self.is_connected = False
            self.status = "ERROR"
            self.last_error = f"Unexpected connection error: {str(ex)}"
            return False, self.last_error

    def disconnect(self):
        """Safely stops background thread and closes serial handle."""
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)

        if self.ser and self.ser.is_open:
            try:
                self.ser.close()
            except Exception:
                pass

        self.is_connected = False
        self.status = "DISCONNECTED"
        self.ser = None

    def is_stale(self, timeout_seconds: float = 4.0) -> bool:
        """Returns True if the serial link is connected but no reading arrived within timeout."""
        if not self.is_connected or not self.last_packet_time:
            return False
        return (datetime.now() - self.last_packet_time).total_seconds() > timeout_seconds

    def _worker_loop(self):
        """Background thread executing blocking readline() calls safely."""
        while not self.stop_event.is_set():
            if not self.ser or not self.ser.is_open:
                break

            try:
                line_bytes = self.ser.readline()
                if not line_bytes:
                    continue

                line_str = line_bytes.decode("utf-8", errors="replace").strip()
                if not line_str:
                    continue

                parse_res = parse_serial_message(line_str)
                now = datetime.now()

                record = {
                    "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "dt": now,
                    "temperature": parse_res["temperature"],
                    "voltage": None,  # Hardware voltage locked per safety protocol
                    "source": "HARDWARE",
                    "device_status": "CONNECTED",
                    "is_valid": parse_res["is_valid"],
                    "raw": parse_res["raw"],
                    "message": parse_res["message"],
                    "fault_type": parse_res["fault_type"]
                }

                with self.lock:
                    self.packets_received += 1
                    self.last_packet_time = now
                    if not parse_res["is_valid"]:
                        self.error_count += 1
                    self.buffer.append(record)

            except serial.SerialException as ex:
                self.last_error = f"COM port disconnected unexpectedly: {str(ex)}"
                self.status = "DISCONNECTED"
                self.is_connected = False
                break
            except Exception as ex:
                self.error_count += 1
                self.last_error = f"Read loop error: {str(ex)}"
                time.sleep(0.1)

    def get_latest_reading(self) -> Optional[Dict]:
        """Returns the most recent telemetry record."""
        with self.lock:
            if not self.buffer:
                return None
            return self.buffer[-1]

    def get_history(self, limit: int = 100) -> List[Dict]:
        """Returns a snapshot of the most recent N records."""
        with self.lock:
            if not self.buffer:
                return []
            items = list(self.buffer)
            return items[-limit:]

    def clear_buffer(self):
        """Clears stored telemetry samples."""
        with self.lock:
            self.buffer.clear()
            self.packets_received = 0
            self.error_count = 0

    def get_diagnostics(self) -> Dict:
        """Returns hardware serial health and link status."""
        stale = self.is_stale()
        link_state = "STALE" if stale else self.status
        return {
            "status": link_state,
            "is_connected": self.is_connected,
            "is_stale": stale,
            "port": self.port_name,
            "baudrate": self.baudrate,
            "packets_received": self.packets_received,
            "error_count": self.error_count,
            "last_packet_time": self.last_packet_time.strftime("%H:%M:%S") if self.last_packet_time else "None",
            "last_error": self.last_error
        }


class SmartVoltSimulationEngine:
    """
    Dedicated Simulation & Test Lab Engine.
    Generates reproducible, controllable IoT test scenarios for laboratory demonstration.
    ALL DATA PRODUCED IS EXPLICITLY LABELED 'SIMULATED'.
    """

    SCENARIO_INFO = {
        "NORMAL": {
            "title": "Normal Baseline Operation",
            "description": "Nominal room temperature with minor natural sensor noise (~26.5°C ± 0.3°C).",
            "expected_behavior": "Baseline tracking; system reports Normal status with 0 anomalies.",
            "expected_severity": "NORMAL"
        },
        "GRADUAL_RISE": {
            "title": "Slow Gradual Temperature Rise",
            "description": "Slow thermal drift (+0.08°C/s over 60s) simulating ambient afternoon drift or warming load.",
            "expected_behavior": "Rolling median adapts smoothly without triggering false statistical alerts.",
            "expected_severity": "NORMAL"
        },
        "TEMPERATURE_SPIKE": {
            "title": "Sudden Temperature Spike",
            "description": "Abrupt transient impulse (+15°C surge) simulating momentary contact resistance or arcing.",
            "expected_behavior": "Sample 1 debounced as Transient Spike; Sample 2 latches persistent Anomaly.",
            "expected_severity": "WARNING"
        },
        "OVERHEATING": {
            "title": "Thermal Runaway / Overheating",
            "description": "Continuous exponential thermal surge reaching 55°C - 68°C.",
            "expected_behavior": "Breaches Warning (40°C) and Critical (50°C) limits, triggering critical alerts.",
            "expected_severity": "CRITICAL"
        },
        "VOLTAGE_FLUCTUATION": {
            "title": "12V DC Bus Ripple",
            "description": "Simulated nominal 12V DC power bus with mild electrical ripple (11.8V to 12.3V).",
            "expected_behavior": "Displays simulated voltage curve within normal power supply tolerances.",
            "expected_severity": "NORMAL"
        },
        "VOLTAGE_UNDER_OVER": {
            "title": "Voltage Sag & Surge Transients",
            "description": "Simulated undervoltage sag (8.8V) followed by an alternator overvoltage surge (16.4V).",
            "expected_behavior": "Flags power quality anomalies on the simulated voltage trajectory.",
            "expected_severity": "WARNING"
        },
        "MISSING_STALE": {
            "title": "Missing / Stale Sensor Communication",
            "description": "Emulates communication freeze or packet drop where telemetry stops arriving.",
            "expected_behavior": "Diagnostics panel detects stale link timeout (> 4s without packet).",
            "expected_severity": "WARNING"
        },
        "SENSOR_FAULT": {
            "title": "1-Wire Sensor Disconnect Fault",
            "description": "Emulates unplugged DS18B20 data probe, generating -127.00°C hardware error code.",
            "expected_behavior": "Parser intercepts fault code and flags SENSOR_FAULT without crashing.",
            "expected_severity": "CRITICAL"
        }
    }

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.buffer = deque(maxlen=max_history)
        self.lock = threading.Lock()

        # Physical thermal simulation state
        self.base_temp = 26.5
        self.current_temp = 26.5
        self.current_voltage = 12.08
        self.scenario = "NORMAL"
        self.is_running = True
        self.step_counter = 0
        self.packets_generated = 0
        self.rng = random.Random(42)

    def set_scenario(self, scenario: str):
        """Updates active simulation scenario and resets state parameters."""
        if scenario in self.SCENARIO_INFO:
            self.scenario = scenario
            if scenario in ["NORMAL", "VOLTAGE_FLUCTUATION"]:
                self.current_temp = 26.5
                self.current_voltage = 12.08
            elif scenario == "GRADUAL_RISE":
                self.current_temp = 26.5

    def start(self):
        self.is_running = True

    def stop(self):
        self.is_running = False

    def reset(self):
        """Resets simulation back to nominal normal state."""
        with self.lock:
            self.base_temp = 26.5
            self.current_temp = 26.5
            self.current_voltage = 12.08
            self.scenario = "NORMAL"
            self.step_counter = 0
            self.packets_generated = 0
            self.buffer.clear()
            self.rng = random.Random(42)

    def tick(self) -> Dict:
        """
        Generates one simulated telemetry point according to active scenario.
        ALL DATA IS EXPLICITLY LABELED 'SIMULATED'.
        """
        now = datetime.now()
        self.step_counter += 1
        self.packets_generated += 1

        is_valid = True
        fault_type = None
        message = "Normal simulated operation"
        device_status = "SIMULATING"

        if self.scenario == "NORMAL":
            noise = self.rng.gauss(0, 0.12)
            drift = 0.3 * (self.rng.random() - 0.5)
            self.current_temp = max(24.5, min(28.5, self.current_temp + drift * 0.1 + noise * 0.2))
            self.current_voltage = max(11.85, min(12.35, 12.10 + self.rng.gauss(0, 0.04)))
            message = "Simulated normal thermal equilibrium"

        elif self.scenario == "GRADUAL_RISE":
            # Slow 0.08°C rise per step up to ~34°C
            self.current_temp = min(34.5, self.current_temp + 0.08 + self.rng.gauss(0, 0.04))
            self.current_voltage = 12.05 + self.rng.gauss(0, 0.03)
            message = f"SIMULATED GRADUAL DRIFT: Ambient rising ({self.current_temp:.2f}°C)"

        elif self.scenario == "TEMPERATURE_SPIKE":
            self.current_temp = self.base_temp + 15.2 + self.rng.gauss(0, 0.3)
            self.current_voltage = 11.75 + self.rng.gauss(0, 0.1)
            message = "SIMULATED TRANSIENT SPIKE: Sudden thermal impulse"

        elif self.scenario == "OVERHEATING":
            # Thermal runaway climbing past 40°C and 50°C
            self.current_temp = min(68.5, self.current_temp + 0.85 + self.rng.uniform(0.1, 0.3))
            self.current_voltage = 11.90 - (self.current_temp - 26.0) * 0.02
            message = f"SIMULATED THERMAL RUNAWAY: {self.current_temp:.1f}°C"

        elif self.scenario == "VOLTAGE_FLUCTUATION":
            self.current_temp = self.base_temp + self.rng.gauss(0, 0.1)
            self.current_voltage = 12.0 + 0.3 * (self.rng.random() - 0.5)
            message = "SIMULATED DC BUS: Normal nominal ripple"

        elif self.scenario == "VOLTAGE_UNDER_OVER":
            self.current_temp = self.base_temp + self.rng.gauss(0, 0.1)
            # Alternate between sag (8.8V) and surge (16.2V)
            if (self.step_counter // 5) % 2 == 0:
                self.current_voltage = 8.85 + self.rng.uniform(-0.1, 0.2)
                message = f"SIMULATED VOLTAGE SAG: Bus dropped to {self.current_voltage:.2f}V"
            else:
                self.current_voltage = 16.30 + self.rng.uniform(-0.2, 0.3)
                message = f"SIMULATED VOLTAGE SURGE: Bus spiked to {self.current_voltage:.2f}V"

        elif self.scenario == "MISSING_STALE":
            # Emulates stale communication: freezes readings
            device_status = "STALE"
            message = "SIMULATED STALE LINK: No fresh packets arriving"

        elif self.scenario == "SENSOR_FAULT":
            self.current_temp = -127.00
            self.current_voltage = 12.05
            is_valid = False
            fault_type = "SENSOR_DISCONNECTED"
            device_status = "ERROR"
            message = "SIMULATED FAULT: Probe disconnected (-127.00°C)"

        temp_rounded = round(self.current_temp, 2)
        volt_rounded = round(self.current_voltage, 2)

        record = {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "dt": now,
            "temperature": temp_rounded,
            "voltage": volt_rounded,
            "source": "SIMULATED",  # MANDATORY REQUIREMENT: ALWAYS LABELED SIMULATED
            "device_status": device_status,
            "is_valid": is_valid,
            "raw": f"[SIMULATED] Temperature: {temp_rounded:.2f} C | Bus: {volt_rounded:.2f} V",
            "message": message,
            "fault_type": fault_type,
            "scenario_id": self.scenario
        }

        with self.lock:
            self.buffer.append(record)

        return record

    def get_latest_reading(self) -> Optional[Dict]:
        with self.lock:
            if not self.buffer:
                return None
            return self.buffer[-1]

    def get_history(self, limit: int = 100) -> List[Dict]:
        with self.lock:
            if not self.buffer:
                return []
            items = list(self.buffer)
            return items[-limit:]

    def clear_buffer(self):
        with self.lock:
            self.buffer.clear()
            self.packets_generated = 0
            self.current_temp = self.base_temp
            self.current_voltage = 12.08
