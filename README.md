# SmartVolt AI — IoT-Based Voltage and Temperature Anomaly Detection

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E.svg)](https://scikit-learn.org/)
[![Hardware](https://img.shields.io/badge/Microcontroller-ESP8266%20NodeMCU-green.svg)](https://www.espressif.com/)
[![Sensor](https://img.shields.io/badge/Sensor-DS18B20%201--Wire-blueviolet.svg)](https://www.analog.com/)

> **A College Engineering Capstone Project**  
> An industrial-grade Edge IoT supervisory platform integrating real-time embedded telemetry with **Scikit-Learn Isolation Forest** machine learning to detect thermal degradation, sudden transient spikes, and electrical anomalies before catastrophic failure occurs.

---

## 📑 Table of Contents
1. [Project Overview & Engineering Objectives](#-project-overview--engineering-objectives)
2. [Hardware Circuitry & Wiring Guide](#-hardware-circuitry--wiring-guide)
3. [Voltage Sensor Safety & ADC Verification Protocol](#-voltage-sensor-safety--adc-verification-protocol)
4. [Machine Learning Methodology (Isolation Forest)](#-machine-learning-methodology-isolation-forest)
5. [Software Architecture & File Manifest](#-software-architecture--file-manifest)
6. [ThingSpeak Cloud Integration & Remote Monitoring (Option B)](#-thingspeak-cloud-integration--remote-monitoring-option-b)
7. [Windows Setup & Quickstart Guide](#-windows-setup--quickstart-guide)
8. [Step-by-Step Faculty Demonstration Script](#-step-by-step-faculty-demonstration-script)
9. [Comprehensive Viva Voce Q&A Kit](#-comprehensive-viva-voce-qa-kit)
10. [Troubleshooting & FAQs](#-troubleshooting--faqs)

---

## 🎯 Project Overview & Engineering Objectives

In industrial power systems, switchgear cubicles, transformer vaults, and electric vehicle battery enclosures, thermal breakdown accounts for over **60% of electrical failure incidents**. 

Conventional SCADA monitoring architectures rely on **rigid scalar thresholds** (e.g., tripping an alarm only if $T > 65^\circ\text{C}$). This classical approach suffers from critical weaknesses:
1. **Blind to Gradient Surges:** An abnormal jump of $+8^\circ\text{C}$ in 2 seconds (indicating an arcing fault or short-circuit) goes completely unnoticed if the absolute temperature is still at $35^\circ\text{C}$.
2. **Ambient Drift Sensitivity:** Natural day-to-night seasonal variations cause high false-alarm rates in summer and delayed trips in winter.
3. **Supervised Learning Bottleneck:** Classical machine learning classifiers (Neural Networks, SVMs) require thousands of labeled failure examples—a commodity virtually impossible to obtain in real operating plants.

### SmartVolt AI Solution:
- **Unsupervised Isolation Forest:** Operates without labeled failure data, isolating statistical outliers by recursively partitioning multi-dimensional telemetry.
- **Dynamic Thermal Kinematics:** Computes the state vector $\mathbf{x} = \begin{bmatrix} T & \frac{\Delta T}{\Delta t} \end{bmatrix}^T$, evaluating both temperature and thermal rate-of-change simultaneously.
- **Asynchronous Telemetry Stream:** A non-blocking background daemon in Python handles high-speed serial I/O at 115200 baud without UI latency or dropped packets.
- **Academic Simulation Engine:** Built-in scenario injection allows laboratory demonstration of thermal runaway, sensor disconnects, and electrical bus transients even when physical hardware is disconnected.

---

## 🔌 Hardware Circuitry & Wiring Guide

### 1. Hardware Bill of Materials (BOM)
| Component | Specification | Purpose |
|---|---|---|
| **Microcontroller** | NodeMCU ESP8266 v2/v3 (ESP-12E/12F) | 80/160 MHz 32-bit MCU, 10-bit ADC, USB UART |
| **Temperature Sensor** | DS18B20 Waterproof Probe | Maxim 1-Wire digital bus, -55°C to +125°C, 12-bit precision (0.0625°C) |
| **Pull-up Resistor** | 4.7 kΩ (1/4 W) | Pulls 1-Wire open-drain data bus to 3.3V rail |
| **Voltage Sensor** | Blue Analog DC Voltage Module (0-25V) | 5:1 resistive voltage divider ($R_1 = 30\,\text{k}\Omega, R_2 = 7.5\,\text{k}\Omega$) *(Held locked per safety)* |
| **Host PC** | Windows Laptop | Runs Python Streamlit dashboard & Scikit-Learn inference |
| **Cabling** | Micro-USB to USB-A data cable | Serial communication (115200 baud) & power delivery |

---

### 2. DS18B20 Probe Wiring Schematic (Pin-to-Pin)

```
        +----------------------------------------+
        |            ESP8266 NodeMCU             |
        |                                        |
        |    [3V3] --------+--------------------+|
        |                  |                    ||
        |               [4.7kΩ]                 ||
        |                  |                    ||
        |    [ D4] --------+                    ||
        |   (GPIO 2)       |                    ||
        |                  |                    ||
        |    [GND] --------|---------------+    ||
        +------------------|---------------|----+|
                           |               |     |
                           |               |     |
                     +-----+---------------+-----+--+
                     |  YELLOW    RED     BLACK/BLUE|
                     |  (Data)   (VCC)      (GND)   |
                     |                              |
                     |    DS18B20 Waterproof Probe  |
                     +------------------------------+
```

> **Wiring Steps:**
> 1. Connect DS18B20 **Red Wire (VCC)** to NodeMCU **3V3**.
> 2. Connect DS18B20 **Black/Blue Wire (GND)** to NodeMCU **GND**.
> 3. Connect DS18B20 **Yellow Wire (Data/DQ)** to NodeMCU **D4 (GPIO 2)**.
> 4. **Crucial:** Solder or breadboard a **4.7 kΩ pull-up resistor** between the Yellow (Data) and Red (3V3) lines. Without this pull-up, the sensor cannot pull the bus high and will output `-127.00°C`.

---

## ⚡ Voltage Sensor Safety & ADC Verification Protocol

### Critical Hardware Interlock Notice
> [!CAUTION]
> **Live hardware voltage monitoring on pin A0 is strictly LOCKED in the codebase.**  
> Connecting unverified voltage sources or household AC mains directly to the ESP8266 will cause instantaneous dielectric destruction of the chip and poses an electrical shock hazard.

### Engineering Analysis:
1. **ESP8266 Chip ADC Limits:** The silicon core of the ESP8266EX contains a single SAR ADC channel that tolerates **$0.0\text{V}$ to $1.0\text{V}$ maximum**.
2. **NodeMCU Board Divider:** On commercial NodeMCU boards, an onboard resistive divider ($220\,\text{k}\Omega$ series and $100\,\text{k}\Omega$ shunt) scales the external A0 pin to accept up to **$3.3\text{V}$ maximum**.
3. **Blue Sensor Characteristics:** The blue analog DC module uses a 5:1 divider with nominal values:
   $$R_1 = 30\,\text{k}\Omega, \quad R_2 = 7.5\,\text{k}\Omega$$
   $$V_{out} = V_{in} \times \frac{R_2}{R_1 + R_2} = V_{in} \times \frac{7.5}{37.5} = 0.20 \times V_{in}$$
4. **Why It Must Not Be Connected Without Calibration:**
   - Inexpensive blue sensor modules use low-grade $5\%$ tolerance carbon film resistors.
   - The ESP8266 ADC is notoriously non-linear near its lower ($<0.1\text{V}$) and upper ($>3.0\text{V}$) rails.
   - Without verifying the common ground reference with an oscilloscope/DMM, ground loop currents through the USB shield can destroy the laptop motherboard.

### Laboratory Verification Protocol (For Future Hardware Activation)
When a regulated benchtop power supply and digital multimeter become available:
1. Verify $R_1 \approx 30\,\text{k}\Omega$ and $R_2 \approx 7.5\,\text{k}\Omega$ with a DMM while disconnected.
2. Supply a known $5.00\,\text{V}$ DC test signal from a bench supply to the module input terminals.
3. Confirm with a calibrated DMM that output pin `S` produces strictly $1.00\,\text{V} \pm 0.05\,\text{V}$.
4. Connect Sensor GND to NodeMCU GND, and Sensor `S` to pin A0.
5. In software, derive the calibration transfer polynomial: $V_{actual} = m \cdot \text{ADC}_{raw} + c$.

*(In the current build, safe evaluation of DC bus sag and surge is provided via the high-fidelity **12V Simulated DC Bus**).*

---

## 🤖 Machine Learning Methodology (Isolation Forest)

### Theoretical Foundation
Proposed by *Fei Tony Liu, Kai Ming Ting, and Zhi-Hua Zhou (IEEE ICDM 2008)*, the **Isolation Forest (iForest)** is an unsupervised ensemble algorithm fundamentally distinct from density or distance estimation algorithms.

Instead of modeling the manifold of normal points, Isolation Forest explicitly **isolates anomalies**:
1. An ensemble of $t = 100$ isolation trees ($iTrees$) is constructed over subsampled telemetry points.
2. At each node, a feature is selected at random from the 2D feature matrix $\mathbf{X} = [T, \Delta T]$.
3. A random split point $p$ is chosen between $\min(\text{feature})$ and $\max(\text{feature})$.
4. Recursive partitioning terminates when either:
   - A node contains only a single data point.
   - The tree reaches its maximum depth limit $h_{\max} = \lceil \log_2(n) \rceil$.

```
         [All Telemetry Points]
             /            \
   (Split: T < 28.5°C)   (Split: T >= 28.5°C)
         /      \               |
      [Dense   [Dense       [ANOMALY ISOLATED!]
     Cluster]  Cluster]    (Path length h(x) = 2)
```

### Mathematical Formulation
Because anomalous instances are **few** and possess **attribute values substantially different from normal points**, they are isolated near the root of the tree, yielding a significantly shorter average path length $h(x)$.

The anomaly score $s(x, n)$ for an instance $x$ over a dataset of size $n$ is defined by:
$$s(x, n) = 2^{-\frac{\mathbb{E}(h(x))}{c(n)}}$$

Where:
- $h(x)$ is the path length (number of edges traversed from root to terminal leaf).
- $\mathbb{E}(h(x))$ is the expectation of path lengths across the ensemble of 100 $iTrees$.
- $c(n)$ is the average path length of unsuccessful searches in a Binary Search Tree (BST), functioning as an asymptotic normalization constant:
  $$c(n) = 2 \left( \ln(n - 1) + 0.5772156649 \right) - \frac{2(n - 1)}{n}$$

### Decision Metric & Normalization:
- If $\mathbb{E}(h(x)) \to 0$, then $s \to 1.0 \implies$ **Definitive Anomaly**.
- If $\mathbb{E}(h(x)) \to c(n)$, then $s \to 0.5 \implies$ **Indeterminate / Ambient Boundary**.
- If $\mathbb{E}(h(x)) \to n - 1$, then $s \to 0.0 \implies$ **Dense Nominal Cluster**.

In SmartVolt AI, the raw Scikit-Learn `decision_function` is mapped to an operator-intuitive **0% to 100% Anomaly Severity Index**:
$$\text{Severity (\%)} = \text{clip}\left(\frac{0.15 - \text{score}_{raw}}{0.40} \times 100, \quad 0.0, \quad 100.0\right)$$

---

## 📂 Software Architecture & File Manifest

The workspace is structured into self-contained, modular components:

```
c:\Users\MIlav\Desktop\smartvolt\
├── app.py                         # Industrial Streamlit SCADA dashboard (7 multi-page views)
├── thingspeak_client.py           # ThingSpeak HTTPS Cloud API Client & Background Async Worker
├── serial_reader.py               # Non-blocking USB serial daemon, stale detection & Simulation Lab
├── model.py                       # Hybrid Anomaly Detector (Thresholds + MAD + Isolation Forest + Joblib)
├── database.py                    # SQLite persistence layer (WAL mode, query filtering, cooldown)
├── reports.py                     # PDF audit report generator (ReportLab) & CSV exporter
├── test_thingspeak.py             # 10-scenario offline unit test suite for Cloud API with mocked requests
├── test_smartvolt_suite.py        # 8-scenario comprehensive automated test suite
├── test_anomaly_detector.py       # 7-scenario anomaly detector unit test suite
├── requirements.txt               # Python dependency manifest (with requests, reportlab & joblib)
├── esp8266_smartvolt.ino          # NodeMCU Arduino C++ firmware (115200 baud)
├── .streamlit/
│   └── secrets.toml.example       # Template for configuring ThingSpeak API keys securely
├── README.md                      # Complete engineering documentation
└── data/
    ├── smartvolt.db               # SQLite database for telemetry and event audit logs
    ├── models/                    # Saved Scikit-Learn Isolation Forest models (.joblib)
    └── sample_telemetry.csv       # Pre-recorded benchmark telemetry dataset
```

### Module Descriptions:
- **`app.py`**: Industrial SCADA dashboard featuring 7 specialized pages: Live Overview, Simulation Lab, Historical Analytics, Model Evaluation, Reports & Audit Export, ThingSpeak Cloud Telemetry, and Diagnostics.
- **`thingspeak_client.py`**: Cloud integration module handling HTTPS REST uploads, Free-tier rate-limiting compliance ($\ge 15\,\text{s}$ interval), credential masking, non-blocking daemon queuing, and Read API historical feed retrieval.
- **`test_thingspeak.py`**: Automated unit test suite with 10 mocked HTTP test cases covering upload success, rate limiting (response `'0'`), HTTP status codes, sensor sanity checking, network timeouts, credential redaction, and private channel feed retrieval.
- **`serial_reader.py`**: Multithreaded background worker with stale link detection and an expanded 8-scenario simulation engine with expected vs actual response mapping.
- **`model.py`**: Hybrid anomaly engine combining two-stage deterministic limits, rolling median & MAD with safe zero-variance floor, debounced persistence, Isolation Forest with Joblib model saving/loading, and zero-leakage comparative benchmarking.
- **`database.py`**: Robust SQLite persistence engine storing all readings and anomaly events with duplicate cooldown throttling and advanced filtering.
- **`reports.py`**: Generates publication-grade multi-page PDF inspection reports with summary statistics, confusion matrix benchmarks, event logs, and safety notices.
- **`test_smartvolt_suite.py`**: Automated test suite verifying serial parsing, validation, thresholds, MAD statistics, ML persistence, SQLite writes, simulation labeling, and report generation.

---

## ☁️ ThingSpeak Cloud Integration & Remote Monitoring (Option B)

### 1. Architecture Overview (Option B: Python Gateway)

The cloud telemetry pipeline is implemented using **Option B: Python Gateway Architecture**. Physical telemetry from the ESP8266 + DS18B20 sensor is ingested via USB serial (115200 baud), validated by statistical edge sanity filters, and uploaded over TLS/HTTPS to ThingSpeak's REST API (`https://api.thingspeak.com/update`).

```
+------------------+         +------------------+         +----------------------+         +---------------------+
|  DS18B20 Sensor  | 1-Wire  | ESP8266 NodeMCU  | USB UART| Python Async Gateway |  HTTPS  |  ThingSpeak Cloud   |
| Waterproof Probe |-------->| GPIO 2 (Pin D4)  |-------->| (thingspeak_client)  |-------->| (api.thingspeak.com)|
| (-55°C to +125°C)|         | (115200 Baud)    |         | Throttled >= 15s     |         | Field 1: Temp (°C)  |
+------------------+         +------------------+         +----------------------+         +---------------------+
```

#### Why Option B is Superior for College & Laboratory Presentations:
1. **Preserves Existing Firmware:** The working Arduino sketch `esp8266_smartvolt.ino` remains completely untouched. No re-flashing or risk of breaking serial output.
2. **Campus Wi-Fi Independence:** University and corporate Wi-Fi networks typically enforce **WPA2-Enterprise (802.1X)** authentication requiring username/passwords and digital certificates that low-cost ESP8266 boards cannot easily negotiate. Option B uses the laptop's existing internet connection seamlessly.
3. **Hardware Safety Isolation:** Raw sensor noise and fault codes (`-127.00°C`, `85.00°C`) are validated and filtered at the Edge before reaching cloud servers.
4. **Zero Latency on Dashboard:** Telemetry transmission runs in a dedicated background daemon thread (`ThingSpeakBackgroundWorker`), preventing UI freezing or serial buffer drops.

---

### 2. ThingSpeak Channel Field Mapping

| Channel Field | Parameter | Data Source | Safety & Validation Rules |
|---|---|---|---|
| **Field 1** | Temperature (°C) | Physical DS18B20 Probe | Validated numeric values between $-55.0^\circ\text{C}$ and $+125.0^\circ\text{C}$. Rejects fault codes. |
| **Field 2** | Voltage (V) | Reserved | **STRICT INTERLOCK: OMITTED & DISABLED.** Never sends simulated voltage as a real measurement. Hardware voltage monitoring remains locked until bench calibration. |
| **Status** | Gateway Metadata | Text String | Appends packet origin and node status (e.g. `SmartVolt AI | HARDWARE`). |

---

### 3. Rate-Limiting Compliance & Free-Tier Rules

ThingSpeak Free tier accounts enforce a strict rate limit: **minimum 15 seconds between updates**.
- If requests are sent faster than 15 seconds, ThingSpeak responds with HTTP 200 and a body of **`"0"`**, indicating the data was dropped.
- The `ThingSpeakClient` and `ThingSpeakBackgroundWorker` enforce an internal rate-limit floor of **$\ge 15.0\,\text{s}$**.
- High-frequency 1 Hz serial readings are consumed locally for real-time charting, while the cloud uploader automatically queues the most recent valid sample and dispatches it once the 15-second window elapses.

---

### 4. Credential Security & Privacy Protocol

> [!IMPORTANT]
> **API keys are NEVER hardcoded into source files, committed to Git, or displayed in plain text.**

Credentials are read with the following priority:
1. **Streamlit Secrets:** `.streamlit/secrets.toml`
2. **OS Environment Variables:** `THINGSPEAK_CHANNEL_ID`, `THINGSPEAK_WRITE_API_KEY`, `THINGSPEAK_READ_API_KEY`, `THINGSPEAK_UPLOAD_INTERVAL`
3. **Session UI Inputs:** Entered during the session via the Cloud Telemetry dashboard.

All displayed API keys are masked using `mask_key()` (e.g., `••••••••7G8H` or `••••••••`). `.streamlit/secrets.toml` is included in `.gitignore` to prevent credential leakage.

---

### 5. Step-by-Step ThingSpeak Setup Guide

Follow these steps to link your SmartVolt AI installation to your ThingSpeak channel:

#### Step A: Create Channel on ThingSpeak
1. Go to [https://thingspeak.com](https://thingspeak.com) and sign in (or create a free MathWorks account).
2. Click **Channels** $\to$ **My Channels** $\to$ **New Channel**.
3. Fill in the channel settings:
   - **Name:** `SmartVolt AI Telemetry`
   - **Description:** `IoT Voltage & Temperature Anomaly Detection Supervisory Node`
   - **Field 1:** Check box and name it `Temperature (C)`
   - **Field 2:** Check box and name it `Voltage (V)` *(Reserved)*
4. Click **Save Channel**.

#### Step B: Obtain API Keys
1. In your new channel, click the **API Keys** tab.
2. Note your **Channel ID** (a 7-digit number, e.g., `2865432`).
3. Copy your **Write API Key** (16-character alphanumeric string).
4. If your channel is private, also copy your **Read API Key**.

#### Step C: Configure Secrets in SmartVolt AI
Create the file `.streamlit/secrets.toml` (you can copy `.streamlit/secrets.toml.example`):
```toml
[thingspeak]
channel_id = "2865432"               # Replace with your actual Channel ID
write_api_key = "XXXXXXXXXXXXXXXX"   # Replace with your 16-character Write API Key
read_api_key = "YYYYYYYYYYYYYYYY"    # Replace with your Read API Key (if private)
upload_interval = 15                 # Minimum 15 seconds for Free tier
```

*Alternative (Environment Variables):*
```powershell
$env:THINGSPEAK_CHANNEL_ID="2865432"
$env:THINGSPEAK_WRITE_API_KEY="XXXXXXXXXXXXXXXX"
$env:THINGSPEAK_READ_API_KEY="YYYYYYYYYYYYYYYY"
$env:THINGSPEAK_UPLOAD_INTERVAL="15"
```

---

### 6. Testing & Verifying Cloud Integration

#### Step 1: Run Offline Test Suite
Verify that all cloud endpoints, rate-limiting handlers, and mocked HTTP calls pass before connecting to live hardware:
```powershell
python test_thingspeak.py
```
*(All 10 unit tests will run and pass with `OK`).*

#### Step 2: Start the Dashboard
```powershell
streamlit run app.py
```

#### Step 3: Test Connection in Dashboard
1. Open the sidebar and select **☁️ ThingSpeak Cloud Telemetry**.
2. Click **🧪 Test Connection**.
   - If configured correctly, you will receive: `✅ Connection verified! Channel Name: 'SmartVolt AI Telemetry' (ID: ...)`.
   - The test incorporates a 5-second cooldown to prevent API flooding.

#### Step 4: Verify Real-Time Upload
1. In the sidebar or Cloud Telemetry page, check the box: **🟢 Enable Automatic Cloud Uplink**.
2. If real hardware is connected, valid temperature readings will upload every 15 seconds.
3. Check the **Cloud Gateway Activity & Upload Log** table in the dashboard to view Entry IDs and HTTP response times.
4. Open your [ThingSpeak Channel View](https://thingspeak.com) in your web browser. You will observe live temperature points populating the **Field 1** chart in real time!
5. Back in the dashboard, scroll down to **Remote Cloud History & Field Visualization** and click **🔄 Fetch Live Cloud Feeds** to pull the cloud feed back into SmartVolt AI via the ThingSpeak Read API.

---

---

## 🚀 Windows Setup & Quickstart Guide

### Step 1: Open Terminal & Verify Python
Open **PowerShell** or **Command Prompt** and navigate to the project directory:
```powershell
cd c:\Users\MIlav\Desktop\smartvolt
python --version
```
*(Python 3.10 through 3.13 are fully supported).*

---

### Step 2: Install Python Dependencies
Install all required libraries using pip:
```powershell
pip install -r requirements.txt
```
*Packages installed:* `streamlit`, `pyserial`, `pandas`, `plotly`, `scikit-learn`, `numpy`.

---

### Step 3: Flash ESP8266 Firmware (Arduino IDE)
1. Open the **Arduino IDE**.
2. Go to **Sketch** $\to$ **Include Library** $\to$ **Manage Libraries...**
3. Search for and install:
   - **OneWire** *(by Jim Studt, Paul Stoffregen)*
   - **DallasTemperature** *(by Miles Burton)*
4. Open the firmware file:
   `c:\Users\MIlav\Desktop\smartvolt\esp8266_smartvolt.ino`
5. Connect your NodeMCU to your laptop via USB.
6. Under **Tools** menu:
   - **Board:** NodeMCU 1.0 (ESP-12E Module)
   - **Upload Speed:** 115200
   - **Port:** Select your active NodeMCU COM port (e.g., `COM3`, `COM4`, etc.)
7. Click **Upload** (Arrow button) and wait for `Done uploading`.

---

### Step 4: CRITICAL STEP — Close Arduino Serial Monitor
> [!IMPORTANT]
> **Windows COM ports are exclusively locked to one application at a time.**  
> If you have the Arduino IDE **Serial Monitor** or **PuTTY** open, Python will throw `Access is denied (PermissionError)`.  
> **Close the Serial Monitor window completely in Arduino IDE before starting the Python dashboard!**

---

### Step 5: Run Automated Anomaly Detection Tests
Verify the ML, statistical MAD, and debouncing algorithms:
```powershell
python test_anomaly_detector.py
```
*(All 7 unit tests will execute, outputting confusion matrix metrics and passing with `OK`).*

---

### Step 6: Launch the SmartVolt AI Dashboard
Run the Streamlit application from PowerShell:
```powershell
streamlit run app.py
```

Your default web browser will automatically open to:
```
http://localhost:8501
```

---

### Step 7: Connect to Real Hardware
1. In the sidebar, select **Real Hardware (USB Serial)**.
2. In the **Select COM Port** dropdown, pick your ESP8266 device (e.g. `COM3` or `COM4`). If you just plugged it in, click the **🔄 (Refresh)** button.
3. Ensure **Baud Rate** is set to `115200`.
4. Click **🟢 Connect**.
5. Within 2 seconds, the dashboard will display live temperature readings streaming from your DS18B20 probe!

---

## 🎓 Step-by-Step Faculty Demonstration Script

Follow this structured 5-minute procedure during your laboratory project presentation:

```
+-------------------------------------------------------------------------------+
|                      FACULTY DEMONSTRATION TIMELINE                           |
|                                                                               |
| [0:00 - 1:00]  Introduction & Hardware Pinout Walkthrough                    |
| [1:00 - 2:00]  Live DS18B20 Telemetry Stream (Room Temp Baseline)             |
| [2:00 - 3:00]  Finger Heat Test: Thermal Gradient vs. Fixed Threshold         |
| [3:00 - 4:00]  Simulation Mode: Thermal Runaway, Spike & Voltage Sag/Surge    |
| [4:00 - 5:00]  Audit Trail: Feature Space Scatter & Immutable CSV Export      |
+-------------------------------------------------------------------------------+
```

### Stage 1: Hardware & System Presentation (1 min)
- Introduce the project: *"SmartVolt AI — IoT-Based Voltage and Temperature Anomaly Detection"*.
- Point out the **NodeMCU ESP8266** connected via USB.
- Show the **DS18B20 waterproof probe** connected to pin D4 with a 4.7kΩ pull-up resistor.
- Explain the communication chain: DS18B20 $\to$ 1-Wire Digital Bus $\to$ ESP8266 $\to$ USB UART (115200 baud) $\to$ Background Python Daemon $\to$ Streamlit Dashboard.

### Stage 2: Live Room Temperature Baseline (1 min)
- Show the **Live Telemetry Dashboard** tab:
  - Temperature reading: $\approx 25^\circ\text{C} - 28^\circ\text{C}$.
  - System Health: `🟢 NORMAL STATE`.
  - ML Anomaly Severity: $< 15\%$.
  - Rate of change: $\approx 0.00^\circ\text{C/s}$.

### Stage 3: Finger Heat Test & Rate-of-Change Anomaly (1 min)
- Firmly grasp the metal tip of the DS18B20 probe with your fingers (body heat transfer).
- Watch the live Plotly temperature curve rise to $\approx 31^\circ\text{C} - 34^\circ\text{C}$.
- **Key Observation for Faculty:** Point out that although the temperature is still **far below** the static $40.0^\circ\text{C}$ safety threshold, the sudden positive rate of change ($\Delta T \approx +0.8^\circ\text{C/s}$) causes the **Isolation Forest** anomaly score to rise immediately! This proves the ML model detects dynamic thermal distress that static thresholds miss.

### Stage 4: Simulation Demonstration (1 min)
- Toggle the sidebar to **Simulation Mode (Demo)**.
- Point out the yellow **`[SIMULATED]`** badges on all cards and tables (demonstrating academic integrity).
- Select **Thermal Runaway**: Watch the simulated coil temperature climb rapidly past $40^\circ\text{C}$ and trigger the `🔴 THRESHOLD EXCEEDED` alarm.
- Click **💥 Inject Spike**: Demonstrate instantaneous detection of an arcing impulse.
- Select **Voltage Sag / Surge**: Demonstrate monitoring of the simulated 12V DC industrial bus dropping to $8.8\text{V}$ or spiking to $16.4\text{V}$.

### Stage 5: Machine Learning & Audit Export (1 min)
- Switch to the **🤖 Isolation Forest ML Engine** tab:
  - Show the 2D feature scatter plot ($T$ vs $\Delta T$). Show how normal points form a tight cluster while spikes are isolated in sparse regions.
- Switch to the **📁 Data Logs & Export** tab:
  - Click **📥 Download Telemetry Log (CSV)** to show that every sample, ML score, and alert is logged with ISO-8601 timestamps.

---

## 💡 Comprehensive Viva Voce Q&A Kit

### Q1: Why did you implement Isolation Forest instead of standard threshold comparison?
> **Answer:** Fixed thresholds operate as crude binary triggers and are completely blind to thermal rate-of-change ($\frac{\Delta T}{\Delta t}$). For instance, if an electrical winding rises from $25^\circ\text{C}$ to $36^\circ\text{C}$ in 2 seconds due to an inter-turn short circuit, a $40^\circ\text{C}$ threshold will remain inactive until substantial damage has occurred. Isolation Forest evaluates the joint kinematic vector $\begin{bmatrix} T & \frac{\Delta T}{\Delta t} \end{bmatrix}^T$, flagging the rapid transient spike as a statistical anomaly immediately.

---

### Q2: Why is Isolation Forest superior to Deep Learning (Autoencoders or LSTMs) for this IoT task?
> **Answer:** 
> 1. **No Failure Data Required:** In real industrial equipment, destructive failures cannot be induced intentionally to generate training data. Isolation Forest is fully **unsupervised**.
> 2. **Linear Time Complexity:** Isolation Forest exhibits an optimal time complexity of $O(t \cdot \psi \log \psi)$, enabling real-time edge execution on commodity hardware without GPUs.
> 3. **Immune to Distance Metrics:** Unlike k-Means or GMMs, Isolation Forest does not calculate Euclidean distances, preventing distortion caused by varying scale between temperature ($^\circ\text{C}$) and rate of change ($^\circ\text{C/s}$).

---

### Q3: Why is live hardware voltage reading disabled on pin A0?
> **Answer:** The ESP8266 bare chip ADC has a strict physical maximum input of $1.0\text{V}$, and the NodeMCU external A0 pin tolerates $3.3\text{V}$ maximum via an onboard divider. Inexpensive blue DC voltage sensors feature uncalibrated $5\%$ tolerance resistors. Connecting an unverified voltage source or inadvertently connecting AC mains would instantly destroy the microcontroller and create electric shock hazards. In accordance with safety engineering principles, hardware voltage monitoring is locked until bench multimeter calibration is completed.

---

### Q4: Why is a 4.7 kΩ pull-up resistor required on the DS18B20 1-Wire bus?
> **Answer:** The Maxim Dallas 1-Wire bus operates using an **open-drain (open-collector) architecture**. The microcontroller and sensor can only actively pull the communication line LOW to ground ($0\text{V}$). To return the line to logic HIGH ($3.3\text{V}$) during idle states and bit transmissions, an external pull-up resistor is physically required. Without this resistor, the data line floats, and the DallasTemperature library returns `-127.00°C`.

---

### Q5: What does a reading of -127.00°C signify?
> **Answer:** `-127.00°C` is the reserved hardware fault code constant `DEVICE_DISCONNECTED_C` defined in the DallasTemperature sensor library. It indicates that the 1-Wire communication bus timed out, the sensor probe was unplugged, or the 4.7kΩ pull-up resistor is missing. The SmartVolt AI serial parser intercepts this code and marks it as `SENSOR_FAULT`.

---

### Q6: How does the software prevent Streamlit from freezing while waiting for serial data?
> **Answer:** We implemented a multithreaded architecture. The `SmartVoltSerialManager` runs a dedicated background daemon thread (`threading.Thread`) that executes blocking `serial.readline()` calls. Parsed records are pushed into a thread-safe `collections.deque` protected by a `threading.Lock`. The Streamlit UI queries the buffer asynchronously via `@st.fragment(run_every="2s")`, ensuring 60 FPS UI responsiveness and zero packet loss.

---

### Q7: What baud rate is used, and what does it mean?
> **Answer:** We utilize **115200 baud** (bits per second). At 115200 baud with standard 8N1 framing (1 start bit, 8 data bits, 1 stop bit = 10 bits per character), the link can transmit up to $11,520$ ASCII characters per second. Since each telemetry line is approximately 25 bytes, this baud rate consumes less than $0.25\%$ of link capacity, leaving ample bandwidth for high-frequency sampling without transmission delay.

---

### Q8: Why did you implement ThingSpeak cloud integration using an Edge Gateway (Option B) rather than flashing Wi-Fi code directly onto the ESP8266?
> **Answer:** Option B separates embedded sensor acquisition from cloud networking, providing significant engineering advantages:
> 1. **Firmware Stability:** Preserves the verified, low-jitter 1-Wire DS18B20 sampling routine on the ESP8266 without introducing network disconnect handling or Wi-Fi reconnection stalls.
> 2. **Enterprise Wi-Fi Compatibility:** University networks use 802.1X WPA2-Enterprise protocols that standard ESP8266 hardware cannot negotiate. The host PC gateway connects seamlessly through the laptop's network interface.
> 3. **Edge Validation & Rate-Limiting:** Telemetry is statistically validated, debounced, and checked for sensor faults at the edge. Fault codes (`-127.00°C`) are prevented from polluting cloud charts.
> 4. **TLS/HTTPS Security:** Encrypted TLS sessions are offloaded from the 80 MHz microcontroller to the PC, avoiding heap exhaustion and buffer overflows on the chip.

---

### Q9: How does the software handle ThingSpeak's Free-tier 15-second rate limit?
> **Answer:** ThingSpeak Free tier returns HTTP 200 with a body of `'0'` if updates arrive faster than once every 15 seconds. In SmartVolt AI, the `ThingSpeakBackgroundWorker` runs an asynchronous daemon thread that tracks `last_upload_time`. When fresh 1 Hz serial readings arrive, they update local charts immediately, but only the most recent validated sample is uploaded to ThingSpeak once $\Delta t \ge 15.0\,\text{s}$. This guarantees zero API rate-limit violations while maintaining real-time local supervisory monitoring.

---

## 🛠️ Troubleshooting & FAQs

### Problem 1: `serial.serialutil.SerialException: could not open port 'COMx': PermissionError(13, 'Access is denied.')`
- **Root Cause:** Another application has opened and locked the COM port. On Windows, serial ports cannot be shared simultaneously.
- **Solution:** Close the **Arduino IDE Serial Monitor**, **PuTTY**, **Tera Term**, or any other terminal software. Then click **🟢 Connect** in the dashboard.

---

### Problem 2: Dashboard displays `Temperature: -127.00 °C` with `SENSOR_FAULT`
- **Root Cause:** The DS18B20 probe is disconnected or the 4.7 kΩ pull-up resistor is missing.
- **Solution:**
  1. Check that the Yellow wire is firmly seated in NodeMCU pin **D4**.
  2. Verify that the 4.7 kΩ resistor is bridged between **D4** and **3V3**.
  3. Verify that the Red wire is connected to **3V3** and Black/Blue to **GND**.

---

### Problem 3: No COM ports appear in the sidebar dropdown
- **Root Cause:** Missing USB-UART bridge drivers on Windows (CH340 or CP2102).
- **Solution:**
  1. Open Windows **Device Manager** and expand **Ports (COM & LPT)**.
  2. If a device appears with a yellow exclamation mark (e.g. `USB2.0-Serial` or `CP2102`), install the appropriate **CH340G** or **Silicon Labs CP210x** VCP driver for Windows.
  3. Re-plug the USB cable and click the **🔄 (Refresh)** button in the sidebar.

---

### Problem 4: ThingSpeak returns `ThingSpeak update failed (Rate limited or incorrect Write API Key)`
- **Root Cause:** Either the Write API Key is incorrect, or requests are being submitted within less than 15 seconds of the previous post.
- **Solution:**
  1. Verify your 16-character Write API Key under your Channel's **API Keys** tab on [thingspeak.com](https://thingspeak.com).
  2. Keep the upload interval set to at least 15 seconds (default).
  3. Ensure no other scripts or devices are writing to the same channel simultaneously.

---

### Problem 5: ThingSpeak returns `HTTP 404: Channel ID not found` or `HTTP 400/403`
- **Root Cause:** The Channel ID is mistyped, or the channel is Private and the Read API Key was omitted when attempting to fetch feeds.
- **Solution:**
  1. Copy the numeric Channel ID directly from the ThingSpeak dashboard (e.g. `2865432`).
  2. For private channels, enter your **Read API Key** in `.streamlit/secrets.toml` or the Cloud Telemetry configuration form.

---

## 📜 Academic Attribution & License
Developed as an advanced college undergraduate engineering project demonstrating:
- Embedded Systems & 1-Wire Digital Sensor Interfacing
- Non-Blocking Asynchronous Microcontroller Serial Telemetry
- Unsupervised Machine Learning for Industrial Fault Isolation
- Modern Web-Based SCADA & Supervisory Control Dashboards

*Licensed under the MIT License. Free for academic research, university coursework, and laboratory instruction.*

