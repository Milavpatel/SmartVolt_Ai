/*
  ====================================================================================
  SmartVolt AI — IoT-Based Voltage and Temperature Anomaly Detection
  NodeMCU ESP8266 Firmware (ds18b20 Telemetry)
  ====================================================================================
  
  PROJECT DESCRIPTION:
  This firmware reads high-precision temperature measurements from a waterproof DS18B20 
  sensor over the Maxim 1-Wire protocol and transmits formatted telemetry over USB Serial 
  at 115200 baud to the SmartVolt AI Python/Streamlit dashboard.

  HARDWARE CONFIGURATION:
  - Microcontroller: ESP8266 NodeMCU v2 or v3 (ESP-12E / ESP-12F)
  - Temperature Sensor: DS18B20 Waterproof Probe (Dallas 1-Wire)
  
  WIRING DIAGRAM (DS18B20 -> NodeMCU):
  - DS18B20 Red Wire   (VCC)  -> NodeMCU 3V3 (or 5V if 3V3 rail is noisy)
  - DS18B20 Black Wire (GND)  -> NodeMCU GND
  - DS18B20 Yellow Wire (DQ)  -> NodeMCU Pin D4 (GPIO 2)
  - PULL-UP RESISTOR: Connect a 4.7kΩ resistor between DS18B20 DQ (D4) and 3V3.
    (Essential for 1-Wire open-drain communication)

  ====================================================================================
  IMPORTANT SAFETY NOTICE REGARDING ANALOG DC VOLTAGE SENSOR (A0):
  ====================================================================================
  - The blue DC voltage sensor uses a 5:1 resistor divider (30kΩ and 7.5kΩ).
  - The ESP8266 bare chip ADC input limit is 0 to 1.0V.
  - The NodeMCU board includes an internal resistor divider (220kΩ / 100kΩ) which 
    scales the external A0 pin to accept 0 to 3.3V maximum.
  - Feeding uncalibrated or higher voltages into A0 CAN PERMANENTLY DESTROY THE ESP8266.
  - NEVER CONNECT HOUSEHOLD AC MAINS (110V/230V) UNDER ANY CIRCUMSTANCES!
  - In compliance with project safety specifications, live hardware voltage reading 
    is LOCKED and commented out below until verified with a regulated bench supply.
  ====================================================================================

  REQUIRED ARDUINO LIBRARIES:
  Install via Arduino IDE Library Manager:
  1. "OneWire" by Jim Studt, Tom Pollard, Robin Bliss, Paul Stoffregen
  2. "DallasTemperature" by Miles Burton, Tim Newsome, etc.
*/

#include <OneWire.h>
#include <DallasTemperature.h>

// Pin Definitions
#define ONE_WIRE_BUS 2        // Pin D4 on NodeMCU corresponds to GPIO 2
#define STATUS_LED   LED_BUILTIN // NodeMCU Onboard LED (Active LOW)

// Setup a oneWire instance to communicate with any OneWire devices
OneWire oneWire(ONE_WIRE_BUS);

// Pass our oneWire reference to Dallas Temperature library
DallasTemperature sensors(&oneWire);

// Telemetry interval (milliseconds)
const unsigned long TELEMETRY_INTERVAL_MS = 1000;
unsigned long lastTelemetryTime = 0;

void setup() {
  // Initialize USB Serial communication at 115200 baud
  Serial.begin(115200);
  
  // Initialize onboard LED for heartbeat indication
  pinMode(STATUS_LED, OUTPUT);
  digitalWrite(STATUS_LED, HIGH); // Turn off LED initially (Active LOW)

  // Wait a brief moment for serial pipeline stabilization
  delay(1000);
  Serial.println();
  Serial.println("==================================================");
  Serial.println(" SmartVolt AI — ESP8266 Telemetry Firmware");
  Serial.println(" Target: NodeMCU ESP8266 | Baud: 115200");
  Serial.println(" Sensor: DS18B20 1-Wire Temperature Probe (Pin D4)");
  Serial.println(" Status: Initializing...");
  Serial.println("==================================================");

  // Start up the DallasTemperature sensor library
  sensors.begin();
  
  // Set sensor resolution: 9, 10, 11, or 12 bits (12-bit gives 0.0625°C precision)
  sensors.setResolution(12);

  int deviceCount = sensors.getDeviceCount();
  Serial.print("1-Wire Devices Found: ");
  Serial.println(deviceCount);

  if (deviceCount == 0) {
    Serial.println("WARNING: No DS18B20 probe detected on Pin D4!");
    Serial.println("Check: 1) 4.7k pull-up resistor between D4 & 3V3");
    Serial.println("       2) Red=3V3, Black=GND, Yellow=D4");
  } else {
    Serial.println("DS18B20 detected successfully. Commencing telemetry stream.");
  }
  Serial.println("==================================================");
}

void loop() {
  unsigned long currentMillis = millis();

  // Send telemetry at regular intervals without blocking delay()
  if (currentMillis - lastTelemetryTime >= TELEMETRY_INTERVAL_MS) {
    lastTelemetryTime = currentMillis;

    // Flash onboard LED briefly during sensor read and transmit
    digitalWrite(STATUS_LED, LOW); // LED ON

    // Request temperature measurement from DS18B20 probe
    sensors.requestTemperatures();
    float tempC = sensors.getTempCByIndex(0);

    // Verify valid reading
    if (tempC == DEVICE_DISCONNECTED_C) {
      // DS18B20 returns -127.00 C if disconnected or disconnected during read
      Serial.println("Temperature: -127.00 C [SENSOR_ERROR_DISCONNECTED]");
    } else if (tempC == 85.00) {
      // 85.00 C is the power-on reset value if read before conversion completes
      Serial.println("Temperature: 85.00 C [POWER_ON_RESET_VALUE]");
    } else {
      // Standard output format parsed by SmartVolt AI Python Dashboard:
      // Format: "Temperature: XX.XX C"
      Serial.print("Temperature: ");
      Serial.print(tempC, 2);
      Serial.println(" C");
    }

    /*
      ===========================================================================
      FUTURE CALIBRATED VOLTAGE READING CODE (KEPT LOCKED PER SAFETY PROTOCOL):
      ===========================================================================
      Once the blue voltage sensor is calibrated against a calibrated multimeter:
      
      int rawADC = analogRead(A0); // 0 to 1023
      float vOut = (rawADC / 1023.0) * 3.3; // NodeMCU A0 max is 3.3V
      float vIn = vOut * 5.0; // Sensor has a 5:1 divider (30k / 7.5k)
      // Serial.print(" | Voltage: "); Serial.print(vIn, 2); Serial.println(" V");
      ===========================================================================
    */

    digitalWrite(STATUS_LED, HIGH); // LED OFF
  }
}

