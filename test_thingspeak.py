"""
SmartVolt AI — Automated ThingSpeak Unit Test Suite
Uses unittest.mock to test HTTP requests, rate-limiting, error recovery,
data validation, and feed queries completely offline without real API credentials.
"""

import unittest
from unittest.mock import patch, MagicMock
import requests
import pandas as pd

from thingspeak_client import (
    ThingSpeakClient,
    ThingSpeakBackgroundWorker,
    mask_key,
    load_credentials_from_environment
)


class TestThingSpeakClient(unittest.TestCase):

    def setUp(self):
        """Initializes client with dummy test credentials."""
        self.client = ThingSpeakClient(
            channel_id="1234567",
            write_api_key="TESTWRITEKEY1234",
            read_api_key="TESTREADKEY5678",
            upload_interval=15
        )

    # -------------------------------------------------------------------------
    # 1. Successful Telemetry Upload (Mock HTTP 200)
    # -------------------------------------------------------------------------
    @patch("requests.Session.post")
    def test_01_upload_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "42"  # New Entry ID
        mock_post.return_value = mock_resp

        res = self.client.upload_reading(
            temp_c=27.50,
            voltage=None,
            is_simulated=False,
            force_now=True
        )

        self.assertTrue(res["success"])
        self.assertEqual(res["entry_id"], 42)
        self.assertEqual(res["status_code"], 200)
        self.assertIn("Entry ID: 42", res["message"])

        # Verify POST payload sent to ThingSpeak
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        payload = kwargs["data"]
        self.assertEqual(payload["api_key"], "TESTWRITEKEY1234")
        self.assertEqual(payload["field1"], "27.50")
        self.assertNotIn("field2", payload)  # Voltage reserved, not sent when None

    # -------------------------------------------------------------------------
    # 2. Rate Limiting Response ('0') & Interval Enforcement
    # -------------------------------------------------------------------------
    @patch("requests.Session.post")
    def test_02_rate_limiting(self, mock_post):
        # Case A: ThingSpeak returns '0' (rate-limited by server)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "0"
        mock_post.return_value = mock_resp

        res = self.client.upload_reading(26.8, is_simulated=False, force_now=True)
        self.assertFalse(res["success"])
        self.assertIn("Rate limited or incorrect Write API Key", res["message"])

        # Case B: Local client interval throttle
        self.client.last_upload_time = 9999999999.0  # Far in the future
        res_throttled = self.client.upload_reading(26.8, is_simulated=False, force_now=False)
        self.assertFalse(res_throttled["success"])
        self.assertIn("Rate limited", res_throttled["message"])

    # -------------------------------------------------------------------------
    # 3. HTTP Errors (400, 403, 404, 500)
    # -------------------------------------------------------------------------
    @patch("requests.Session.post")
    def test_03_http_errors(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.reason = "Bad Request"
        mock_post.return_value = mock_resp

        res_400 = self.client.upload_reading(27.0, force_now=True)
        self.assertFalse(res_400["success"])
        self.assertIn("Invalid ThingSpeak Write API Key", res_400["message"])

        mock_resp.status_code = 404
        res_404 = self.client.upload_reading(27.0, force_now=True)
        self.assertFalse(res_404["success"])
        self.assertIn("not found", res_404["message"])

    # -------------------------------------------------------------------------
    # 4. Sensor Reading Validation (-127°C, 85°C, None, Out-of-Range)
    # -------------------------------------------------------------------------
    def test_04_reading_validation(self):
        # 1-Wire Disconnected (-127°C)
        res_disc = self.client.upload_reading(-127.00, force_now=True)
        self.assertFalse(res_disc["success"])
        self.assertIn("Sensor fault code (-127.00°C", res_disc["message"])

        # Power-on reset constant (85.00°C)
        res_reset = self.client.upload_reading(85.00, force_now=True)
        self.assertFalse(res_reset["success"])
        self.assertIn("Sensor reset code (85.00°C", res_reset["message"])

        # None reading
        res_none = self.client.upload_reading(None, force_now=True)
        self.assertFalse(res_none["success"])
        self.assertIn("Null reading", res_none["message"])

        # Out of physical range (> 125°C)
        res_high = self.client.upload_reading(140.0, force_now=True)
        self.assertFalse(res_high["success"])
        self.assertIn("outside physical DS18B20 limits", res_high["message"])

    # -------------------------------------------------------------------------
    # 5. Network Timeouts & Connection Failures
    # -------------------------------------------------------------------------
    @patch("requests.Session.post")
    def test_05_network_failures(self, mock_post):
        # Timeout
        mock_post.side_effect = requests.exceptions.Timeout("Connection timed out")
        res_timeout = self.client.upload_reading(27.0, force_now=True)
        self.assertFalse(res_timeout["success"])
        self.assertIn("Network timeout", res_timeout["message"])

        # Connection error / Wi-Fi outage
        mock_post.side_effect = requests.exceptions.ConnectionError("Failed to resolve api.thingspeak.com")
        res_conn = self.client.upload_reading(27.0, force_now=True)
        self.assertFalse(res_conn["success"])
        self.assertIn("Connection failed", res_conn["message"])

    # -------------------------------------------------------------------------
    # 6. Simulated Data Isolation (Academic Integrity)
    # -------------------------------------------------------------------------
    def test_06_simulated_data_isolation(self):
        # Simulated reading without explicit permission MUST be blocked
        res_block = self.client.upload_reading(
            temp_c=27.5,
            is_simulated=True,
            allow_simulated=False,
            force_now=True
        )
        self.assertFalse(res_block["success"])
        self.assertIn("withheld from production ThingSpeak channel", res_block["message"])

    # -------------------------------------------------------------------------
    # 7. Hardware Safety Rule: Voltage Reservation (Field 2)
    # -------------------------------------------------------------------------
    @patch("requests.Session.post")
    def test_07_voltage_reservation(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "101"
        mock_post.return_value = mock_resp

        # Simulated voltage MUST NOT be sent as field2
        self.client.upload_reading(
            temp_c=27.5,
            voltage=12.10,
            is_simulated=True,
            allow_simulated=True,
            force_now=True
        )
        _, kwargs = mock_post.call_args
        self.assertNotIn("field2", kwargs["data"])

        # Real verified voltage CAN be sent
        self.client.upload_reading(
            temp_c=27.5,
            voltage=12.10,
            is_simulated=False,
            force_now=True
        )
        _, kwargs2 = mock_post.call_args
        self.assertIn("field2", kwargs2["data"])
        self.assertEqual(kwargs2["data"]["field2"], "12.10")

    # -------------------------------------------------------------------------
    # 8. Channel Feed Query & DataFrame Parsing (Read API)
    # -------------------------------------------------------------------------
    @patch("requests.Session.get")
    def test_08_fetch_channel_feeds(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "channel": {
                "id": 1234567,
                "name": "SmartVolt Laboratory",
                "field1": "Temperature (C)",
                "field2": "Voltage (V)",
                "last_entry_id": 50
            },
            "feeds": [
                {"created_at": "2026-10-29T04:00:00Z", "entry_id": 49, "field1": "26.85", "field2": None},
                {"created_at": "2026-10-29T04:00:20Z", "entry_id": 50, "field1": "27.10", "field2": None}
            ]
        }
        mock_get.return_value = mock_resp

        ok, msg, df, ch_info = self.client.fetch_channel_feeds(results=10)
        self.assertTrue(ok)
        self.assertEqual(len(df), 2)
        self.assertIn("temperature_c", df.columns)
        self.assertEqual(df.iloc[0]["temperature_c"], 26.85)
        self.assertEqual(df.iloc[1]["temperature_c"], 27.10)
        self.assertEqual(ch_info["name"], "SmartVolt Laboratory")

    # -------------------------------------------------------------------------
    # 9. Test Connection Cooldown
    # -------------------------------------------------------------------------
    @patch("requests.Session.get")
    def test_09_test_connection_cooldown(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"channel": {"name": "Lab Channel"}, "feeds": []}
        mock_get.return_value = mock_resp

        # First test
        res1 = self.client.test_connection()
        self.assertTrue(res1["success"])

        # Immediate second test should hit 5s cooldown
        res2 = self.client.test_connection()
        self.assertFalse(res2["success"])
        self.assertEqual(res2["status"], "COOLDOWN")

    # -------------------------------------------------------------------------
    # 10. Credential Masking Security Rule
    # -------------------------------------------------------------------------
    def test_10_credential_masking(self):
        # Long key (16 chars)
        masked_16 = mask_key("1A2B3C4D5E6F7G8H")
        self.assertEqual(masked_16, "••••••••7G8H")
        self.assertNotIn("1A2B3C4D", masked_16)

        # Short key
        masked_short = mask_key("123")
        self.assertEqual(masked_short, "••••••••")

        # Empty / None
        self.assertEqual(mask_key(""), "Not Configured")
        self.assertEqual(mask_key(None), "Not Configured")


if __name__ == "__main__":
    unittest.main()

