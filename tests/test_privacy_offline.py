"""Unit tests verifying zero outbound network calls during inference (Pass 4 E7)."""

from __future__ import annotations

import io
import socket
import unittest
from unittest.mock import patch

from PIL import Image

from pulmoai.future_risk_service import FutureRiskService
from pulmoai.image_service import ImageScreeningService
from pulmoai.report_reader import process_lab_report
from pulmoai.risk_service import HealthRiskService
from pulmoai.service import PulmoAIService


class OfflineInferencePrivacyTests(unittest.TestCase):
    """Verify that all default inference pipelines execute 100% locally with zero outbound network calls."""

    def setUp(self) -> None:
        self.orig_connect = socket.socket.connect

    def tearDown(self) -> None:
        socket.socket.connect = self.orig_connect

    def _assert_no_network_call(self, func, *args, **kwargs):
        network_call_made = []

        def forbidden_connect(sock, address):
            # Allow local loopback (127.0.0.1, localhost) if any, forbid all external connections
            host = address[0]
            if host not in ("127.0.0.1", "localhost", "::1"):
                network_call_made.append(address)
                raise PermissionError(f"Outbound network access forbidden in offline inference test: {address}")
            return self.orig_connect(sock, address)

        with patch.object(socket.socket, "connect", side_effect=forbidden_connect):
            result = func(*args, **kwargs)
            self.assertEqual(
                len(network_call_made),
                0,
                f"Expected zero outbound network calls, but detected: {network_call_made}",
            )
            return result

    def test_scrna_seq_inference_is_offline(self) -> None:
        """Verify scRNA-seq classification makes zero network calls."""
        service = PulmoAIService.load()
        dummy_csv = b"gene,count\n" + b"\n".join(f"{g},10".encode() for g in service.feature_names)
        res = self._assert_no_network_call(service.predict_upload, dummy_csv)
        self.assertIn("predicted_class", res)

    def test_nhanes_risk_inference_is_offline(self) -> None:
        """Verify NHANES chronic disease risk engine makes zero network calls."""
        service = HealthRiskService.load()
        inputs = {
            "age": 55.0,
            "sex_male": 1,
            "bmi": 28.5,
            "systolic_bp": 130.0,
            "total_cholesterol": 210.0,
            "hdl_cholesterol": 45.0,
        }
        res = self._assert_no_network_call(service.predict_risk, inputs)
        self.assertIn("risk_predictions", res)

    def test_future_mortality_risk_inference_is_offline(self) -> None:
        """Verify future mortality risk service makes zero network calls."""
        service = FutureRiskService.load()
        inputs = {
            "age": 58.0,
            "sex_male": 1,
            "current_smoker": 1,
            "systolic_bp": 135.0,
            "total_cholesterol": 215.0,
            "hdl_cholesterol": 48.0,
            "waist_cm": 98.0,
        }
        res = self._assert_no_network_call(service.predict_future_risk, inputs)
        self.assertIn("future_endpoints", res)

    def test_image_screening_inference_is_offline(self) -> None:
        """Verify medical image screening and modality gate make zero network calls."""
        service = ImageScreeningService.load()
        img = Image.new("RGB", (64, 64), color=(120, 120, 120))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        res = self._assert_no_network_call(service.predict, png_bytes, "chest_xray")
        self.assertIn("predicted_class", res)

    def test_local_report_reader_inference_is_offline(self) -> None:
        """Verify local lab report parser (without Gemini key) makes zero network calls."""
        sample_report = b"""
        METROPOLITAN CLINICAL LABORATORY
        Patient Name: Test Subject   Age: 50   Sex: Male
        ----------------------------------------------------
        Fasting Blood Glucose: 95.0 mg/dL [70 - 99]
        Total Cholesterol: 185.0 mg/dL [< 200]
        ----------------------------------------------------
        """
        res = self._assert_no_network_call(process_lab_report, sample_report, "sample.txt", "text/plain", gemini_key=None)
        self.assertIn("extracted_tests", res)
        self.assertGreaterEqual(len(res["extracted_tests"]), 1)


if __name__ == "__main__":
    unittest.main()
