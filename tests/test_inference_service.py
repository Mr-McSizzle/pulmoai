from __future__ import annotations

import unittest

import numpy as np

from pulmoai.service import MAX_UPLOAD_BYTES, PulmoAIService


class InferenceServiceInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = PulmoAIService(
            {
                "pipeline": _EncodedClassifier(),
                "feature_names": ["g1", "g2", "g3"],
                "label_order": ["Donor", "COPD"],
                "model_name": "Test classifier",
            },
            "test-artifact",
        )

    def test_validation_accepts_one_complete_long_profile(self) -> None:
        csv_bytes = b"gene_id,count\ng1,2\ng2,3\ng3,5\n"
        result = self.service.validate_upload(csv_bytes)
        self.assertTrue(result["valid"])
        self.assertEqual(result["profile_count"], 1)
        self.assertEqual(result["recognized_gene_count"], 3)

    def test_prediction_maps_encoded_class_to_saved_label(self) -> None:
        csv_bytes = b"gene,count\ng3,5\ng1,2\ng2,3\n"
        result = self.service.predict_upload(csv_bytes)
        self.assertEqual(result["predicted_class"], "COPD")
        self.assertTrue(result["probabilities_available"])
        self.assertIn("class_probabilities", result)
        self.assertIn("Donor", result["class_probabilities"])
        self.assertIn("COPD", result["class_probabilities"])
        self.assertAlmostEqual(
            sum(result["class_probabilities"].values()), 1.0, places=4
        )

    def test_large_upload_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "8 MB"):
            self.service.parse_csv_bytes(b"x" * (MAX_UPLOAD_BYTES + 1))

    def test_missing_gene_is_rejected_before_prediction(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing"):
            self.service.validate_upload(b"gene,count\ng1,2\ng2,3\n")

    def test_empty_upload_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty"):
            self.service.validate_upload(b"")

    def test_prediction_without_predict_proba(self) -> None:
        class _NoProbaClassifier:
            classes_ = np.array([0, 1])
            def predict(self, values: np.ndarray) -> np.ndarray:
                return np.array([0])
        service = PulmoAIService(
            {
                "pipeline": _NoProbaClassifier(),
                "feature_names": ["g1", "g2", "g3"],
                "label_order": ["Donor", "COPD"],
                "model_name": "Test classifier",
            },
            "test-artifact",
        )
        csv_bytes = b"gene,count\ng3,5\ng1,2\ng2,3\n"
        result = service.predict_upload(csv_bytes)
        self.assertEqual(result["predicted_class"], "Donor")
        self.assertFalse(result["probabilities_available"])
        self.assertIsNone(result["class_probabilities"])


class _EncodedClassifier:
    classes_ = np.array([0, 1])

    def predict(self, values: np.ndarray) -> np.ndarray:
        if values.shape != (1, 3):
            raise AssertionError("service passed an unexpected expression shape")
        return np.array([1])

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        if values.shape != (1, 3):
            raise AssertionError("service passed an unexpected expression shape")
        return np.array([[0.2, 0.8]])


if __name__ == "__main__":
    unittest.main()
