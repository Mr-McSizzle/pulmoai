from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from backend.main import app
from backend.storage import DuplicateEmailError, UserDataStore

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class UserDataStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.store = UserDataStore(Path(self.tempdir.name) / "private")
        self.store.initialize()

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_passwords_are_hashed_and_sessions_are_revocable(self) -> None:
        user = self.store.register("Alex Example", "Alex@Example.com", "correct-horse-battery")
        self.assertEqual(user["email"], "alex@example.com")
        self.assertNotIn("password", user)
        self.assertEqual(self.store.authenticate("ALEX@example.com", "correct-horse-battery")["id"], user["id"])
        self.assertIsNone(self.store.authenticate("alex@example.com", "wrong-password"))

        token, _ = self.store.create_session(user["id"])
        self.assertEqual(self.store.user_for_session(token)["id"], user["id"])
        self.store.revoke_session(token)
        self.assertIsNone(self.store.user_for_session(token))

        with self.assertRaises(DuplicateEmailError):
            self.store.register("Alex Example", "alex@example.com", "correct-horse-battery")

        import sqlite3

        connection = sqlite3.connect(self.store.database_path)
        password_hash = connection.execute("SELECT password_hash FROM users WHERE id = ?", (user["id"],)).fetchone()[0]
        connection.close()
        self.assertNotEqual(password_hash, "correct-horse-battery")

    def test_records_are_isolated_timestamped_downloadable_and_deletable(self) -> None:
        owner = self.store.register("Owner", "owner@example.com", "owner-password-123")
        other = self.store.register("Other", "other@example.com", "other-password-123")
        record = self.store.save_record(
            owner["id"],
            "lab_report",
            "../private/report.txt",
            "text/plain",
            {"raw_text": "test report", "count": 1},
            "Extracted 1 lab value",
            b"a private test report",
        )

        self.assertEqual(record["filename"], "report.txt")
        self.assertTrue(record["has_file"])
        self.assertTrue(record["uploaded_at"].endswith("+00:00"))
        self.assertEqual(len(self.store.list_records(owner["id"])), 1)
        self.assertEqual(self.store.list_records(other["id"]), [])
        self.assertEqual(self.store.get_record(owner["id"], record["id"])["result"]["count"], 1)
        self.assertIsNone(self.store.get_record(other["id"], record["id"]))

        stored_file, metadata = self.store.get_record_file(owner["id"], record["id"])
        self.assertEqual(stored_file.read_bytes(), b"a private test report")
        self.assertEqual(metadata["filename"], "report.txt")
        self.assertFalse(self.store.delete_record(other["id"], record["id"]))
        self.assertTrue(self.store.delete_record(owner["id"], record["id"]))
        self.assertFalse(stored_file.exists())
        self.assertEqual(self.store.list_records(owner["id"]), [])


class UserStorageApiTests(unittest.TestCase):
    def test_authentication_report_history_and_user_isolation(self) -> None:
        temporary_directory = tempfile.TemporaryDirectory()
        previous_store = getattr(app.state, "user_data_store", None)
        app.state.user_data_store = UserDataStore(Path(temporary_directory.name) / "private")
        try:
            with TestClient(app) as client:
                self.assertEqual(client.get("/api/uploads").status_code, 401)
                registration = client.post(
                    "/api/auth/register",
                    json={
                        "full_name": "Report Owner",
                        "email": "report-owner@example.test",
                        "password": "safe-password-123",
                    },
                )
                self.assertEqual(registration.status_code, 201, registration.text)
                self.assertIn("httponly", registration.headers["set-cookie"].lower())

                csv_path = PROJECT_ROOT / "samples" / "scrna_sample_donor.csv"
                csv_contents = csv_path.read_bytes()
                prediction = client.post(
                    "/api/predict",
                    files={"file": ("donor-profile.csv", csv_contents, "text/csv")},
                )
                self.assertEqual(prediction.status_code, 200, prediction.text)
                prediction_result = prediction.json()
                self.assertTrue(prediction_result["upload_id"])
                self.assertTrue(prediction_result["uploaded_at"])

                original_image_service = getattr(app.state, "image_service", None)

                class FakeImageService:
                    def is_target_pneumonia_image(self, contents: bytes) -> bool:
                        return False

                    def predict(self, contents: bytes, **kwargs: object) -> dict[str, object]:
                        return {
                            "image_type": "blood_smear",
                            "modality_name": "Blood Smear",
                            "recognized": True,
                            "status": "success",
                            "predicted_class": "neutrophil",
                            "confidence": 0.9,
                            "disclaimer": "Research-only test result.",
                        }

                app.state.image_service = FakeImageService()
                try:
                    image_response = client.post(
                        "/api/image/predict",
                        files={"file": ("smear.png", b"test image bytes", "image/png")},
                        data={"image_type": "blood_smear"},
                    )
                finally:
                    if original_image_service is None:
                        del app.state.image_service
                    else:
                        app.state.image_service = original_image_service
                self.assertEqual(image_response.status_code, 200, image_response.text)
                image_result = image_response.json()
                self.assertTrue(image_result["upload_id"])
                self.assertTrue(image_result["uploaded_at"])

                risk_response = client.post(
                    "/api/risk/predict",
                    json={"inputs": {"age": 62, "sex_male": 1, "bmi": 29.5, "systolic_bp": 138, "diastolic_bp": 85}},
                )
                self.assertEqual(risk_response.status_code, 200, risk_response.text)

                report = b"LAB REPORT\nTotal Cholesterol: 240 mg/dL (Reference: <200)\nHDL Cholesterol: 40 mg/dL"
                extraction = client.post(
                    "/api/report/extract",
                    files={"file": ("panel.txt", report, "text/plain")},
                )
                self.assertEqual(extraction.status_code, 200, extraction.text)
                extraction_result = extraction.json()
                self.assertTrue(extraction_result["upload_id"])
                self.assertTrue(extraction_result["uploaded_at"])

                records = client.get("/api/uploads").json()
                self.assertEqual(len(records), 4)
                records_by_kind = {record["kind"]: record for record in records}
                self.assertEqual(records_by_kind["lab_report"]["filename"], "panel.txt")
                self.assertEqual(records_by_kind["lab_report"]["uploaded_at"], extraction_result["uploaded_at"])
                self.assertEqual(records_by_kind["scrna_csv"]["filename"], "donor-profile.csv")
                self.assertEqual(records_by_kind["scrna_csv"]["uploaded_at"], prediction_result["uploaded_at"])
                self.assertEqual(records_by_kind["medical_image"]["filename"], "smear.png")
                self.assertEqual(records_by_kind["medical_image"]["uploaded_at"], image_result["uploaded_at"])
                self.assertEqual(records_by_kind["risk_assessment"]["has_file"], False)
                download = client.get(f"/api/uploads/{prediction_result['upload_id']}/download")
                self.assertEqual(download.status_code, 200)
                self.assertEqual(download.content, csv_contents)
                saved_detail = client.get(f"/api/uploads/{extraction_result['upload_id']}")
                self.assertEqual(saved_detail.status_code, 200)
                self.assertIn("raw_text", saved_detail.json()["result"])
                self.assertEqual(client.delete(f"/api/uploads/{extraction_result['upload_id']}").status_code, 204)

                second_registration = client.post(
                    "/api/auth/register",
                    json={
                        "full_name": "Another User",
                        "email": "another-user@example.test",
                        "password": "another-password-123",
                    },
                )
                self.assertEqual(second_registration.status_code, 201, second_registration.text)
                self.assertEqual(client.get("/api/uploads").json(), [])
                self.assertEqual(client.get(f"/api/uploads/{extraction_result['upload_id']}").status_code, 404)

                client.post("/api/auth/logout")
                self.assertEqual(client.get("/api/uploads").status_code, 401)
                login = client.post(
                    "/api/auth/login",
                    json={"email": "another-user@example.test", "password": "another-password-123"},
                )
                self.assertEqual(login.status_code, 200, login.text)
                self.assertEqual(client.get("/api/auth/me").json()["user"]["email"], "another-user@example.test")
        finally:
            app.state.user_data_store = previous_store
            temporary_directory.cleanup()


if __name__ == "__main__":
    unittest.main()
