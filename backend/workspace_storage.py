"""Patient-scoped longitudinal records, additive SQLite migration and audit trail."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import date
from typing import Any
from uuid import uuid4

from backend.storage import UserDataStore


def migrate_workspace(connection: sqlite3.Connection) -> None:
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS patients (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            alias TEXT NOT NULL, reported_sex TEXT NOT NULL DEFAULT 'unknown', created_at TEXT NOT NULL,
            UNIQUE(user_id, alias)
        );
        CREATE TABLE IF NOT EXISTS audit_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            action TEXT NOT NULL, resource_id TEXT NOT NULL, occurred_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS audit_owner_idx ON audit_events(user_id, id DESC);
    """)
    columns = {row[1] for row in connection.execute("PRAGMA table_info(uploads)")}
    for name, definition in {
        "patient_id": "TEXT REFERENCES patients(id)", "observed_on": "TEXT",
        "series_label": "TEXT NOT NULL DEFAULT ''", "review_status": "TEXT NOT NULL DEFAULT 'unreviewed'",
        "content_sha256": "TEXT",
    }.items():
        if name not in columns:
            connection.execute(f"ALTER TABLE uploads ADD COLUMN {name} {definition}")
    connection.execute("CREATE INDEX IF NOT EXISTS uploads_patient_idx ON uploads(user_id, patient_id, observed_on)")
    connection.execute("""CREATE UNIQUE INDEX IF NOT EXISTS uploads_duplicate_idx
        ON uploads(user_id, patient_id, kind, observed_on, content_sha256)
        WHERE patient_id IS NOT NULL AND content_sha256 IS NOT NULL""")


class WorkspaceRepository:
    def __init__(self, store: UserDataStore):
        self.store = store

    def _audit(self, connection: sqlite3.Connection, user_id: str, action: str, resource_id: str) -> None:
        connection.execute("INSERT INTO audit_events(user_id, action, resource_id, occurred_at) VALUES (?, ?, ?, ?)",
                           (user_id, action, resource_id, self.store._now()))

    def patients(self, user_id: str) -> list[dict[str, Any]]:
        with self.store._connection() as c:
            return [dict(r) for r in c.execute("""SELECT p.id, p.alias, p.reported_sex, p.created_at,
                COUNT(u.id) AS record_count, MAX(u.observed_on) AS latest_observation
                FROM patients p LEFT JOIN uploads u ON p.id=u.patient_id AND p.user_id=u.user_id
                WHERE p.user_id=? GROUP BY p.id ORDER BY p.alias COLLATE NOCASE""", (user_id,))]

    def patient(self, user_id: str, patient_id: str) -> dict[str, Any] | None:
        with self.store._connection() as c:
            row = c.execute("SELECT id, alias, reported_sex, created_at FROM patients WHERE id=? AND user_id=?",
                            (patient_id, user_id)).fetchone()
            return dict(row) if row else None

    def create_patient(self, user_id: str, alias: str, reported_sex: str) -> dict[str, Any]:
        alias = " ".join(alias.split())
        if not alias or len(alias) > 80:
            raise ValueError("Use a patient alias between 1 and 80 characters.")
        patient_id = uuid4().hex
        with self.store._connection() as c:
            try:
                c.execute("INSERT INTO patients VALUES (?, ?, ?, ?, ?)",
                          (patient_id, user_id, alias, reported_sex, self.store._now()))
            except sqlite3.IntegrityError as error:
                raise ValueError("A patient with this alias already exists.") from error
            self._audit(c, user_id, "patient.created", patient_id)
        return self.patient(user_id, patient_id)

    @staticmethod
    def validate_date(observed_on: str) -> str:
        parsed = date.fromisoformat(observed_on)
        if parsed.isoformat() != observed_on or parsed > date.today():
            raise ValueError("Use a valid report date (YYYY-MM-DD) that is not in the future.")
        return observed_on

    def duplicate(self, user_id: str, patient_id: str, kind: str, observed_on: str, digest: str) -> str | None:
        with self.store._connection() as c:
            row = c.execute("""SELECT id FROM uploads WHERE user_id=? AND patient_id=? AND kind=?
                AND observed_on=? AND content_sha256=?""", (user_id, patient_id, kind, observed_on, digest)).fetchone()
            return row[0] if row else None

    def attach(self, user_id: str, record_id: str, patient_id: str, observed_on: str, series_label: str = "") -> None:
        self.validate_date(observed_on)
        if not self.patient(user_id, patient_id):
            raise LookupError("Patient not found.")
        if not self.store.get_record(user_id, record_id):
            raise LookupError("Record not found.")
        original = self.store.get_record_file(user_id, record_id)
        digest = hashlib.sha256(original[0].read_bytes()).hexdigest() if original else None
        with self.store._connection() as c:
            try:
                c.execute("""UPDATE uploads SET review_status=CASE WHEN patient_id=? AND observed_on=? AND series_label=? THEN review_status ELSE 'unreviewed' END, patient_id=?, observed_on=?, series_label=?, content_sha256=?
                    WHERE id=? AND user_id=?""", (patient_id, observed_on, series_label.strip(), patient_id, observed_on, series_label.strip(), digest, record_id, user_id))
            except sqlite3.IntegrityError as error:
                raise ValueError("This file is already linked to this patient on this date.") from error
            self._audit(c, user_id, "record.linked", record_id)

    def records(self, user_id: str, patient_id: str) -> list[dict[str, Any]]:
        with self.store._connection() as c:
            rows = c.execute("""SELECT * FROM uploads WHERE user_id=? AND patient_id=?
                ORDER BY observed_on, uploaded_at, id""", (user_id, patient_id)).fetchall()
        from pulmoai.provenance import data_origin
        def with_origin(row):
            result = json.loads(row["result_json"])
            result.setdefault("data_origin", data_origin(result.get("raw_text", "").encode(), row["original_filename"]))
            return result
        return [{**self.store._public_record(r), "patient_id": r["patient_id"], "observed_on": r["observed_on"],
                 "series_label": r["series_label"], "review_status": r["review_status"],
                 "result": with_origin(r)} for r in rows]

    def review(self, user_id: str, record_id: str, corrections: dict[str, float]) -> None:
        with self.store._connection() as c:
            row = c.execute("SELECT * FROM uploads WHERE id=? AND user_id=?", (record_id, user_id)).fetchone()
            if not row:
                raise LookupError("Record not found.")
            result = json.loads(row["result_json"])
            tests = result.get("extracted_tests", [])
            if row["kind"] == "lab_report" and not tests:
                raise ValueError("A report with no extracted values cannot be reviewed.")
            if set(corrections) - {t["canonical_name"] for t in tests}:
                raise ValueError("Corrections must refer to extracted tests in this report.")
            result["reviewed_values"] = {t["canonical_name"]: corrections.get(t["canonical_name"], result.get("reviewed_values", {}).get(t["canonical_name"], t["value"])) for t in tests}
            c.execute("UPDATE uploads SET review_status='reviewed', result_json=? WHERE id=? AND user_id=?",
                      (json.dumps(result, allow_nan=False), record_id, user_id))
            self._audit(c, user_id, "record.reviewed", record_id)

    def audit(self, user_id: str) -> list[dict[str, Any]]:
        with self.store._connection() as c:
            return [dict(r) for r in c.execute("""SELECT action, resource_id, occurred_at FROM audit_events
                WHERE user_id=? ORDER BY id DESC LIMIT 100""", (user_id,))]
