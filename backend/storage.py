from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sqlite3
import unicodedata
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4


PASSWORD_ITERATIONS = 600_000
SESSION_DAYS = 14
ALLOWED_RECORD_KINDS = {
    "scrna_csv",
    "lab_report",
    "medical_image",
    "risk_assessment",
    "future_risk_assessment",
}


class StorageError(Exception):
    pass


class DuplicateEmailError(StorageError):
    pass


class UserDataStore:
    def __init__(self, data_dir: Path | str):
        self.data_dir = Path(data_dir).expanduser().resolve()
        self.database_path = self.data_dir / "pulmoai.sqlite3"
        self.files_dir = self.data_dir / "uploads"

    def initialize(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.files_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.data_dir, 0o700)
        os.chmod(self.files_dir, 0o700)
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_salt TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS sessions_user_idx ON sessions(user_id);
                CREATE TABLE IF NOT EXISTS uploads (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                    kind TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    stored_filename TEXT,
                    size_bytes INTEGER NOT NULL,
                    result_json TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    uploaded_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS uploads_user_date_idx
                    ON uploads(user_id, uploaded_at DESC);
                """
            )
            from backend.workspace_storage import migrate_workspace
            migrate_workspace(connection)
        if self.database_path.exists():
            os.chmod(self.database_path, 0o600)

    def register(self, full_name: str, email: str, password: str) -> dict[str, str]:
        name = " ".join(full_name.split())
        normalized_email = email.strip().casefold()
        if not name or len(name) > 120:
            raise ValueError("Enter a name up to 120 characters.")
        if len(normalized_email) > 254 or not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", normalized_email):
            raise ValueError("Enter a valid email address.")
        if len(password) < 8 or len(password) > 128:
            raise ValueError("Password must be between 8 and 128 characters.")

        user_id = uuid4().hex
        salt = secrets.token_bytes(16)
        password_hash = self._password_hash(password, salt).hex()
        created_at = self._now()
        try:
            with self._connection() as connection:
                connection.execute(
                    "INSERT INTO users (id, full_name, email, password_salt, password_hash, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (user_id, name, normalized_email, salt.hex(), password_hash, created_at),
                )
        except sqlite3.IntegrityError as error:
            raise DuplicateEmailError("An account with this email already exists. Sign in instead.") from error
        return {"id": user_id, "full_name": name, "email": normalized_email, "created_at": created_at}

    def authenticate(self, email: str, password: str) -> dict[str, str] | None:
        normalized_email = email.strip().casefold()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT id, full_name, email, password_salt, password_hash, created_at "
                "FROM users WHERE email = ? COLLATE NOCASE",
                (normalized_email,),
            ).fetchone()
        if row is None:
            self._password_hash(password, bytes(16))
            return None
        candidate = self._password_hash(password, bytes.fromhex(row["password_salt"])).hex()
        if not secrets.compare_digest(candidate, row["password_hash"]):
            return None
        return self._public_user(row)

    def create_session(self, user_id: str) -> tuple[str, str]:
        token = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        expires_at = (now + timedelta(days=SESSION_DAYS)).isoformat(timespec="seconds")
        with self._connection() as connection:
            connection.execute(
                "INSERT INTO sessions (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (self._token_hash(token), user_id, now.isoformat(timespec="seconds"), expires_at),
            )
        return token, expires_at

    def user_for_session(self, token: str | None) -> dict[str, str] | None:
        if not token:
            return None
        token_hash = self._token_hash(token)
        now = self._now()
        with self._connection() as connection:
            row = connection.execute(
                "SELECT users.id, users.full_name, users.email, users.created_at, sessions.expires_at "
                "FROM sessions JOIN users ON users.id = sessions.user_id WHERE sessions.token_hash = ?",
                (token_hash,),
            ).fetchone()
            if row is None:
                return None
            if row["expires_at"] <= now:
                connection.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
                return None
        return self._public_user(row)

    def revoke_session(self, token: str | None) -> None:
        if not token:
            return
        with self._connection() as connection:
            connection.execute("DELETE FROM sessions WHERE token_hash = ?", (self._token_hash(token),))

    def save_record(
        self,
        user_id: str,
        kind: str,
        original_filename: str,
        content_type: str,
        result: dict[str, Any],
        summary: str,
        contents: bytes | None = None,
    ) -> dict[str, Any]:
        if kind not in ALLOWED_RECORD_KINDS:
            raise ValueError("Unsupported stored record kind.")
        result_json = json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        record_id = uuid4().hex
        safe_name = self._safe_display_filename(original_filename)
        stored_filename = None
        size_bytes = len(contents) if contents is not None else 0
        target_path: Path | None = None

        if contents is not None:
            user_dir = self.files_dir / user_id
            user_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(user_dir, 0o700)
            suffix = Path(safe_name).suffix.lower()
            if not re.fullmatch(r"\.[a-z0-9]{1,10}", suffix):
                suffix = ""
            stored_filename = f"{record_id}{suffix}"
            target_path = user_dir / stored_filename
            with target_path.open("xb") as output:
                os.chmod(target_path, 0o600)
                output.write(contents)

        uploaded_at = self._now()
        try:
            with self._connection() as connection:
                connection.execute(
                    "INSERT INTO uploads (id, user_id, kind, original_filename, content_type, stored_filename, "
                    "size_bytes, result_json, summary, uploaded_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        record_id,
                        user_id,
                        kind,
                        safe_name,
                        content_type or "application/octet-stream",
                        stored_filename,
                        size_bytes,
                        result_json,
                        summary[:240],
                        uploaded_at,
                    ),
                )
                connection.execute("INSERT INTO audit_events(user_id, action, resource_id, occurred_at) VALUES (?, ?, ?, ?)",
                                   (user_id, "record.created", record_id, uploaded_at))
        except Exception:
            if target_path is not None:
                target_path.unlink(missing_ok=True)
            raise
        return {
            "id": record_id,
            "kind": kind,
            "filename": safe_name,
            "content_type": content_type or "application/octet-stream",
            "size_bytes": size_bytes,
            "summary": summary[:240],
            "uploaded_at": uploaded_at,
            "has_file": stored_filename is not None,
        }

    def list_records(self, user_id: str) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT * "
                "FROM uploads WHERE user_id = ? ORDER BY uploaded_at DESC, id DESC",
                (user_id,),
            ).fetchall()
        return [self._public_record(row) for row in rows]

    def get_record(self, user_id: str, record_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT * "
                "FROM uploads WHERE id = ? AND user_id = ?",
                (record_id, user_id),
            ).fetchone()
        if row is None:
            return None
        record = self._public_record(row)
        record["result"] = json.loads(row["result_json"])
        return record

    def get_record_file(self, user_id: str, record_id: str) -> tuple[Path, dict[str, Any]] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT id, original_filename, content_type, stored_filename FROM uploads "
                "WHERE id = ? AND user_id = ?",
                (record_id, user_id),
            ).fetchone()
        if row is None or not row["stored_filename"]:
            return None
        path = (self.files_dir / user_id / row["stored_filename"]).resolve()
        if self.files_dir.resolve() not in path.parents or not path.is_file():
            return None
        return path, {
            "filename": row["original_filename"],
            "content_type": row["content_type"],
        }

    def delete_record(self, user_id: str, record_id: str) -> bool:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT stored_filename FROM uploads WHERE id = ? AND user_id = ?",
                (record_id, user_id),
            ).fetchone()
            if row is None:
                return False
            if row["stored_filename"]:
                target = self.files_dir / user_id / row["stored_filename"]
                target.unlink(missing_ok=True)
            connection.execute("DELETE FROM uploads WHERE id = ? AND user_id = ?", (record_id, user_id))
            connection.execute("INSERT INTO audit_events(user_id, action, resource_id, occurred_at) VALUES (?, ?, ?, ?)",
                               (user_id, "record.deleted", record_id, self._now()))
        return True

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=15000")
        return connection

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _password_hash(password: str, salt: bytes) -> bytes:
        return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS)

    @staticmethod
    def _public_user(row: sqlite3.Row) -> dict[str, str]:
        return {
            "id": row["id"],
            "full_name": row["full_name"],
            "email": row["email"],
            "created_at": row["created_at"],
        }

    @staticmethod
    def _safe_display_filename(filename: str) -> str:
        normalized = unicodedata.normalize("NFKC", str(filename)).replace("\\", "/")
        name = normalized.rsplit("/", 1)[-1]
        name = "".join(character for character in name if character.isprintable() and character not in "/\\")
        name = name.strip().strip(".")[:180]
        return name or "uploaded-file"

    @staticmethod
    def _public_record(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "kind": row["kind"],
            "filename": row["original_filename"],
            "content_type": row["content_type"],
            "size_bytes": row["size_bytes"],
            "summary": row["summary"],
            "uploaded_at": row["uploaded_at"],
            "has_file": row["stored_filename"] is not None,
            **{key: row[key] for key in ("patient_id", "observed_on", "series_label", "review_status") if key in row.keys()},
        }
