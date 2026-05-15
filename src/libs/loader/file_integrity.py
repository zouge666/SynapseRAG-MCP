from __future__ import annotations

import hashlib
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class FileIntegrityError(RuntimeError):
    pass


class FileIntegrityChecker(ABC):
    @abstractmethod
    def compute_sha256(self, path: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def should_skip(self, file_hash: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def mark_success(self, file_hash: str, file_path: str, file_size: int | None = None, chunk_count: int | None = None) -> None:
        raise NotImplementedError

    @abstractmethod
    def mark_failed(self, file_hash: str, error_msg: str, file_path: str = "", file_size: int | None = None) -> None:
        raise NotImplementedError


class SQLiteIntegrityChecker(FileIntegrityChecker):
    def __init__(self, db_path: str = "data/db/ingestion_history.db", timeout: float = 30.0) -> None:
        self.db_path = Path(db_path)
        self.timeout = timeout
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def compute_sha256(self, path: str) -> str:
        file_path = Path(path)
        if not file_path.is_file():
            raise FileIntegrityError(f"file not found: {path}")
        digest = hashlib.sha256()
        with file_path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def should_skip(self, file_hash: str) -> bool:
        self._validate_hash(file_hash)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM ingestion_history WHERE file_hash = ? AND status = 'success'",
                (file_hash,),
            ).fetchone()
        return row is not None

    def mark_success(self, file_hash: str, file_path: str, file_size: int | None = None, chunk_count: int | None = None) -> None:
        self._validate_hash(file_hash)
        if not isinstance(file_path, str) or not file_path:
            raise FileIntegrityError("file_path must be a non-empty string")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO ingestion_history "
                "(file_hash, file_path, file_size, status, processed_at, error_msg, chunk_count) "
                "VALUES (?, ?, ?, 'success', CURRENT_TIMESTAMP, NULL, ?) "
                "ON CONFLICT(file_hash) DO UPDATE SET "
                "file_path = excluded.file_path, "
                "file_size = excluded.file_size, "
                "status = excluded.status, "
                "processed_at = CURRENT_TIMESTAMP, "
                "error_msg = NULL, "
                "chunk_count = excluded.chunk_count",
                (file_hash, file_path, file_size, chunk_count),
            )

    def mark_failed(self, file_hash: str, error_msg: str, file_path: str = "", file_size: int | None = None) -> None:
        self._validate_hash(file_hash)
        if not isinstance(error_msg, str) or not error_msg:
            raise FileIntegrityError("error_msg must be a non-empty string")
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO ingestion_history "
                "(file_hash, file_path, file_size, status, processed_at, error_msg, chunk_count) "
                "VALUES (?, ?, ?, 'failed', CURRENT_TIMESTAMP, ?, NULL) "
                "ON CONFLICT(file_hash) DO UPDATE SET "
                "file_path = excluded.file_path, "
                "file_size = excluded.file_size, "
                "status = excluded.status, "
                "processed_at = CURRENT_TIMESTAMP, "
                "error_msg = excluded.error_msg, "
                "chunk_count = NULL",
                (file_hash, file_path, file_size, error_msg),
            )

    def get_record(self, file_hash: str) -> dict[str, Any] | None:
        self._validate_hash(file_hash)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT file_hash, file_path, file_size, status, processed_at, error_msg, chunk_count "
                "FROM ingestion_history "
                "WHERE file_hash = ?",
                (file_hash,),
            ).fetchone()
        return dict(row) if row is not None else None

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS ingestion_history ("
                "file_hash TEXT PRIMARY KEY, "
                "file_path TEXT NOT NULL, "
                "file_size INTEGER, "
                "status TEXT NOT NULL CHECK(status IN ('success', 'failed', 'processing')), "
                "processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
                "error_msg TEXT, "
                "chunk_count INTEGER"
                ")"
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_ingestion_history_status ON ingestion_history(status)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_ingestion_history_processed_at ON ingestion_history(processed_at)")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=self.timeout)
        connection.row_factory = sqlite3.Row
        return connection

    def _validate_hash(self, file_hash: str) -> None:
        if not isinstance(file_hash, str) or not file_hash:
            raise FileIntegrityError("file_hash must be a non-empty string")
