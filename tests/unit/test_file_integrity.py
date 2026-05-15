import hashlib
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from libs.loader.file_integrity import FileIntegrityError, SQLiteIntegrityChecker


def test_compute_sha256_is_stable_for_same_file(tmp_path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello world", encoding="utf-8")
    checker = SQLiteIntegrityChecker(str(tmp_path / "ingestion_history.db"))

    first = checker.compute_sha256(str(file_path))
    second = checker.compute_sha256(str(file_path))

    assert first == second
    assert first == hashlib.sha256(b"hello world").hexdigest()


def test_mark_success_makes_file_hash_skippable(tmp_path) -> None:
    db_path = tmp_path / "data" / "db" / "ingestion_history.db"
    checker = SQLiteIntegrityChecker(str(db_path))
    file_hash = "abc123"

    assert checker.should_skip(file_hash) is False

    checker.mark_success(file_hash, "docs/sample.pdf", file_size=42, chunk_count=3)

    assert checker.should_skip(file_hash) is True
    assert checker.get_record(file_hash)["status"] == "success"
    assert checker.get_record(file_hash)["chunk_count"] == 3


def test_database_file_and_wal_mode_are_created(tmp_path) -> None:
    db_path = tmp_path / "data" / "db" / "ingestion_history.db"
    checker = SQLiteIntegrityChecker(str(db_path))

    checker.mark_success("hash-1", "docs/sample.pdf")

    assert db_path.exists()
    with sqlite3.connect(db_path) as connection:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        indexes = {row[1] for row in connection.execute("PRAGMA index_list('ingestion_history')").fetchall()}

    assert journal_mode == "wal"
    assert "idx_ingestion_history_status" in indexes
    assert "idx_ingestion_history_processed_at" in indexes


def test_mark_failed_records_error_and_does_not_skip(tmp_path) -> None:
    checker = SQLiteIntegrityChecker(str(tmp_path / "ingestion_history.db"))

    checker.mark_failed("hash-1", "parse failed", file_path="docs/bad.pdf", file_size=12)

    record = checker.get_record("hash-1")
    assert checker.should_skip("hash-1") is False
    assert record["status"] == "failed"
    assert record["error_msg"] == "parse failed"
    assert record["file_path"] == "docs/bad.pdf"


def test_mark_success_overwrites_failed_record(tmp_path) -> None:
    checker = SQLiteIntegrityChecker(str(tmp_path / "ingestion_history.db"))

    checker.mark_failed("hash-1", "parse failed")
    checker.mark_success("hash-1", "docs/good.pdf", file_size=30, chunk_count=2)

    record = checker.get_record("hash-1")
    assert checker.should_skip("hash-1") is True
    assert record["status"] == "success"
    assert record["error_msg"] is None
    assert record["file_path"] == "docs/good.pdf"


def test_concurrent_success_writes_are_supported(tmp_path) -> None:
    checker = SQLiteIntegrityChecker(str(tmp_path / "ingestion_history.db"))
    hashes = [f"hash-{index}" for index in range(12)]

    def mark_success(file_hash: str) -> None:
        checker.mark_success(file_hash, f"docs/{file_hash}.pdf", file_size=len(file_hash))

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(mark_success, hashes))

    assert all(checker.should_skip(file_hash) for file_hash in hashes)


def test_validation_errors_are_clear(tmp_path) -> None:
    checker = SQLiteIntegrityChecker(str(tmp_path / "ingestion_history.db"))

    with pytest.raises(FileIntegrityError, match="file not found"):
        checker.compute_sha256(str(tmp_path / "missing.txt"))
    with pytest.raises(FileIntegrityError, match="file_hash"):
        checker.should_skip("")
    with pytest.raises(FileIntegrityError, match="file_path"):
        checker.mark_success("hash-1", "")
    with pytest.raises(FileIntegrityError, match="error_msg"):
        checker.mark_failed("hash-1", "")
