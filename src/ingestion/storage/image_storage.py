from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from typing import Any


class ImageStorageError(ValueError):
    pass


class ImageStorage:
    def __init__(
        self,
        image_root: str | Path = "data/images",
        db_path: str | Path = "data/db/image_index.db",
        timeout: float = 30.0,
    ) -> None:
        self.image_root = Path(image_root)
        self.db_path = Path(db_path)
        self.timeout = timeout
        self.image_root.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def save_image(
        self,
        image_id: str,
        image: bytes | str | Path,
        collection: str = "default",
        doc_hash: str = "",
        page_num: int | None = None,
        extension: str = ".png",
    ) -> str:
        self._validate_image_id(image_id)
        self._validate_collection(collection)
        suffix = self._extension(extension, image)
        collection_dir = self.image_root / collection
        collection_dir.mkdir(parents=True, exist_ok=True)
        file_path = collection_dir / f"{image_id}{suffix}"
        if isinstance(image, bytes):
            if not image:
                raise ImageStorageError("image bytes must not be empty")
            file_path.write_bytes(image)
        else:
            source_path = Path(image)
            if not source_path.is_file():
                raise ImageStorageError(f"image file not found: {image}")
            shutil.copyfile(source_path, file_path)
        self._upsert_index(image_id, str(file_path), collection, doc_hash, page_num)
        return str(file_path)

    def get_path(self, image_id: str) -> str | None:
        self._validate_image_id(image_id)
        with self._connect() as connection:
            row = connection.execute("SELECT file_path FROM image_index WHERE image_id = ?", (image_id,)).fetchone()
        return str(row["file_path"]) if row is not None else None

    def get_image(self, image_id: str) -> dict[str, Any] | None:
        self._validate_image_id(image_id)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT image_id, file_path, collection, doc_hash, page_num, created_at FROM image_index WHERE image_id = ?",
                (image_id,),
            ).fetchone()
        return dict(row) if row is not None else None

    def list_images(self, collection: str | None = None, doc_hash: str | None = None) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if collection is not None:
            clauses.append("collection = ?")
            params.append(collection)
        if doc_hash is not None:
            clauses.append("doc_hash = ?")
            params.append(doc_hash)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT image_id, file_path, collection, doc_hash, page_num, created_at FROM image_index"
                f"{where} ORDER BY image_id",
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_images(self, collection: str | None = None, doc_hash: str | None = None, delete_files: bool = True) -> int:
        images = self.list_images(collection=collection, doc_hash=doc_hash)
        if not images:
            return 0
        ids = [image["image_id"] for image in images]
        placeholders = ",".join("?" for _ in ids)
        with self._connect() as connection:
            connection.execute(f"DELETE FROM image_index WHERE image_id IN ({placeholders})", ids)
        if delete_files:
            for image in images:
                path = Path(image["file_path"])
                if path.exists():
                    path.unlink()
        return len(images)

    def _upsert_index(self, image_id: str, file_path: str, collection: str, doc_hash: str, page_num: int | None) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO image_index (image_id, file_path, collection, doc_hash, page_num, created_at) "
                "VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP) "
                "ON CONFLICT(image_id) DO UPDATE SET "
                "file_path = excluded.file_path, "
                "collection = excluded.collection, "
                "doc_hash = excluded.doc_hash, "
                "page_num = excluded.page_num, "
                "created_at = CURRENT_TIMESTAMP",
                (image_id, file_path, collection, doc_hash, page_num),
            )

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS image_index ("
                "image_id TEXT PRIMARY KEY, "
                "file_path TEXT NOT NULL, "
                "collection TEXT, "
                "doc_hash TEXT, "
                "page_num INTEGER, "
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ")"
            )
            connection.execute("CREATE INDEX IF NOT EXISTS idx_collection ON image_index(collection)")
            connection.execute("CREATE INDEX IF NOT EXISTS idx_doc_hash ON image_index(doc_hash)")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=self.timeout)
        connection.row_factory = sqlite3.Row
        return connection

    def _extension(self, extension: str, image: bytes | str | Path) -> str:
        if isinstance(image, (str, Path)):
            suffix = Path(image).suffix
            if suffix:
                return suffix.lower()
        if not isinstance(extension, str) or not extension:
            raise ImageStorageError("extension must be a non-empty string")
        return extension if extension.startswith(".") else f".{extension}"

    def _validate_image_id(self, image_id: str) -> None:
        if not isinstance(image_id, str) or not image_id:
            raise ImageStorageError("image_id must be a non-empty string")
        if "/" in image_id or "\\" in image_id:
            raise ImageStorageError("image_id must not contain path separators")

    def _validate_collection(self, collection: str) -> None:
        if not isinstance(collection, str) or not collection:
            raise ImageStorageError("collection must be a non-empty string")
        if "/" in collection or "\\" in collection:
            raise ImageStorageError("collection must not contain path separators")
