from pathlib import Path

import pytest

from ingestion.storage import ImageStorage, ImageStorageError


def storage(tmp_path: Path) -> ImageStorage:
    return ImageStorage(image_root=tmp_path / "images", db_path=tmp_path / "db" / "image_index.db")


def test_save_image_writes_file_and_indexes_path(tmp_path: Path) -> None:
    image_storage = storage(tmp_path)

    path = image_storage.save_image(
        "img-1",
        b"image-bytes",
        collection="default",
        doc_hash="doc-hash",
        page_num=3,
    )

    assert Path(path).read_bytes() == b"image-bytes"
    assert Path(path).name == "img-1.png"
    assert Path(path).parent == tmp_path / "images" / "default"
    assert image_storage.get_path("img-1") == path
    image = image_storage.get_image("img-1")
    assert image is not None
    assert image["image_id"] == "img-1"
    assert image["file_path"] == path
    assert image["collection"] == "default"
    assert image["doc_hash"] == "doc-hash"
    assert image["page_num"] == 3
    assert image["created_at"]


def test_mapping_persists_in_sqlite_database(tmp_path: Path) -> None:
    first = storage(tmp_path)
    path = first.save_image("img-1", b"image-bytes", collection="docs")

    second = storage(tmp_path)

    assert second.get_path("img-1") == path
    assert (tmp_path / "db" / "image_index.db").exists()


def test_list_images_filters_by_collection(tmp_path: Path) -> None:
    image_storage = storage(tmp_path)
    image_storage.save_image("img-2", b"two", collection="docs", doc_hash="hash-b")
    image_storage.save_image("img-1", b"one", collection="docs", doc_hash="hash-a")
    image_storage.save_image("img-3", b"three", collection="notes", doc_hash="hash-a")

    images = image_storage.list_images(collection="docs")

    assert [image["image_id"] for image in images] == ["img-1", "img-2"]


def test_list_images_filters_by_doc_hash(tmp_path: Path) -> None:
    image_storage = storage(tmp_path)
    image_storage.save_image("img-1", b"one", collection="docs", doc_hash="hash-a")
    image_storage.save_image("img-2", b"two", collection="docs", doc_hash="hash-b")

    images = image_storage.list_images(doc_hash="hash-a")

    assert [image["image_id"] for image in images] == ["img-1"]


def test_save_image_from_file_preserves_source_suffix(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    source.write_bytes(b"jpg-bytes")
    image_storage = storage(tmp_path)

    path = image_storage.save_image("img-1", source, collection="docs")

    assert Path(path).read_bytes() == b"jpg-bytes"
    assert Path(path).suffix == ".jpg"


def test_save_image_updates_existing_mapping(tmp_path: Path) -> None:
    image_storage = storage(tmp_path)
    first_path = image_storage.save_image("img-1", b"one", collection="docs", doc_hash="hash-a", page_num=1)
    second_path = image_storage.save_image("img-1", b"two", collection="notes", doc_hash="hash-b", page_num=2)

    image = image_storage.get_image("img-1")

    assert image is not None
    assert first_path != second_path
    assert Path(second_path).read_bytes() == b"two"
    assert image["file_path"] == second_path
    assert image["collection"] == "notes"
    assert image["doc_hash"] == "hash-b"
    assert image["page_num"] == 2


def test_delete_images_removes_matching_records_and_files(tmp_path: Path) -> None:
    image_storage = storage(tmp_path)
    remove_path = Path(image_storage.save_image("img-1", b"one", collection="docs", doc_hash="hash-a"))
    keep_path = Path(image_storage.save_image("img-2", b"two", collection="docs", doc_hash="hash-b"))

    removed = image_storage.delete_images(collection="docs", doc_hash="hash-a")

    assert removed == 1
    assert image_storage.get_path("img-1") is None
    assert image_storage.get_path("img-2") == str(keep_path)
    assert not remove_path.exists()
    assert keep_path.exists()


def test_empty_image_bytes_raise(tmp_path: Path) -> None:
    with pytest.raises(ImageStorageError, match="empty"):
        storage(tmp_path).save_image("img-1", b"")


def test_invalid_image_id_raises(tmp_path: Path) -> None:
    with pytest.raises(ImageStorageError, match="image_id"):
        storage(tmp_path).save_image("bad/id", b"image")


def test_invalid_collection_raises(tmp_path: Path) -> None:
    with pytest.raises(ImageStorageError, match="collection"):
        storage(tmp_path).save_image("img-1", b"image", collection="bad/name")


def test_missing_source_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ImageStorageError, match="not found"):
        storage(tmp_path).save_image("img-1", tmp_path / "missing.png")


def test_image_storage_can_be_imported_from_package() -> None:
    from ingestion.storage import ImageStorage as ExportedImageStorage

    assert ExportedImageStorage is ImageStorage
