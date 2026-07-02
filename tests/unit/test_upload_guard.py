import pytest

from observability.dashboard.services.upload_guard import (
    MAX_PAGES,
    UploadRejected,
    sanitize_display_name,
    save_upload_securely,
    validate_pdf_upload,
)


def make_pdf(pages: int = 1, text: str = "hello") -> bytes:
    import pymupdf

    document = pymupdf.open()
    for _ in range(pages):
        page = document.new_page()
        page.insert_text((72, 72), text)
    data = document.tobytes()
    document.close()
    return data


def test_valid_pdf_accepted() -> None:
    stats = validate_pdf_upload("doc.pdf", make_pdf(pages=2, text="hi"))
    assert stats["pages"] == 2
    assert stats["text_chars"] > 0


def test_rejects_non_pdf_magic() -> None:
    with pytest.raises(UploadRejected, match="Only PDF"):
        validate_pdf_upload("fake.pdf", b"not a pdf at all")


def test_rejects_empty_file() -> None:
    with pytest.raises(UploadRejected, match="empty"):
        validate_pdf_upload("empty.pdf", b"")


def test_rejects_oversized_file() -> None:
    data = b"%PDF-1.4\n" + b"0" * (10 * 1024 * 1024)
    with pytest.raises(UploadRejected, match="10 MB"):
        validate_pdf_upload("big.pdf", data)


def test_rejects_too_many_pages() -> None:
    with pytest.raises(UploadRejected, match=f"{MAX_PAGES}"):
        validate_pdf_upload("long.pdf", make_pdf(pages=MAX_PAGES + 1))


def test_rejects_too_much_text() -> None:
    import pymupdf

    document = pymupdf.open()
    filler = "lorem ipsum dolor sit amet " * 600
    for _ in range(40):
        page = document.new_page()
        page.insert_textbox(page.rect, filler, fontsize=6)
    data = document.tobytes()
    total = 0
    probe = pymupdf.open(stream=data, filetype="pdf")
    for page in probe:
        total += len(page.get_text())
    probe.close()
    document.close()
    assert total > 500_000
    with pytest.raises(UploadRejected, match="too much text"):
        validate_pdf_upload("dense.pdf", data)


def test_sanitize_display_name_strips_traversal() -> None:
    assert sanitize_display_name("../../etc/passwd.pdf") == "passwd.pdf"
    assert sanitize_display_name("..\\..\\win.pdf") == "win.pdf"
    assert sanitize_display_name("正常 文件 (v2).pdf") == "正常_文件_v2_.pdf"


def test_save_upload_uses_random_server_filename(tmp_path) -> None:
    first = save_upload_securely(tmp_path, make_pdf())
    second = save_upload_securely(tmp_path, make_pdf())

    assert first.name != second.name
    assert first.suffix == ".pdf"
    assert first.parent == tmp_path
