from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path


class VisionImageError(ValueError):
    pass


def prepare_image_base64(image: str | bytes, max_size: int) -> tuple[str, str]:
    mime, image_bytes = _load_image_bytes(image)
    image_bytes = resize_image(image_bytes, mime, max_size)
    return base64.b64encode(image_bytes).decode("ascii"), detect_mime(image_bytes, mime)


def _load_image_bytes(image: str | bytes) -> tuple[str, bytes]:
    if isinstance(image, bytes):
        return detect_mime(image), image
    if image.startswith("data:image/"):
        header, _, data = image.partition(",")
        mime = header.removeprefix("data:").removesuffix(";base64")
        try:
            return mime, base64.b64decode(data, validate=True)
        except ValueError as error:
            raise VisionImageError("image data URL is not valid base64") from error
    file_path = _existing_path(image)
    if file_path is not None:
        image_bytes = file_path.read_bytes()
        return detect_mime(image_bytes, file_path.suffix), image_bytes
    try:
        image_bytes = base64.b64decode(image, validate=True)
    except ValueError as error:
        raise VisionImageError("image must be an existing path, bytes, data URL, or base64 string") from error
    return detect_mime(image_bytes), image_bytes


def resize_image(image_bytes: bytes, mime: str, max_size: int) -> bytes:
    try:
        from PIL import Image
    except ImportError:
        return image_bytes
    with Image.open(BytesIO(image_bytes)) as image:
        width, height = image.size
        if max(width, height) <= max_size:
            return image_bytes
        ratio = max_size / max(width, height)
        size = (max(1, int(width * ratio)), max(1, int(height * ratio)))
        resized = image.resize(size, Image.Resampling.LANCZOS)
        output = BytesIO()
        fmt = "JPEG" if mime == "image/jpeg" else "PNG"
        if fmt == "JPEG" and resized.mode in {"RGBA", "P"}:
            resized = resized.convert("RGB")
        resized.save(output, format=fmt)
        return output.getvalue()


def detect_mime(image_bytes: bytes, fallback: str | None = None) -> str:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"GIF87a") or image_bytes.startswith(b"GIF89a"):
        return "image/gif"
    if fallback in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if fallback == ".png":
        return "image/png"
    if fallback == ".gif":
        return "image/gif"
    if fallback and fallback.startswith("image/"):
        return fallback
    return "image/png"


def _existing_path(image: str) -> Path | None:
    try:
        file_path = Path(image)
        return file_path if file_path.exists() else None
    except OSError:
        return None
