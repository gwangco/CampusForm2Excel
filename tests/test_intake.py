"""T-03 업로드 검사."""
from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from campusform2excel.intake import MAX_BYTES, MAX_FILES, Upload, validate_uploads


def encoded(file_format: str = "PNG") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (64, 64), "white").save(buffer, format=file_format)
    return buffer.getvalue()


def test_t03_ten_files_accepted():
    uploads = [Upload(f"p{i}.png", encoded()) for i in range(MAX_FILES)]
    result = validate_uploads(uploads)
    assert result.ok and len(result.accepted) == MAX_FILES


def test_t03_eleven_files_rejected_as_a_whole():
    result = validate_uploads([Upload(f"p{i}.png", encoded()) for i in range(MAX_FILES + 1)])
    assert not result.ok and not result.accepted
    assert "최대 10장" in result.rejected[0].reason


def test_t03_exact_limit_allowed_and_one_byte_over_rejected():
    image = encoded()
    at_limit = image + b"\0" * (MAX_BYTES - len(image))
    assert validate_uploads([Upload("limit.png", at_limit)]).ok

    result = validate_uploads([Upload("over.png", at_limit + b"\0")])
    assert not result.ok and "20MB" in result.rejected[0].reason


@pytest.mark.parametrize(
    ("name", "data", "message"),
    [
        ("broken.png", encoded()[:40], "손상"),
        ("broken.jpg", encoded("JPEG")[:60], "손상"),
        ("noise.png", b"not an image", "손상"),
        ("empty.png", b"", "손상"),
        ("fake.png", encoded("GIF"), "JPG·PNG가 아닙니다"),
        ("scan.pdf", b"%PDF-1.7", "JPG·PNG 파일만"),
        ("phone.heic", b"....", "JPG·PNG 파일만"),
        ("doc.hwp", b"....", "JPG·PNG 파일만"),
    ],
)
def test_t03_invalid_files_rejected(name, data, message):
    result = validate_uploads([Upload(name, data)])
    assert not result.ok and message in result.rejected[0].reason


def test_t03_uppercase_extension_and_jpeg_accepted():
    assert validate_uploads([Upload("PHOTO.JPG", encoded("JPEG")), Upload("a.jpeg", encoded("JPEG"))]).ok


def test_t03_one_bad_file_is_reported_with_the_good_one_kept():
    result = validate_uploads([Upload("good.png", encoded()), Upload("bad.png", b"x")])
    assert not result.ok
    assert [item.name for item in result.accepted] == ["good.png"]
    assert [item.name for item in result.rejected] == ["bad.png"]


def test_t03_nothing_uploaded_is_not_ok():
    assert not validate_uploads([]).ok
