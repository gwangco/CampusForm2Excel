"""업로드 파일 검사 (FR-03). JPG·PNG, 1개당 20MB 이하, 한 작업 최대 10장."""
from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import PurePath

from PIL import Image

MAX_FILES = 10
MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 100_000_000
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}


@dataclass(frozen=True)
class Upload:
    name: str
    data: bytes


@dataclass(frozen=True)
class Rejection:
    name: str
    reason: str


@dataclass
class IntakeResult:
    accepted: list[Upload] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.accepted) and not self.rejected


def _problem(upload: Upload) -> str | None:
    if PurePath(upload.name).suffix.lower() not in ALLOWED_EXTENSIONS:
        return "JPG·PNG 파일만 올릴 수 있습니다"
    if len(upload.data) > MAX_BYTES:
        return "파일 크기가 20MB를 넘습니다"
    try:
        with Image.open(BytesIO(upload.data)) as image:
            if image.format not in ALLOWED_FORMATS:
                return "파일 내용이 JPG·PNG가 아닙니다"
            if image.width * image.height > MAX_PIXELS:
                return "이미지 해상도가 너무 큽니다"
            image.load()
    except Exception:  # Pillow는 손상 유형에 따라 여러 예외를 던진다.
        return "손상되었거나 열 수 없는 파일입니다"
    return None


def validate_uploads(uploads: list[Upload]) -> IntakeResult:
    """파일을 검사한다. 장수를 넘으면 작업 전체를 거절한다."""
    if len(uploads) > MAX_FILES:
        reason = f"한 작업에는 최대 {MAX_FILES}장까지 올릴 수 있습니다 (현재 {len(uploads)}장)"
        return IntakeResult(rejected=[Rejection(upload.name, reason) for upload in uploads])

    result = IntakeResult()
    for upload in uploads:
        problem = _problem(upload)
        if problem:
            result.rejected.append(Rejection(upload.name, problem))
        else:
            result.accepted.append(upload)
    return result
