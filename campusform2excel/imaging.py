"""이미지 입출력. cv2.imread·imwrite는 Windows 한글 경로를 열지 못하므로 쓰지 않는다."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


def decode_image(data: bytes) -> np.ndarray:
    """파일 바이트를 BGR 배열로 읽는다. 휴대폰 사진의 EXIF 회전을 반영한다."""
    with Image.open(BytesIO(data)) as image:
        upright = ImageOps.exif_transpose(image).convert("RGB")
    return cv2.cvtColor(np.asarray(upright), cv2.COLOR_RGB2BGR)


def read_image(path: Path) -> np.ndarray:
    return decode_image(Path(path).read_bytes())


def write_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise OSError(f"이미지를 저장하지 못했습니다: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded.tobytes())


def to_rgb(image: np.ndarray) -> np.ndarray:
    """화면 표시용으로 BGR을 RGB로 바꾼다."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
