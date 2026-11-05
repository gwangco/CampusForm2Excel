"""크기·노이즈 보정 (FR-04). 기울기는 alignment 모듈이 표 네 모서리로 함께 보정한다."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

MAX_LONG_SIDE = 3200
MIN_SHORT_SIDE = 900
# 임시값이다. 개발 18장을 촬영한 뒤 11월 4주에 확정한다.
MIN_SHARPNESS = 30.0


class PreprocessError(Exception):
    """이미지를 보정할 수 없을 때 발생한다. message는 사용자에게 그대로 보여 준다."""


@dataclass
class Normalized:
    image: np.ndarray
    scale: float  # 원본 좌표 × scale = 보정 이미지 좌표
    sharpness: float
    warnings: list[str] = field(default_factory=list)


def normalize(image: np.ndarray) -> Normalized:
    """원본은 건드리지 않고 크기와 노이즈를 보정한 새 이미지를 돌려준다."""
    height, width = image.shape[:2]
    if min(height, width) < MIN_SHORT_SIDE:
        raise PreprocessError(
            f"해상도가 낮습니다({width}×{height}). 짧은 변이 {MIN_SHORT_SIDE}픽셀 이상이 되도록 다시 촬영해 주세요."
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    warnings = []
    if sharpness < MIN_SHARPNESS:
        warnings.append("사진이 흐릿합니다. 인식이 어려우면 다시 촬영해 주세요.")

    scale = min(1.0, MAX_LONG_SIDE / max(height, width))
    resized = image
    if scale < 1.0:
        resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # 획의 경계를 살리면서 촬영 노이즈만 줄인다.
    denoised = cv2.bilateralFilter(resized, d=5, sigmaColor=30, sigmaSpace=5)
    return Normalized(denoised, scale, sharpness, warnings)
