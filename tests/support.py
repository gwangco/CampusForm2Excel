"""시험용 합성 사진 도구.

여기서 만드는 이미지는 기능 계약을 확인하기 위한 것이다. 글꼴로 그린 글자이므로
손글씨가 아니며, 개발 18장·최종 36장·외부 10장 데이터에 포함하지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Photo:
    image: np.ndarray
    matrix: np.ndarray  # 정규 페이지 좌표를 사진 좌표로 옮기는 행렬


def simulate_photo(page: np.ndarray, seed: int = 0, canvas: tuple[int, int] = (2000, 2700),
                   jitter: float = 0.035, max_angle: float = 4.0) -> Photo:
    """책상 위에서 비스듬히 찍은 사진처럼 만든다."""
    rng = np.random.default_rng(seed)
    height, width = page.shape[:2]
    canvas_w, canvas_h = canvas

    scale = 0.9 * min(canvas_w / width, canvas_h / height)
    center = np.array([canvas_w / 2, canvas_h / 2])
    half = np.array([width, height]) * scale / 2
    corners = np.array([center - half, center + [half[0], -half[1]], center + half, center + [-half[0], half[1]]])

    angle = np.radians(rng.uniform(-max_angle, max_angle))
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    corners = (corners - center) @ rotation.T + center
    corners += rng.uniform(-jitter, jitter, (4, 2)) * [canvas_w, canvas_h] * 0.5

    source = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(source, corners.astype(np.float32))
    photo = cv2.warpPerspective(page, matrix, canvas, flags=cv2.INTER_AREA, borderValue=(72, 64, 58))

    # 조명 기울기와 촬영 노이즈
    light = np.linspace(0.86, 1.0, canvas_w, dtype=np.float32)[None, :, None]
    noisy = photo.astype(np.float32) * light + rng.normal(0, 3.0, photo.shape)
    return Photo(np.clip(noisy, 0, 255).astype(np.uint8), matrix)


def quad_iou(first: np.ndarray, second: np.ndarray) -> float:
    """두 볼록 사각형의 IoU."""
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    overlap, _ = cv2.intersectConvexConvex(first, second)
    union = cv2.contourArea(first) + cv2.contourArea(second) - overlap
    return float(overlap / union) if union > 0 else 0.0
