"""표 영역을 찾아 정규 페이지로 펴고 셀을 페이지·행·열에 대응시킨다 (FR-04, FR-06).

처리 순서
  1. 이미지에서 표 후보(가장 큰 연결 성분들)의 네 모서리를 찾는다.
  2. 원근 변환으로 양식의 정규 좌표에 맞춘다. 기울기와 원근이 여기서 함께 보정된다.
  3. 펴진 이미지에서 실제 격자선을 찾아 양식의 격자선과 비교한다.
     일치도가 낮으면 정합 실패로 처리하므로 다른 양식을 고른 경우도 여기서 걸러진다.
  4. 찾은 격자선으로 셀 좌표를 미세 조정하고, 셀마다 원본 사진의 위치를 함께 기록한다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .templates import Template

DETECT_LONG_SIDE = 1400  # 표 후보 탐색용 축소 크기
MIN_TABLE_AREA_RATIO = 0.08  # 이미지 대비 표 후보의 최소 면적
MAX_CANDIDATES = 4
LINE_KERNEL_PX = 60  # 이보다 긴 직선만 격자선으로 본다 (손글씨 획 제외)
LINE_MIN_COVER = 0.5  # 표 길이의 절반 이상 이어져야 격자선이다
EDGE_TOLERANCE_PX = 12  # 200dpi 기준 약 1.5mm
PASS_GRID_SCORE = 0.8
ROTATIONS = (0, 90, 270, 180)
_CV_ROTATE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


class AlignmentError(Exception):
    """셀 정합 실패. message는 사용자에게 그대로 보여 준다."""

    def __init__(self, code: str, message: str, grid_score: float | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.grid_score = grid_score


@dataclass(frozen=True)
class CellBox:
    cell_id: str
    page: int
    row: int
    col: int
    column_key: str
    column_type: str
    bbox: tuple[int, int, int, int]  # 보정 이미지 좌표
    source_quad: list[list[float]]  # 업로드 원본의 네 꼭짓점 좌표


@dataclass
class AlignedPage:
    image: np.ndarray
    grid_score: float
    rotation: int
    skew_deg: float
    cells: list[CellBox]


@dataclass
class _Fit:
    score: float
    rotation: int
    quad: np.ndarray
    homography: np.ndarray
    x_edges: list[int]
    y_edges: list[int]


def make_cell_id(page: int, row: int, col: int) -> str:
    return f"p{page:02d}-r{row:02d}-c{col:02d}"


def _order_corners(points: np.ndarray) -> np.ndarray:
    """꼭짓점을 왼쪽 위부터 시계 방향으로 정렬한다."""
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    return np.roll(ordered, -start, axis=0).astype(np.float32)


def _hull_to_quad(hull: np.ndarray) -> np.ndarray:
    perimeter = cv2.arcLength(hull, True)
    for ratio in (0.01, 0.02, 0.03, 0.05, 0.08):
        approx = cv2.approxPolyDP(hull, ratio * perimeter, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    # 손글씨가 테두리 밖으로 나가 다각형이 단순해지지 않으면 네 방향의 극점을 쓴다.
    points = hull.reshape(-1, 2)
    sums = points.sum(axis=1)
    diffs = points[:, 0] - points[:, 1]
    corners = [points[sums.argmin()], points[diffs.argmax()], points[sums.argmax()], points[diffs.argmin()]]
    return np.array(corners, dtype=np.float32)


def _candidate_quads(gray: np.ndarray) -> list[np.ndarray]:
    """표일 가능성이 높은 네 모서리 후보를 큰 순서로 돌려준다."""
    scale = min(1.0, DETECT_LONG_SIDE / max(gray.shape))
    small = gray if scale == 1.0 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    binary = cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 10)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    box_areas = stats[:, cv2.CC_STAT_WIDTH] * stats[:, cv2.CC_STAT_HEIGHT]
    order = sorted(range(1, count), key=lambda index: box_areas[index], reverse=True)

    quads = []
    for index in order[:MAX_CANDIDATES]:
        if box_areas[index] < MIN_TABLE_AREA_RATIO * small.size:
            break
        ys, xs = np.nonzero(labels == index)
        hull = cv2.convexHull(np.column_stack((xs, ys)).astype(np.int32))
        quads.append(_order_corners(_hull_to_quad(hull)) / scale)
    return quads


def _line_positions(profile: np.ndarray, window: int = 3) -> list[float]:
    """투영 프로파일에서 격자선 중심 위치를 찾는다."""
    # 살짝 기운 선은 여러 칸에 나뉘므로 주변 칸을 합쳐서 판단한다.
    smoothed = np.minimum(np.convolve(profile, np.ones(2 * window + 1), mode="same"), 1.0)
    above = smoothed >= LINE_MIN_COVER
    positions = []
    start = None
    for index, flag in enumerate(np.append(above, False)):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            positions.append((start + index - 1) / 2)
            start = None
    return positions


def _match_edges(expected: list[int], found: list[float]) -> tuple[list[int], float]:
    """양식의 격자선과 찾은 격자선을 맞춘다. 없는 선과 남는 선은 모두 감점이다."""
    snapped = []
    used: set[int] = set()
    for edge in expected:
        near = [(abs(pos - edge), index) for index, pos in enumerate(found)
                if index not in used and abs(pos - edge) <= EDGE_TOLERANCE_PX]
        if near:
            _, index = min(near)
            used.add(index)
            snapped.append(int(round(found[index])))
        else:
            snapped.append(edge)
    extra = len(found) - len(used)
    return snapped, len(used) / (len(expected) + extra)


def _fit_grid(page_gray: np.ndarray, template: Template) -> tuple[float, list[int], list[int]]:
    x0, y0, x1, y1 = template.table_bbox
    pad = EDGE_TOLERANCE_PX
    left, top = max(x0 - pad, 0), max(y0 - pad, 0)
    region = page_gray[top:y1 + pad, left:x1 + pad]

    binary = cv2.adaptiveThreshold(region, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 12)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((LINE_KERNEL_PX, 1), np.uint8))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((1, LINE_KERNEL_PX), np.uint8))

    x_found = [pos + left for pos in _line_positions(vertical.sum(axis=0) / 255 / (y1 - y0))]
    y_found = [pos + top for pos in _line_positions(horizontal.sum(axis=1) / 255 / (x1 - x0))]

    x_edges, x_score = _match_edges(template.x_edges, x_found)
    y_edges, y_score = _match_edges(template.y_edges, y_found)
    return min(x_score, y_score), x_edges, y_edges


def _rotate(image: np.ndarray, rotation: int) -> np.ndarray:
    return image if rotation == 0 else cv2.rotate(image, _CV_ROTATE[rotation])


def _rotation_matrix(rotation: int, width: int, height: int) -> np.ndarray:
    """회전 전 좌표를 회전 후 좌표로 옮기는 행렬."""
    matrices = {
        0: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        90: [[0, -1, height - 1], [1, 0, 0], [0, 0, 1]],
        180: [[-1, 0, width - 1], [0, -1, height - 1], [0, 0, 1]],
        270: [[0, 1, 0], [-1, 0, width - 1], [0, 0, 1]],
    }
    return np.array(matrices[rotation], dtype=np.float64)


def _best_fit(image: np.ndarray, template: Template) -> _Fit | None:
    x0, y0, x1, y1 = template.table_bbox
    target = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    best: _Fit | None = None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    for rotation in ROTATIONS:
        rotated = _rotate(gray, rotation)
        for quad in _candidate_quads(rotated):
            homography = cv2.getPerspectiveTransform(quad, target)
            flat = cv2.warpPerspective(rotated, homography, template.page_size, borderValue=255)
            score, x_edges, y_edges = _fit_grid(flat, template)
            if best is None or score > best.score:
                best = _Fit(score, rotation, quad, homography, x_edges, y_edges)
            if score >= PASS_GRID_SCORE:
                return best
    return best


def align_page(image: np.ndarray, template: Template, page: int, scale: float = 1.0) -> AlignedPage:
    """보정 이미지를 양식에 맞춘다. scale은 원본 좌표에서 image 좌표로의 배율이다."""
    fit = _best_fit(image, template)
    if fit is None:
        raise AlignmentError(
            "table_not_found",
            "표를 찾지 못했습니다. 표의 네 모서리가 모두 보이도록 정면에서 다시 촬영해 주세요.",
        )
    if fit.score < PASS_GRID_SCORE:
        raise AlignmentError(
            "grid_mismatch",
            f"표 격자가 '{template.label}'와 맞지 않습니다 (일치도 {fit.score:.2f}). "
            "양식 선택을 확인하거나 다시 촬영해 주세요.",
            grid_score=fit.score,
        )

    rotated = _rotate(image, fit.rotation)
    flat = cv2.warpPerspective(
        rotated, fit.homography, template.page_size, flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255)
    )

    height, width = image.shape[:2]
    to_source = np.linalg.inv(
        fit.homography @ _rotation_matrix(fit.rotation, width, height) @ np.diag([scale, scale, 1.0])
    )
    top_edge = fit.quad[1] - fit.quad[0]
    skew = math.degrees(math.atan2(float(top_edge[1]), float(top_edge[0])))
    return AlignedPage(flat, fit.score, fit.rotation, skew, _cell_boxes(template, fit, page, to_source))


def _cell_boxes(template: Template, fit: _Fit, page: int, to_source: np.ndarray) -> list[CellBox]:
    specs = template.cells(fit.x_edges, fit.y_edges)
    corners = np.array(
        [[(s.bbox[0], s.bbox[1]), (s.bbox[2], s.bbox[1]), (s.bbox[2], s.bbox[3]), (s.bbox[0], s.bbox[3])]
         for s in specs],
        dtype=np.float64,
    )
    source = cv2.perspectiveTransform(corners.reshape(1, -1, 2), to_source).reshape(-1, 4, 2)
    return [
        CellBox(make_cell_id(page, spec.row, spec.col), page, spec.row, spec.col, spec.column_key,
                spec.column_type, spec.bbox, np.round(quad, 1).tolist())
        for spec, quad in zip(specs, source)
    ]


def crop_cell(image: np.ndarray, bbox: tuple[int, int, int, int], margin: int = 0) -> np.ndarray:
    """셀 영역을 잘라 낸다. margin만큼 안쪽으로 줄여 격자선을 제외한다."""
    x0, y0, x1, y1 = bbox
    return image[y0 + margin:y1 - margin, x0 + margin:x1 - margin].copy()
