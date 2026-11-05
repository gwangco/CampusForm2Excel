"""PaddleOCR 셀 인식 (FR-07, AI-01, AI-02, AI-06).

입력 셀마다 결과를 정확히 하나씩 돌려준다. 빈칸·판독 불가·추론 실패도 결과로 남기며,
어떤 경우에도 셀이 사라지지 않는다. 사전 후보(FR-08)와 표식 판정(FR-09)은 11월 3주에
이 결과의 candidates·mark_state 필드를 채운다.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

import cv2
import numpy as np

from .paths import MODEL_LOCK_PATH, OCR_SETTINGS_PATH, model_home

logger = logging.getLogger(__name__)

STATUS_RECOGNIZED = "recognized"  # 글자를 읽음
STATUS_EMPTY = "empty"  # 빈칸
STATUS_UNREADABLE = "unreadable"  # 잉크는 있으나 글자를 읽지 못함
STATUS_FAILED = "failed"  # 입력 오류나 추론 오류
STATUS_DEFERRED = "deferred"  # 체크 칸: 표식 판정은 11월 3주 구현 범위

REASON_LOW_CONFIDENCE = "low_confidence"
REASON_NO_SCORE = "no_score"
REASON_NO_TEXT = "no_text"
REASON_INFERENCE_ERROR = "inference_error"
REASON_INVALID_IMAGE = "invalid_image"
REASON_INVALID_CELL_ID = "invalid_cell_id"
REASON_DUPLICATE_CELL_ID = "duplicate_cell_id"
REASON_MARK_PENDING = "mark_check_pending"


class ModelError(RuntimeError):
    """모델이 없거나 잠금 파일과 다를 때 발생한다."""


@dataclass
class CellInput:
    cell_id: str
    page: int
    row: int
    col: int
    bbox: tuple[int, int, int, int]
    image: np.ndarray | None
    column_type: str
    template_id: str
    template_version: str
    dictionary_version: str | None = None


@dataclass
class CellResult:
    cell_id: str
    page: int
    row: int
    col: int
    bbox: tuple[int, int, int, int]
    raw_text: str
    confidence: float | None
    status: str
    review_required: bool
    reasons: list[str]
    mark_state: str | None
    candidates: list[str]
    template_id: str
    template_version: str
    dictionary_version: str | None
    model_name: str
    model_sha256: str
    config_version: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Engine(Protocol):
    """줄 이미지 목록을 받아 (문자열, 점수) 목록을 돌려주는 인식 엔진."""

    model_name: str
    model_sha256: str

    def predict(self, images: list[np.ndarray]) -> list[tuple[str, float | None]]: ...


@dataclass
class _Pending:
    index: int
    lines: list[np.ndarray] = field(default_factory=list)


def load_settings() -> dict[str, Any]:
    return json.loads(OCR_SETTINGS_PATH.read_text(encoding="utf-8"))


def load_model_lock() -> dict[str, Any]:
    return json.loads(MODEL_LOCK_PATH.read_text(encoding="utf-8"))


def verify_model() -> str:
    """모델 파일을 잠금 파일과 대조하고 모델 폴더 경로를 돌려준다."""
    lock = load_model_lock()
    model_dir = model_home() / "models" / lock["model_name"]
    for name, expected in lock["files"].items():
        path = model_dir / name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            raise ModelError(
                f"모델 파일을 확인할 수 없습니다({name}). 'python scripts/setup_model.py'를 실행해 주세요."
            )
    return str(model_dir)


class PaddleEngine:
    """로컬 모델을 한 번만 읽어 재사용하는 PaddleOCR 엔진."""

    def __init__(self, settings: dict[str, Any] | None = None):
        settings = settings or load_settings()
        lock = load_model_lock()
        model_dir = verify_model()

        # 모델은 고정된 로컬 경로에서만 읽는다. 실행 중에는 외부 통신을 하지 않는다.
        os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
        os.environ["PADDLE_PDX_CACHE_HOME"] = str(model_home() / "cache")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ.setdefault("GLOG_minloglevel", "2")
        from paddleocr import TextRecognition

        self.model_name = lock["model_name"]
        self.model_sha256 = lock["files"]["inference.pdiparams"]["sha256"]
        self._batch_size = settings["batch_size"]
        self._lock = threading.Lock()  # 추론기는 스레드 안전하지 않고 Streamlit은 세션별 스레드를 쓴다.
        self._model = TextRecognition(
            model_name=self.model_name,
            model_dir=model_dir,
            device="cpu",
            cpu_threads=settings["cpu_threads"],
        )

    def predict(self, images: list[np.ndarray]) -> list[tuple[str, float | None]]:
        with self._lock:
            outputs = list(self._model.predict(input=images, batch_size=self._batch_size))
        return [(output["rec_text"], output["rec_score"]) for output in outputs]


def _strip_edge_lines(dark: np.ndarray, depth: int = 6, cover: float = 0.7) -> None:
    """잘림 경계에 남은 격자선을 지운다.

    두 선이 모서리에서 이어지면 하나의 덩어리가 되므로 행·열 단위로 판단한다.
    """
    height, width = dark.shape
    for row in [*range(min(depth, height)), *range(max(height - depth, 0), height)]:
        if dark[row].mean() >= cover:
            dark[row] = 0
    for col in [*range(min(depth, width)), *range(max(width - depth, 0), width)]:
        if dark[:, col].mean() >= cover:
            dark[:, col] = 0


def _ink_mask(gray: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
    """셀에서 글씨로 볼 픽셀을 찾는다. 작은 점과 남은 격자선 조각은 뺀다."""
    blank = settings["blank"]
    dark = (gray < float(np.median(gray)) - blank["dark_delta"]).astype(np.uint8)
    _strip_edge_lines(dark)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= blank["min_component_px"]
    keep[0] = False  # 0번은 배경이다.
    return keep[labels]


def _line_bands(mask: np.ndarray, settings: dict[str, Any]) -> list[tuple[int, int]]:
    """글씨가 있는 세로 구간을 줄별로 나눈다."""
    min_gap = settings["line_split"]["min_gap_px"]
    min_height = settings["line_split"]["min_line_height_px"]
    rows = np.flatnonzero(mask.any(axis=1))

    bands: list[list[int]] = []
    for row in rows:
        if bands and row - bands[-1][1] <= min_gap:
            bands[-1][1] = int(row)
        else:
            bands.append([int(row), int(row)])

    # '를'처럼 한 글자가 위아래로 떨어진 경우를 줄로 오해하지 않도록 낮은 구간을 합친다.
    merged: list[list[int]] = []
    for band in bands:
        previous_short = merged and merged[-1][1] - merged[-1][0] + 1 < min_height
        current_short = band[1] - band[0] + 1 < min_height
        if merged and (previous_short or current_short):
            merged[-1][1] = band[1]
        else:
            merged.append(band)
    return [(top, bottom) for top, bottom in merged]


def extract_lines(image: np.ndarray, settings: dict[str, Any]) -> list[np.ndarray]:
    """셀 이미지를 줄 이미지 목록으로 바꾼다. 빈칸이면 빈 목록을 돌려준다."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = _ink_mask(gray, settings)
    if mask.mean() < settings["blank"]["min_ink_ratio"]:
        return []

    pad = settings["trim_padding_px"]
    height, width = gray.shape
    lines = []
    for top, bottom in _line_bands(mask, settings):
        columns = np.flatnonzero(mask[top:bottom + 1].any(axis=0))
        lines.append(image[
            max(top - pad, 0):min(bottom + pad + 1, height),
            max(int(columns[0]) - pad, 0):min(int(columns[-1]) + pad + 1, width),
        ])
    return lines


def _input_problem(cell: CellInput, seen: set[str]) -> str | None:
    if not cell.cell_id:
        return REASON_INVALID_CELL_ID
    if cell.cell_id in seen:
        return REASON_DUPLICATE_CELL_ID
    seen.add(cell.cell_id)
    image = cell.image
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        return REASON_INVALID_IMAGE
    return None


def _predict_lines(images: list[np.ndarray], engine: Engine) -> list[tuple[str, float | None] | None]:
    """일괄 인식하고, 실패하면 한 줄씩 다시 시도해 문제 있는 줄만 None으로 남긴다."""
    if not images:
        return []
    try:
        outputs = engine.predict(images)
        if len(outputs) == len(images):
            return list(outputs)
        logger.warning("인식 결과 수가 입력과 다릅니다: %d/%d", len(outputs), len(images))
    except Exception:
        logger.exception("일괄 인식 실패. 한 줄씩 다시 시도합니다.")

    results: list[tuple[str, float | None] | None] = []
    for image in images:
        try:
            single = engine.predict([image])
            results.append(single[0] if len(single) == 1 else None)
        except Exception:
            logger.exception("줄 인식 실패")
            results.append(None)
    return results


def _clean_score(score: Any) -> float | None:
    if score is None:
        return None
    value = float(score)
    return None if math.isnan(value) else value


def _assemble(outputs: list[tuple[str, float | None] | None], threshold: float) -> dict[str, Any]:
    """한 셀에 속한 줄의 결과를 합친다. 점수는 가장 낮은 줄을 따른다."""
    if any(output is None for output in outputs):
        return {"status": STATUS_FAILED, "reasons": [REASON_INFERENCE_ERROR]}

    texts = [output[0].strip() for output in outputs]
    scores = [_clean_score(output[1]) for output in outputs]
    text = " ".join(part for part in texts if part)
    if not text:
        return {"status": STATUS_UNREADABLE, "reasons": [REASON_NO_TEXT]}

    reasons = []
    if not all(texts):
        reasons.append(REASON_NO_TEXT)
    confidence = None if any(score is None for score in scores) else min(scores)
    if confidence is None:
        reasons.append(REASON_NO_SCORE)
    elif confidence < threshold:
        reasons.append(REASON_LOW_CONFIDENCE)
    return {"status": STATUS_RECOGNIZED, "raw_text": text, "confidence": confidence, "reasons": reasons}


def _make_result(cell: CellInput, engine: Engine, settings: dict[str, Any], *, status: str,
                 raw_text: str = "", confidence: float | None = None,
                 reasons: list[str] | None = None) -> CellResult:
    reasons = reasons or []
    return CellResult(
        cell_id=cell.cell_id, page=cell.page, row=cell.row, col=cell.col, bbox=cell.bbox,
        raw_text=raw_text, confidence=confidence, status=status,
        review_required=bool(reasons),
        reasons=reasons, mark_state=None, candidates=[],
        template_id=cell.template_id, template_version=cell.template_version,
        dictionary_version=cell.dictionary_version,
        model_name=engine.model_name, model_sha256=engine.model_sha256,
        config_version=settings["config_version"],
    )


def recognize_cells(cells: list[CellInput], engine: Engine,
                    settings: dict[str, Any] | None = None) -> list[CellResult]:
    """셀 목록을 인식한다. 결과의 순서와 개수는 입력과 같다."""
    settings = settings or load_settings()
    threshold = settings["review"]["low_confidence_threshold"]
    results: list[CellResult | None] = [None] * len(cells)
    pending: list[_Pending] = []
    seen: set[str] = set()

    for index, cell in enumerate(cells):
        problem = _input_problem(cell, seen)
        if problem:
            results[index] = _make_result(cell, engine, settings, status=STATUS_FAILED, reasons=[problem])
        elif cell.column_type == "check":
            results[index] = _make_result(
                cell, engine, settings, status=STATUS_DEFERRED, reasons=[REASON_MARK_PENDING]
            )
        else:
            lines = extract_lines(cell.image, settings)
            if lines:
                pending.append(_Pending(index, lines))
            else:
                results[index] = _make_result(cell, engine, settings, status=STATUS_EMPTY)

    outputs = _predict_lines([line for item in pending for line in item.lines], engine)
    cursor = 0
    for item in pending:
        cell_outputs = outputs[cursor:cursor + len(item.lines)]
        cursor += len(item.lines)
        fields = _assemble(cell_outputs, threshold)
        results[item.index] = _make_result(cells[item.index], engine, settings, **fields)

    return [result for result in results if result is not None]
