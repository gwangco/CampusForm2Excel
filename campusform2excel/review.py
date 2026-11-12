"""셀 검수 저장과 양식 규칙 검사 (FR-13~15)."""
from __future__ import annotations

import re
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from contextlib import closing
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from .alignment import CellBox, crop_cell
from .imaging import read_image
from .ocr import extract_lines, load_settings
from .paths import JOBS_DIR
from .pipeline import PAGE_RECOGNIZED, Job, Page, job_dir, load_job, page_cells, page_results
from .templates import NUMERIC_TYPES, Template

INVALID_XLSX_TEXT = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]")


class ReviewError(ValueError):
    """다시 불러오거나 검수를 마쳐야 진행할 수 있다."""


@dataclass(frozen=True)
class SavedReview:
    revision: int
    value: str


@dataclass
class ReviewCell:
    box: CellBox
    result: dict[str, Any]
    saved: SavedReview | None
    issues: list[str] = field(default_factory=list)

    @property
    def value(self) -> str:
        return normalize_value(self.saved.value if self.saved else self.result["raw_text"])


@dataclass
class PageReview:
    cells: list[ReviewCell]
    rows: dict[int, list[ReviewCell]]
    duplicates: dict[tuple[str, ...], list[int]]


def normalize_value(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip()


def _open_db(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("""
        CREATE TABLE IF NOT EXISTS review (
            revision INTEGER PRIMARY KEY,
            template_id TEXT NOT NULL,
            template_version TEXT NOT NULL,
            cell_id TEXT NOT NULL,
            raw_text TEXT NOT NULL,
            previous_value TEXT NOT NULL,
            value TEXT NOT NULL,
            saved_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    db.execute("""
        CREATE INDEX IF NOT EXISTS review_cell
        ON review (template_id, template_version, cell_id, revision)
    """)
    return db


def _load_reviews(job: Job, root: Path) -> dict[str, SavedReview]:
    path = job_dir(job, root) / "review.sqlite3"
    if not path.exists():
        return {}
    with closing(_open_db(path)) as db:
        rows = db.execute("""
            SELECT cell_id, revision, value FROM review
            WHERE template_id = ? AND template_version = ?
            ORDER BY revision
        """, (job.template_id, job.template_version))
        return {cell_id: SavedReview(revision, value) for cell_id, revision, value in rows}


def save_review(job: Job, cell: ReviewCell, value: str, root: Path = JOBS_DIR) -> None:
    """원본 OCR 값과 이전 검수값을 남기고 셀의 최신 검수값을 저장한다."""
    scope = (job.template_id, job.template_version, cell.box.cell_id)
    current = load_job(job.job_id, root)
    if (current.template_id, current.template_version) != scope[:2] or not any(
        page.number == cell.box.page and page.status == PAGE_RECOGNIZED for page in current.pages
    ):
        raise ReviewError("양식 또는 페이지 상태가 바뀌었습니다. 화면을 새로 고쳐 주세요.")

    value = normalize_value(value)
    with closing(_open_db(job_dir(job, root) / "review.sqlite3")) as db, db:
        db.execute("BEGIN IMMEDIATE")
        latest = db.execute("""
            SELECT revision, value FROM review
            WHERE template_id = ? AND template_version = ? AND cell_id = ?
            ORDER BY revision DESC LIMIT 1
        """, scope).fetchone()
        if latest and latest[1] == value:
            return
        if (latest[0] if latest else None) != (cell.saved.revision if cell.saved else None):
            raise ReviewError("다른 화면에서 이 셀을 수정했습니다. 화면을 새로 고쳐 주세요.")
        db.execute("""
            INSERT INTO review (template_id, template_version, cell_id, raw_text, previous_value, value)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (*scope, cell.result["raw_text"], cell.value, value))


def _date_key(value: str, formats: list[str]) -> str | None:
    for fmt in formats:
        try:
            if "%Y" not in fmt and "%y" not in fmt:
                # 연도가 생략된 양식은 2월 29일을 허용하고 입력에 없는 연도를 쓰지 않는다.
                return datetime.strptime(f"{value}/2000", f"{fmt}/%Y").strftime("--%m-%d")
            return datetime.strptime(value, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _numeric_value(value: str, kind: str) -> Decimal | None:
    pattern = r"[+-]?[0-9]+" if kind == "int" else r"[+-]?[0-9]+(?:\.[0-9]+)?"
    return Decimal(value) if re.fullmatch(pattern, value) else None


def _validate_value(value: str, column: dict[str, Any]) -> str | None:
    if not value:
        return "필수 값이 비어 있습니다." if column.get("required") else None
    if len(value) > 32767 or INVALID_XLSX_TEXT.search(value):
        return "Excel 셀에 저장할 수 없는 문자 또는 길이입니다."
    kind = column["type"]
    if kind == "student_id" and not re.fullmatch(rf"[0-9]{{{column['length']}}}", value):
        return f"학번을 숫자 {column['length']}자리로 입력해 주세요."
    if kind == "date" and _date_key(value, column["date_formats"]) is None:
        return "양식에 지정된 날짜 형식과 실제 날짜를 확인해 주세요."
    if kind in NUMERIC_TYPES:
        number = _numeric_value(value, kind)
        if number is None:
            return "정수를 입력해 주세요." if kind == "int" else "숫자를 입력해 주세요."
        if ("min" in column and number < column["min"]) or ("max" in column and number > column["max"]):
            return f"허용 범위를 확인해 주세요. ({column.get('min', '제한 없음')} ~ {column.get('max', '제한 없음')})"
    return None


def _validate_row(cells: list[ReviewCell], columns: list[dict[str, Any]]) -> None:
    by_key = {cell.box.column_key: cell for cell in cells}
    for cell, column in zip(cells, columns):
        problem = _validate_value(cell.value, column)
        if problem:
            cell.issues.append(problem)
        if column["type"] == "total" and cell.value and not problem:
            sources = [by_key[key] for key in column["sum_of"]]
            numbers = [_numeric_value(source.value, source.box.column_type) for source in sources]
            if all(number is not None for number in numbers):
                expected = sum(numbers)
                if Decimal(cell.value) != expected:
                    cell.issues.append(f"항목 합계가 {expected:f}입니다.")


def _find_duplicates(rows: dict[int, list[ReviewCell]], template: Template) -> dict[tuple[str, ...], list[int]]:
    groups = defaultdict(list)
    for row, cells in rows.items():
        by_key = {cell.box.column_key: cell for cell in cells}
        key_cells = [by_key[key] for key in template.raw["business_key"]]
        if key_cells and all(cell.value for cell in key_cells):
            key = []
            for cell in key_cells:
                value = cell.value
                if cell.box.column_type == "date":
                    column = template.columns[cell.box.col - 1]
                    value = _date_key(value, column["date_formats"]) or value
                key.append(value)
            groups[tuple(key)].append(row)
    return {key: members for key, members in groups.items() if len(members) > 1}


def inspect_page(job: Job, page: Page, template: Template, root: Path = JOBS_DIR) -> PageReview:
    """셀 ID로 인식과 수정값을 연결하고 현재 값으로 경고를 다시 계산한다."""
    results = page_results(job, page, root)
    counts = Counter(result["cell_id"] for result in results)
    by_id = {result["cell_id"]: result for result in results}
    reviews = _load_reviews(job, root)
    cells = []
    rows = defaultdict(list)
    for box in page_cells(job, page, root):
        result = by_id.get(box.cell_id)
        if counts[box.cell_id] != 1:
            reason = "duplicate_cell_id" if result else "missing_result"
            result = {"raw_text": "", "confidence": None, "status": "failed", "reasons": [reason]}
        cell = ReviewCell(box, result, reviews.get(box.cell_id))
        cells.append(cell)
        rows[box.row].append(cell)

    data_rows = {}
    settings = load_settings()
    corrected = None
    for row, row_cells in sorted(rows.items()):
        row_cells.sort(key=lambda cell: cell.box.col)
        active = any(
            cell.value or (cell.saved is None and cell.result["status"] not in ("empty", "deferred"))
            for cell in row_cells
        )
        if not active:
            # 체크 열만 표시된 행도 검수에서 놓치지 않도록 글씨 픽셀 유무만 확인한다.
            for cell in row_cells:
                if cell.result["status"] == "deferred" and cell.saved is None:
                    if corrected is None:
                        corrected = read_image(job_dir(job, root) / page.corrected)
                    image = crop_cell(corrected, cell.box.bbox, settings["cell_margin_px"])
                    if image.size and extract_lines(image, settings):
                        active = True
                        break
        if active:
            for cell in row_cells:
                if cell.saved is None:
                    cell.issues.extend(cell.result["reasons"])
            _validate_row(row_cells, template.columns)
            data_rows[row] = row_cells
    return PageReview(cells, data_rows, _find_duplicates(data_rows, template))
