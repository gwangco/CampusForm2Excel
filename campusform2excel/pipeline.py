"""작업 단위 처리 흐름. 페이지를 순서대로 처리하고 결과를 작업 폴더에 저장한다.

작업 폴더 구조
  data/jobs/<job_id>/job.json          작업·페이지 상태
  data/jobs/<job_id>/originals/        업로드 원본 (바이트 그대로 보존)
  data/jobs/<job_id>/corrected/        보정 이미지
  data/jobs/<job_id>/layout/pNN.json   셀 좌표
  data/jobs/<job_id>/results/pNN.json  셀별 인식 결과
"""
from __future__ import annotations

import hashlib
import json
import logging
import secrets
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path, PurePath
from typing import Any, Callable

from .alignment import AlignmentError, CellBox, align_page, crop_cell
from .imaging import decode_image, read_image, write_png
from .intake import Upload
from .ocr import CellInput, CellResult, Engine, load_settings, recognize_cells
from .paths import JOBS_DIR
from .preprocess import PreprocessError, normalize
from .templates import Template

logger = logging.getLogger(__name__)

PAGE_UPLOADED = "uploaded"
PAGE_ALIGNED = "aligned"
PAGE_ALIGN_FAILED = "align_failed"
PAGE_RECOGNIZED = "recognized"
PAGE_OCR_FAILED = "ocr_failed"

Progress = Callable[[int, int, str], None]


@dataclass
class Page:
    number: int
    file_name: str
    original: str
    sha256: str
    status: str = PAGE_UPLOADED
    message: str = ""
    warnings: list[str] = field(default_factory=list)
    corrected: str | None = None
    grid_score: float | None = None
    rotation: int | None = None
    skew_deg: float | None = None
    scale: float | None = None


@dataclass
class Job:
    job_id: str
    template_id: str
    template_version: str
    created_at: str
    pages: list[Page] = field(default_factory=list)


def job_dir(job: Job | str, root: Path = JOBS_DIR) -> Path:
    return root / (job if isinstance(job, str) else job.job_id)


def save_job(job: Job, root: Path = JOBS_DIR) -> None:
    path = job_dir(job, root) / "job.json"
    path.write_text(json.dumps(asdict(job), ensure_ascii=False, indent=2), encoding="utf-8")


def load_job(job_id: str, root: Path = JOBS_DIR) -> Job:
    raw = json.loads((job_dir(job_id, root) / "job.json").read_text(encoding="utf-8"))
    pages = [Page(**page) for page in raw.pop("pages")]
    return Job(pages=pages, **raw)


def create_job(uploads: list[Upload], template: Template, root: Path = JOBS_DIR) -> Job:
    """검사를 통과한 파일로 작업을 만든다. 파일 순서가 페이지 순서다."""
    now = datetime.now()
    job = Job(
        job_id=f"{now:%Y%m%d-%H%M%S}-{secrets.token_hex(3)}",
        template_id=template.template_id,
        template_version=template.version,
        created_at=now.isoformat(timespec="seconds"),
    )
    originals = job_dir(job, root) / "originals"
    originals.mkdir(parents=True)
    for number, upload in enumerate(uploads, start=1):
        suffix = PurePath(upload.name).suffix.lower()
        relative = f"originals/p{number:02d}{suffix}"
        (job_dir(job, root) / relative).write_bytes(upload.data)
        job.pages.append(Page(number, upload.name, relative, hashlib.sha256(upload.data).hexdigest()))
    save_job(job, root)
    return job


def set_template(job: Job, template: Template, root: Path = JOBS_DIR) -> None:
    """양식을 바꾼다. 원본은 유지하고 이전 양식의 보정·인식 결과는 지운다."""
    job.template_id = template.template_id
    job.template_version = template.version
    for folder in ("corrected", "layout", "results"):
        shutil.rmtree(job_dir(job, root) / folder, ignore_errors=True)
    for page in job.pages:
        page.status, page.message, page.warnings = PAGE_UPLOADED, "", []
        page.corrected = page.grid_score = page.rotation = page.skew_deg = page.scale = None
    save_job(job, root)


def _layout_path(job: Job, page: Page, root: Path) -> Path:
    return job_dir(job, root) / "layout" / f"p{page.number:02d}.json"


def _result_path(job: Job, page: Page, root: Path) -> Path:
    return job_dir(job, root) / "results" / f"p{page.number:02d}.json"


def _align_one(job: Job, page: Page, template: Template, root: Path) -> None:
    folder = job_dir(job, root)
    original = decode_image((folder / page.original).read_bytes())
    normalized = normalize(original)
    aligned = align_page(normalized.image, template, page.number, normalized.scale)

    page.corrected = f"corrected/p{page.number:02d}.png"
    write_png(folder / page.corrected, aligned.image)
    layout = _layout_path(job, page, root)
    layout.parent.mkdir(exist_ok=True)
    layout.write_text(json.dumps([asdict(cell) for cell in aligned.cells], ensure_ascii=False), encoding="utf-8")

    page.status, page.message = PAGE_ALIGNED, ""
    page.warnings = normalized.warnings
    page.grid_score = round(aligned.grid_score, 3)
    page.rotation = aligned.rotation
    page.skew_deg = round(aligned.skew_deg, 2)
    page.scale = normalized.scale


def align_job(job: Job, template: Template, root: Path = JOBS_DIR, progress: Progress | None = None) -> None:
    """페이지별로 보정·셀 정합을 한다. 한 쪽이 실패해도 나머지는 계속 처리한다."""
    for index, page in enumerate(job.pages, start=1):
        if progress:
            progress(index, len(job.pages), page.file_name)
        try:
            _align_one(job, page, template, root)
        except (PreprocessError, AlignmentError) as error:
            page.status, page.message = PAGE_ALIGN_FAILED, str(error)
            if isinstance(error, AlignmentError):
                page.grid_score = error.grid_score
        except Exception:
            logger.exception("보정 중 오류: %s", page.file_name)
            page.status, page.message = PAGE_ALIGN_FAILED, "이미지를 처리하지 못했습니다. 다시 촬영해 주세요."
        save_job(job, root)


def page_cells(job: Job, page: Page, root: Path = JOBS_DIR) -> list[CellBox]:
    raw = json.loads(_layout_path(job, page, root).read_text(encoding="utf-8"))
    return [CellBox(**{**cell, "bbox": tuple(cell["bbox"])}) for cell in raw]


def _page_inputs(job: Job, page: Page, margin: int, root: Path) -> list[CellInput]:
    image = read_image(job_dir(job, root) / page.corrected)
    return [
        CellInput(cell.cell_id, cell.page, cell.row, cell.col, cell.bbox, crop_cell(image, cell.bbox, margin),
                  cell.column_type, job.template_id, job.template_version)
        for cell in page_cells(job, page, root)
    ]


def recognize_job(job: Job, engine: Engine, settings: dict[str, Any] | None = None,
                  root: Path = JOBS_DIR, progress: Progress | None = None) -> None:
    """정합에 성공한 페이지를 순서대로 인식하고 셀별 결과를 저장한다."""
    settings = settings or load_settings()
    ready = [page for page in job.pages if page.status in (PAGE_ALIGNED, PAGE_RECOGNIZED, PAGE_OCR_FAILED)]
    for index, page in enumerate(ready, start=1):
        if progress:
            progress(index, len(ready), page.file_name)
        try:
            results: list[CellResult] = recognize_cells(
                _page_inputs(job, page, settings["cell_margin_px"], root), engine, settings
            )
            path = _result_path(job, page, root)
            path.parent.mkdir(exist_ok=True)
            path.write_text(
                json.dumps([result.to_dict() for result in results], ensure_ascii=False, indent=1),
                encoding="utf-8",
            )
            page.status, page.message = PAGE_RECOGNIZED, ""
        except Exception:
            logger.exception("인식 중 오류: %s", page.file_name)
            page.status, page.message = PAGE_OCR_FAILED, "이 쪽을 인식하지 못했습니다. 다시 시도해 주세요."
        save_job(job, root)


def page_results(job: Job, page: Page, root: Path = JOBS_DIR) -> list[dict[str, Any]]:
    path = _result_path(job, page, root)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
