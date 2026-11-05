"""작업 흐름: 원본 보존, 순서, 일부 실패, 양식 재선택, 디스크 복원."""
from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import FakeEngine
from support import simulate_photo

from campusform2excel.intake import Upload
from campusform2excel.pipeline import (
    PAGE_ALIGN_FAILED,
    PAGE_ALIGNED,
    PAGE_OCR_FAILED,
    PAGE_RECOGNIZED,
    PAGE_UPLOADED,
    align_job,
    create_job,
    job_dir,
    load_job,
    page_results,
    recognize_job,
    set_template,
)

FIXTURES = Path(__file__).parent / "fixtures"


def jpeg(image: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return encoded.tobytes()


@pytest.fixture
def form(templates):
    return next(t for t in templates if t.template_id == "attendance")


@pytest.fixture
def good_photo(blank_pages, form):
    return jpeg(simulate_photo(blank_pages[form.template_id], seed=21).image)


@pytest.fixture
def filled_photo():
    return (FIXTURES / "attendance_synthetic.jpg").read_bytes()


def test_originals_are_stored_byte_for_byte(tmp_path, form, good_photo):
    job = create_job([Upload("사진 1.JPG", good_photo)], form, tmp_path)
    page = job.pages[0]
    stored = (job_dir(job, tmp_path) / page.original).read_bytes()
    assert stored == good_photo
    assert page.sha256 == hashlib.sha256(good_photo).hexdigest()

    align_job(job, form, tmp_path)
    assert (job_dir(job, tmp_path) / page.original).read_bytes() == good_photo


def test_pages_keep_the_chosen_order(tmp_path, form, good_photo):
    job = create_job([Upload("c.jpg", good_photo), Upload("a.jpg", good_photo), Upload("b.jpg", good_photo)],
                     form, tmp_path)
    assert [(p.number, p.file_name) for p in job.pages] == [(1, "c.jpg"), (2, "a.jpg"), (3, "b.jpg")]


def test_one_bad_page_does_not_stop_the_others(tmp_path, form, good_photo):
    blank_paper = jpeg(np.full((2000, 1500, 3), 235, np.uint8))
    job = create_job(
        [Upload("1.jpg", good_photo), Upload("2.jpg", blank_paper), Upload("3.jpg", good_photo)], form, tmp_path
    )
    align_job(job, form, tmp_path)
    assert [p.status for p in job.pages] == [PAGE_ALIGNED, PAGE_ALIGN_FAILED, PAGE_ALIGNED]
    assert "다시 촬영" in job.pages[1].message

    recognize_job(job, FakeEngine(), root=tmp_path)
    assert [p.status for p in job.pages] == [PAGE_RECOGNIZED, PAGE_ALIGN_FAILED, PAGE_RECOGNIZED]


def test_low_resolution_page_reports_guidance(tmp_path, form):
    tiny = jpeg(np.full((300, 300, 3), 255, np.uint8))
    job = create_job([Upload("small.jpg", tiny)], form, tmp_path)
    align_job(job, form, tmp_path)
    assert job.pages[0].status == PAGE_ALIGN_FAILED
    assert "해상도" in job.pages[0].message


def test_t05_wrong_form_can_be_corrected_on_the_same_photos(tmp_path, templates, form, good_photo):
    wrong = next(t for t in templates if t.template_id == "lab_evaluation")
    job = create_job([Upload("p.jpg", good_photo)], wrong, tmp_path)
    align_job(job, wrong, tmp_path)
    assert job.pages[0].status == PAGE_ALIGN_FAILED
    assert "양식 선택을 확인" in job.pages[0].message

    set_template(job, form, tmp_path)
    assert job.pages[0].status == PAGE_UPLOADED and job.pages[0].message == ""
    align_job(job, form, tmp_path)
    assert job.pages[0].status == PAGE_ALIGNED
    assert (job.template_id, job.template_version) == ("attendance", "1")


def test_changing_form_clears_results_from_the_old_form(tmp_path, templates, form, good_photo):
    job = create_job([Upload("p.jpg", good_photo)], form, tmp_path)
    align_job(job, form, tmp_path)
    recognize_job(job, FakeEngine(), root=tmp_path)

    other = next(t for t in templates if t.template_id == "lab_evaluation")
    set_template(job, other, tmp_path)
    assert page_results(job, job.pages[0], tmp_path) == []
    assert not (job_dir(job, tmp_path) / "corrected").exists()


def test_every_cell_gets_a_result_and_the_job_restores_from_disk(tmp_path, form, filled_photo):
    job = create_job([Upload("filled.jpg", filled_photo)], form, tmp_path)
    align_job(job, form, tmp_path)
    recognize_job(job, FakeEngine(), root=tmp_path)

    restored = load_job(job.job_id, tmp_path)
    assert restored == job
    rows = page_results(restored, restored.pages[0], tmp_path)
    assert len(rows) == len(form.cells())
    assert {row["template_id"] for row in rows} == {"attendance"}
    assert all(row["bbox"] for row in rows)


def test_engine_crash_on_a_page_is_recorded_and_retry_succeeds(tmp_path, form, filled_photo, monkeypatch):
    job = create_job([Upload("filled.jpg", filled_photo)], form, tmp_path)
    align_job(job, form, tmp_path)

    def crash(*args, **kwargs):
        raise RuntimeError("추론기 오류")

    monkeypatch.setattr("campusform2excel.pipeline.recognize_cells", crash)
    recognize_job(job, FakeEngine(), root=tmp_path)
    assert job.pages[0].status == PAGE_OCR_FAILED

    monkeypatch.undo()
    recognize_job(job, FakeEngine(), root=tmp_path)
    assert job.pages[0].status == PAGE_RECOGNIZED


def test_filled_page_finds_writing_and_blanks(tmp_path, form, filled_photo):
    job = create_job([Upload("filled.jpg", filled_photo)], form, tmp_path)
    align_job(job, form, tmp_path)
    recognize_job(job, FakeEngine(), root=tmp_path)
    rows = page_results(job, job.pages[0], tmp_path)
    by_position = {(row["row"], row["col"]): row["status"] for row in rows}

    assert by_position[(1, 1)] == "recognized"
    assert by_position[(1, 4)] == "deferred"
    assert by_position[(15, 1)] == "empty"
    assert sum(status == "recognized" for status in by_position.values()) == 14
