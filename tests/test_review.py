"""검수 경고, 수정값 보존, 양식별 XLSX 출력."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from openpyxl import load_workbook

from campusform2excel.export import xlsx_bytes
from campusform2excel.intake import Upload
from campusform2excel.pipeline import align_job, create_job, recognize_job
from campusform2excel.review import ReviewError, inspect_page, save_review
from campusform2excel.templates import NUMERIC_TYPES, load_template
from make_fixtures import SELECTION

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("template_id", list(SELECTION))
def test_export_keeps_template_columns_and_reviewed_values(template_id, tmp_path, fake_engine):
    template = load_template(template_id, "1")
    photo = FIXTURES / f"{template_id}_synthetic.jpg"
    job = create_job([Upload(photo.name, photo.read_bytes())], template, tmp_path)
    align_job(job, template, tmp_path)
    recognize_job(job, fake_engine, root=tmp_path)
    page = job.pages[0]

    review = inspect_page(job, page, template, tmp_path)
    with pytest.raises(ReviewError, match="남은 검수"):
        xlsx_bytes(review, template)

    expected = SELECTION[template_id]["rows"]
    for cell in review.cells:
        if cell.box.row <= len(expected):
            save_review(job, cell, expected[cell.box.row - 1][cell.box.col - 1], tmp_path)

    review = inspect_page(job, page, template, tmp_path)
    sheet = load_workbook(BytesIO(xlsx_bytes(review, template))).active
    assert list(sheet.values) == [
        tuple(column["label"] for column in template.columns),
        *(
            tuple(int(value) if value and column["type"] in NUMERIC_TYPES else value or None
                  for value, column in zip(row, template.columns))
            for row in expected
        ),
    ]

    if template_id == "lab_evaluation":
        score = review.rows[1][3]
        save_review(job, score, "26", tmp_path)
        review = inspect_page(job, page, template, tmp_path)
        assert "허용 범위" in review.rows[1][3].issues[0]
        assert review.rows[1][-1].issues == ["항목 합계가 88입니다."]
        with pytest.raises(ReviewError, match="다른 화면"):
            save_review(job, score, "20", tmp_path)
        with pytest.raises(ReviewError, match="남은 검수"):
            xlsx_bytes(review, template)

    if template_id == "attendance":
        save_review(job, review.rows[2][0], expected[0][0], tmp_path)
        save_review(job, review.rows[2][2], "2026-11-03", tmp_path)
        review = inspect_page(job, page, template, tmp_path)
        assert list(review.duplicates.values()) == [[1, 2]]
        with pytest.raises(ReviewError, match="중복 행"):
            xlsx_bytes(review, template)
        sheet = load_workbook(BytesIO(xlsx_bytes(review, template, duplicates_confirmed=True))).active
        assert sheet.max_row == len(expected) + 1
        assert sheet["A2"].value == sheet["A3"].value
