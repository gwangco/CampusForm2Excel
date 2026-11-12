"""앱 화면 시험 (T-05 양식 선택 포함). 실제 브라우저 없이 Streamlit을 실행한다."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
import streamlit as st
from openpyxl import load_workbook
from streamlit.testing.v1 import AppTest

from campusform2excel.pipeline import load_job, page_results
from campusform2excel.review import inspect_page
from campusform2excel.templates import load_template
from make_fixtures import ATTENDANCE_ROWS

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"


@pytest.fixture
def app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)


def upload(app, name):
    app.file_uploader[0].upload(name, (FIXTURES / name).read_bytes(), "image/jpeg")
    return app


def test_app_starts_with_three_steps_and_settings(app):
    app.run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ["1단계 양식·업로드", "2단계 인식·검수", "3단계 파일 저장", "양식 설정"]


def test_t05_three_forms_offered(app):
    app.run()
    assert app.selectbox[0].options == ["출석표 v1", "장비 대여 장부 v1", "실습 평가표 v1"]


def test_start_button_is_disabled_until_a_file_is_uploaded(app):
    app.run()
    assert app.button[0].disabled


def test_upload_and_align_shows_result(app):
    app.run()
    upload(app, "attendance_synthetic.jpg").run()
    assert not app.button[0].disabled

    app.button[0].click().run()
    assert not app.exception
    assert app.query_params["job"]
    assert any("정합 결과" in item.value for item in app.subheader)
    assert not app.error


def test_recognition_shows_counts_and_cell_detail_with_real_model(app):
    """업로드부터 인식 결과와 원본 대조까지 한 번에 지나간다."""
    app.run()
    upload(app, "attendance_synthetic.jpg").run()
    app.button[0].click().run()

    start = next(button for button in app.button if button.label == "셀 인식 시작")
    start.click().run()
    assert not app.exception

    metrics = {metric.label: metric.value for metric in app.metric}
    assert metrics["전체 셀"] == "100"
    assert metrics["인식"] == "14"
    assert metrics["빈칸"] == "66"
    assert metrics["실패"] == "0"

    picker = next(box for box in app.selectbox if box.label == "원본과 대조할 셀")
    assert picker.value == "p01-r01-c01"
    assert any("2400731" in item.value for item in app.markdown)

    picker.select("p01-r01-c02").run()
    assert any("김가람" in item.value for item in app.markdown)


def test_saved_job_is_restored_from_the_address_after_refresh(app):
    app.run()
    upload(app, "attendance_synthetic.jpg").run()
    app.button[0].click().run()
    job_id = app.query_params["job"]

    fresh = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    fresh.query_params["job"] = job_id
    fresh.run()
    assert not fresh.exception
    assert any("정합 결과" in item.value for item in fresh.subheader)


def test_single_photo_to_reviewed_xlsx(app, monkeypatch):
    """실제 CPU OCR → 원본 대조·수정 → 복원 → 다운로드 바이트 검증."""
    downloads = []
    download_button = st.download_button

    def capture(label, data, **kwargs):
        if label == "확정 XLSX 다운로드":
            downloads.append(data)
        return download_button(label, data, **kwargs)

    def click(label, current):
        next(button for button in current.button if button.label == label).click().run()
        assert not current.exception

    def edit(cell_id, value, current):
        picker = next(box for box in current.selectbox if box.label == "원본과 대조할 셀")
        picker.select(cell_id).run()
        current.text_input[0].set_value(value)
        click("셀 확인·저장", current)

    monkeypatch.setattr(st, "download_button", capture)
    upload(app.run(), "attendance_synthetic.jpg").run()
    click("보정·셀 정합 시작", app)
    click("셀 인식 시작", app)
    job = load_job(app.query_params["job"])
    page = job.pages[0]
    template = load_template(job.template_id, job.template_version)
    original_results = page_results(job, page)
    assert not downloads

    edit("p01-r01-c01", "12", app)
    assert any("학번을 숫자 7자리" in message.value for message in app.warning)
    assert not downloads

    expected = [row.copy() for row in ATTENDANCE_ROWS]
    expected[0][0] = "0000731"
    expected[0][-1] = "=1+1"
    for cell in inspect_page(job, page, template).cells:
        if cell.box.row <= len(expected):
            value = expected[cell.box.row - 1][cell.box.col - 1]
            if cell.issues or cell.value != value:
                edit(cell.box.cell_id, value, app)

    assert downloads
    assert page_results(job, page) == original_results

    fresh = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    fresh.query_params["job"] = job.job_id
    fresh.run()
    assert not fresh.exception
    assert fresh.text_input[0].value == "0000731"

    sheet = load_workbook(BytesIO(downloads[-1])).active
    assert list(sheet.values) == [
        tuple(column["label"] for column in template.columns),
        *(tuple(value or None for value in row) for row in expected),
    ]
    assert (sheet["A2"].data_type, sheet["A2"].number_format) == ("s", "@")
    assert sheet["E2"].data_type == "s"


@pytest.mark.parametrize("job_id", ["../../etc", "20261105-000000-abcdef", "<script>"])
def test_invalid_job_address_is_ignored(job_id):
    fresh = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60)
    fresh.query_params["job"] = job_id
    fresh.run()
    assert not fresh.exception
    assert not any("정합 결과" in item.value for item in fresh.subheader)


def test_t05_wrong_form_shows_guidance_then_can_be_corrected(app):
    app.run()
    upload(app, "attendance_synthetic.jpg").run()
    app.selectbox[0].select_index(2).run()
    app.button[0].click().run()
    assert any("양식 선택을 확인" in item.value for item in app.error)

    app.selectbox[0].select_index(0).run()
    retry = next(button for button in app.button if "다시 정합" in button.label)
    retry.click().run()
    assert not app.exception
    assert not app.error
