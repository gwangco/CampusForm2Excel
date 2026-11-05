"""앱 화면 시험 (T-05 양식 선택 포함). 실제 브라우저 없이 Streamlit을 실행한다."""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

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
