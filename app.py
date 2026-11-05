"""CampusForm2Excel 로컬 앱.

    streamlit run app.py

11월 1주 범위: 양식 선택·업로드·보정·셀 정합(SC-01, SC-02)과 인식 결과 조회(SC-06).
값 수정·저장은 11월 2주, 파일 저장은 11월 2주 이후에 이 화면에 연결한다.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from campusform2excel.alignment import crop_cell
from campusform2excel.imaging import read_image, to_rgb
from campusform2excel.intake import MAX_FILES, Upload, validate_uploads
from campusform2excel.ocr import ModelError, PaddleEngine, load_model_lock, load_settings
from campusform2excel.paths import FORMS_DIR
from campusform2excel.pipeline import (
    PAGE_ALIGN_FAILED,
    PAGE_ALIGNED,
    PAGE_OCR_FAILED,
    PAGE_RECOGNIZED,
    Job,
    Page,
    align_job,
    create_job,
    job_dir,
    load_job,
    page_cells,
    page_results,
    recognize_job,
    set_template,
)
from campusform2excel.templates import Template, list_templates, load_template

JOB_ID_PATTERN = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")
PREVIEW_WIDTH = 560

STATUS_LABELS = {
    "recognized": "인식",
    "empty": "빈칸",
    "unreadable": "판독 불가",
    "failed": "실패",
    "deferred": "체크 칸(11월 3주)",
}
REASON_LABELS = {
    "low_confidence": "낮은 신뢰도",
    "no_score": "점수 없음",
    "no_text": "글자 없음",
    "inference_error": "인식 오류",
    "invalid_image": "이미지 오류",
    "invalid_cell_id": "셀 ID 오류",
    "duplicate_cell_id": "셀 ID 중복",
    "mark_check_pending": "표식 판정 전",
}


@st.cache_resource(show_spinner=False)
def get_engine() -> PaddleEngine:
    """모델은 앱이 켜져 있는 동안 한 번만 읽는다."""
    return PaddleEngine()


def current_job() -> Job | None:
    """주소의 작업 ID로 작업을 복원한다. 화면을 새로 고쳐도 결과가 유지된다."""
    job_id = st.query_params.get("job", "")
    if not JOB_ID_PATTERN.match(job_id) or not (job_dir(job_id) / "job.json").exists():
        return None
    return load_job(job_id)


def job_template(job: Job) -> Template:
    return load_template(job.template_id, job.template_version)


def grid_overlay(job: Job, page: Page) -> np.ndarray:
    """보정 이미지 위에 셀 격자를 그린다."""
    image = read_image(job_dir(job) / page.corrected)
    for cell in page_cells(job, page):
        x0, y0, x1, y1 = cell.bbox
        cv2.rectangle(image, (x0, y0), (x1, y1), (60, 170, 40), 2)
    return to_rgb(image)


def render_page_status(job: Job) -> None:
    for page in job.pages:
        title = f"{page.number}쪽 · {page.file_name}"
        if page.status == PAGE_ALIGN_FAILED:
            st.error(f"{title}: {page.message}")
            continue
        if page.corrected is None:
            continue
        with st.expander(f"{title} · 격자 일치도 {page.grid_score:.2f}", expanded=len(job.pages) == 1):
            for warning in page.warnings:
                st.warning(warning)
            st.caption(f"기울기 {page.skew_deg:.1f}° · 회전 {page.rotation}° 보정")
            st.image(grid_overlay(job, page), width=PREVIEW_WIDTH)


def ordered_uploads(accepted: list[Upload]) -> list[Upload] | None:
    """사용자가 정한 페이지 순서를 받는다. 순서가 잘못되면 None을 돌려준다."""
    if len(accepted) == 1:
        return accepted

    st.caption("페이지 순서를 확인해 주세요.")
    table = pd.DataFrame({"순서": range(1, len(accepted) + 1), "파일": [item.name for item in accepted]})
    edited = st.data_editor(
        table, hide_index=True, disabled=["파일"], key="page_order",
        column_config={"순서": st.column_config.NumberColumn(min_value=1, max_value=len(accepted), step=1)},
    )
    order = edited["순서"].tolist()
    if sorted(order) != list(range(1, len(accepted) + 1)):
        st.error("순서 번호가 겹치거나 비어 있습니다. 1부터 차례대로 입력해 주세요.")
        return None
    return [item for _, item in sorted(zip(order, accepted), key=lambda pair: pair[0])]


def render_upload_tab(templates: list[Template], job: Job | None) -> None:
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.subheader("양식 선택")
        selected = st.selectbox("처리할 양식", templates, format_func=lambda item: item.label, key="template")
        st.caption(
            " · ".join(column["label"] for column in selected.columns)
            + f"  ({selected.data_row_count}행)"
        )
        blank = FORMS_DIR / f"{selected.template_id}_v{selected.version}.png"
        if blank.exists():
            st.download_button(
                "빈 양식 내려받기", blank.read_bytes(), file_name=blank.name, mime="image/png",
            )

    with right:
        st.subheader("사진 올리기")
        files = st.file_uploader(
            f"JPG·PNG, 1장당 20MB 이하, 최대 {MAX_FILES}장",
            type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="uploads",
        )
        uploads = [Upload(file.name, file.getvalue()) for file in files or []]
        intake = validate_uploads(uploads)
        for rejection in intake.rejected:
            st.error(f"{rejection.name}: {rejection.reason}")

        ordered = ordered_uploads(intake.accepted) if intake.ok else None
        if st.button("보정·셀 정합 시작", type="primary", disabled=ordered is None):
            job = create_job(ordered, selected)
            run_alignment(job, selected)
            st.query_params["job"] = job.job_id
            st.rerun()

    if job is None:
        return

    st.divider()
    st.subheader(f"정합 결과 · {job_template(job).label}")
    if (selected.template_id, selected.version) != (job.template_id, job.template_version):
        st.info(f"양식 선택이 바뀌었습니다. 같은 사진을 '{selected.label}'로 다시 정합할 수 있습니다.")
        if st.button(f"'{selected.label}'로 다시 정합"):
            set_template(job, selected)
            run_alignment(job, selected)
            st.rerun()
    render_page_status(job)


def run_alignment(job: Job, template: Template) -> None:
    bar = st.progress(0.0, text="보정 준비 중")

    def report(index: int, total: int, name: str) -> None:
        bar.progress((index - 1) / total, text=f"{index}/{total}쪽 보정 중: {name}")

    align_job(job, template, progress=report)
    bar.empty()


def run_recognition(job: Job) -> None:
    try:
        with st.spinner("OCR 모델을 불러오는 중입니다"):
            engine = get_engine()
    except ModelError as error:
        st.error(str(error))
        return

    bar = st.progress(0.0, text="인식 준비 중")

    def report(index: int, total: int, name: str) -> None:
        bar.progress((index - 1) / total, text=f"{index}/{total}쪽 인식 중: {name}")

    recognize_job(job, engine, progress=report)
    bar.empty()
    st.rerun()


def result_table(rows: list[dict[str, Any]], template: Template) -> pd.DataFrame:
    labels = [column["label"] for column in template.columns]
    return pd.DataFrame([
        {
            "셀 ID": row["cell_id"],
            "행": row["row"],
            "열": labels[row["col"] - 1],
            "인식값": row["raw_text"],
            "신뢰도": row["confidence"],
            "상태": STATUS_LABELS.get(row["status"], row["status"]),
            "검수": "필요" if row["review_required"] else "",
            "사유": ", ".join(REASON_LABELS.get(reason, reason) for reason in row["reasons"]),
        }
        for row in rows
    ])


def render_cell_detail(job: Job, page: Page, row: dict[str, Any]) -> None:
    """SC-06: 선택한 셀의 원본과 인식값을 함께 보여 준다."""
    image = read_image(job_dir(job) / page.corrected)
    st.image(to_rgb(crop_cell(image, tuple(row["bbox"]))), caption=f"{row['cell_id']} 원본 셀")
    confidence = "없음" if row["confidence"] is None else f"{row['confidence']:.3f}"
    st.markdown(f"**인식값** `{row['raw_text'] or '(없음)'}`  ·  **신뢰도** {confidence}")
    st.caption(f"모델 {row['model_name']} · 설정 {row['config_version']} · "
               f"양식 {row['template_id']} v{row['template_version']}")


def render_results(job: Job, template: Template) -> None:
    done = [page for page in job.pages if page.status == PAGE_RECOGNIZED]
    rows_by_page = {page.number: page_results(job, page) for page in done}
    rows = [row for page_rows in rows_by_page.values() for row in page_rows]
    counts = Counter(row["status"] for row in rows)

    columns = st.columns(5)
    columns[0].metric("전체 셀", len(rows))
    columns[1].metric("인식", counts["recognized"])
    columns[2].metric("빈칸", counts["empty"])
    columns[3].metric("검수 필요", sum(row["review_required"] for row in rows))
    columns[4].metric("실패", counts["failed"] + counts["unreadable"])

    page = st.selectbox("페이지", done, format_func=lambda item: f"{item.number}쪽 · {item.file_name}")
    show_empty = st.toggle("빈칸도 보기", value=False)
    visible = [row for row in rows_by_page[page.number] if show_empty or row["status"] != "empty"]
    if not visible:
        st.info("글자가 적힌 셀이 없습니다.")
        return

    table_area, detail_area = st.columns([3, 2], gap="large")
    with table_area:
        st.dataframe(
            result_table(visible, template), hide_index=True,
            column_config={"신뢰도": st.column_config.NumberColumn(format="%.3f")},
        )
    with detail_area:
        by_id = {row["cell_id"]: row for row in visible}
        cell_id = st.selectbox(
            "원본과 대조할 셀", list(by_id), key=f"cell_{page.number}",
            format_func=lambda item: f"{item} · {by_id[item]['raw_text'] or '(없음)'}",
        )
        render_cell_detail(job, page, by_id[cell_id])


def render_review_tab(job: Job | None) -> None:
    if job is None:
        st.info("1단계에서 양식을 고르고 사진을 올려 주세요.")
        return

    ready = [page for page in job.pages if page.status in (PAGE_ALIGNED, PAGE_RECOGNIZED, PAGE_OCR_FAILED)]
    if not ready:
        st.warning("정합에 성공한 쪽이 없습니다. 1단계의 안내를 확인해 주세요.")
        return

    template = job_template(job)
    st.caption(f"{template.label} · 정합 성공 {len(ready)}/{len(job.pages)}쪽")
    if st.button("셀 인식 시작", type="primary"):
        run_recognition(job)
    for page in job.pages:
        if page.status == PAGE_OCR_FAILED:
            st.error(f"{page.number}쪽: {page.message}")

    if any(page.status == PAGE_RECOGNIZED for page in job.pages):
        render_results(job, template)
        st.caption("값 수정과 저장은 11월 2주에 연결됩니다.")


def render_settings_tab(templates: list[Template]) -> None:
    """SC-01: 등록된 양식 설정 조회."""
    selected = st.selectbox("양식", templates, format_func=lambda item: item.label, key="settings_template")
    x0, y0, x1, y1 = selected.table_bbox
    st.caption(f"표 영역 ({x0}, {y0}) ~ ({x1}, {y1}) · 데이터 {selected.data_row_count}행 · "
               f"업무 키 {', '.join(selected.raw['business_key'])}")
    rules = pd.DataFrame([
        {
            "열": column["label"],
            "키": column["key"],
            "타입": column["type"],
            "필수": "예" if column.get("required") else "",
            "규칙": ", ".join(
                f"{name}={value}" for name, value in column.items()
                if name not in ("key", "label", "type", "required")
            ),
        }
        for column in selected.columns
    ])
    st.dataframe(rules, hide_index=True)
    with st.expander("설정 파일 원문"):
        st.json(selected.raw)

    lock = load_model_lock()
    st.subheader("OCR 실행 설정")
    st.caption(f"모델 {lock['model_name']} · 라이선스 {lock['license']} · 설정 {load_settings()['config_version']}")
    st.code(lock["files"]["inference.pdiparams"]["sha256"], language=None)


def main() -> None:
    st.set_page_config(page_title="CampusForm2Excel", layout="wide")
    st.title("CampusForm2Excel")
    st.caption("교내 수기 양식을 검수 가능한 표로 변환합니다. 사진은 이 PC 밖으로 전송되지 않습니다.")

    templates = list_templates()
    job = current_job()

    upload, review, export, settings = st.tabs(["1단계 양식·업로드", "2단계 인식·검수", "3단계 파일 저장", "양식 설정"])
    with upload:
        render_upload_tab(templates, job)
    with review:
        render_review_tab(job)
    with export:
        st.info("병합과 XLSX·CSV 저장은 다음 일정에 연결됩니다.")
    with settings:
        render_settings_tab(templates)


if __name__ == "__main__":
    main()
