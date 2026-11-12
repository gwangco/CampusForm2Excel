"""CampusForm2Excel 로컬 앱.

    streamlit run app.py

양식 선택·업로드·보정, 원본 대조·검수 및 한 페이지 XLSX 저장을 지원한다.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from campusform2excel.export import XLSX_MIME, xlsx_bytes
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
    recognize_job,
    set_template,
)
from campusform2excel.review import PageReview, ReviewCell, ReviewError, inspect_page, save_review
from campusform2excel.templates import Template, list_templates, load_template

JOB_ID_PATTERN = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")
PREVIEW_WIDTH = 560

STATUS_LABELS = {
    "recognized": "인식",
    "empty": "빈칸",
    "unreadable": "판독 불가",
    "failed": "실패",
    "deferred": "수동 확인",
}
REASON_LABELS = {
    "low_confidence": "낮은 신뢰도",
    "no_score": "점수 없음",
    "no_text": "글자 없음",
    "inference_error": "인식 오류",
    "invalid_image": "이미지 오류",
    "invalid_cell_id": "셀 ID 오류",
    "duplicate_cell_id": "셀 ID 중복",
    "missing_result": "OCR 응답 누락",
    "mark_check_pending": "원본의 체크 칸을 확인해 주세요.",
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

    recognize_job(job, engine, progress=report, only_pending=True)
    bar.empty()
    st.rerun()


def result_table(cells: list[ReviewCell], template: Template) -> pd.DataFrame:
    labels = [column["label"] for column in template.columns]
    return pd.DataFrame([
        {
            "셀 ID": cell.box.cell_id,
            "행": cell.box.row,
            "열": labels[cell.box.col - 1],
            "OCR 원본값": cell.result["raw_text"],
            "현재 값": cell.value,
            "신뢰도": cell.result["confidence"],
            "인식 상태": STATUS_LABELS.get(cell.result["status"], cell.result["status"]),
            "검수": "필요" if cell.issues else ("저장됨" if cell.saved else ""),
            "사유": " · ".join(REASON_LABELS.get(issue, issue) for issue in cell.issues),
        }
        for cell in cells
    ])


def render_cell_detail(job: Job, page: Page, cell: ReviewCell) -> None:
    """원본 사진의 해당 셀을 대조하면서 값 수정 또는 빈칸을 확정한다."""
    image = read_image(job_dir(job) / page.original)
    quad = np.rint(cell.box.source_quad).astype(np.int32)
    x, y, width, height = cv2.boundingRect(quad)
    cv2.polylines(image, [quad], True, (60, 170, 40), 2)
    crop = image[max(0, y):y + height, max(0, x):x + width]
    st.image(to_rgb(crop if crop.size else image), caption=f"{cell.box.cell_id} · 업로드 원본")
    row = cell.result
    confidence = "없음" if row["confidence"] is None else f"{row['confidence']:.3f}"
    st.markdown(f"**OCR 원본값** `{row['raw_text'] or '(없음)'}`  ·  **신뢰도** {confidence}")
    if "model_name" in row:
        st.caption(f"모델 {row['model_name']} · 설정 {row['config_version']} · "
                   f"양식 {job.template_id} v{job.template_version}")
    if cell.issues:
        st.warning(" · ".join(REASON_LABELS.get(issue, issue) for issue in cell.issues))
    if cell.saved:
        st.caption("검수값을 저장했습니다.")

    revision = cell.saved.revision if cell.saved else 0
    form_key = f"edit_{job.job_id}_{job.template_id}_{job.template_version}_{cell.box.cell_id}_{revision}"
    with st.form(form_key):
        value = st.text_input("확정할 값", value=cell.value, max_chars=32767)
        if st.form_submit_button("셀 확인·저장"):
            try:
                save_review(job, cell, value)
            except ReviewError as error:
                st.error(str(error))
            else:
                st.rerun()


def render_results(job: Job, page: Page, template: Template, review: PageReview) -> None:
    counts = Counter(cell.result["status"] for cell in review.cells)

    columns = st.columns(5)
    columns[0].metric("전체 셀", len(review.cells))
    columns[1].metric("인식", counts["recognized"])
    columns[2].metric("빈칸", counts["empty"])
    columns[3].metric("검수 필요", sum(bool(cell.issues) for cell in review.cells))
    columns[4].metric("실패", counts["failed"] + counts["unreadable"])

    show_empty = st.toggle("미사용 행도 보기", value=not review.rows)
    only_issues = st.toggle("검수 필요한 셀만 보기", value=False)
    visible = [
        cell for cell in review.cells
        if (show_empty or cell.box.row in review.rows) and (not only_issues or cell.issues)
    ]
    if not visible:
        st.info("이 조건에 해당하는 셀이 없습니다.")
        return

    table_area, detail_area = st.columns([3, 2], gap="large")
    with table_area:
        st.dataframe(
            result_table(visible, template), hide_index=True,
            column_config={"신뢰도": st.column_config.NumberColumn(format="%.3f")},
        )
    with detail_area:
        by_id = {cell.box.cell_id: cell for cell in visible}
        cell_id = st.selectbox(
            "원본과 대조할 셀", list(by_id), key=f"cell_{job.job_id}_{job.template_id}_{page.number}",
            format_func=lambda item: (
                f"{by_id[item].box.row}행 · {template.columns[by_id[item].box.col - 1]['label']} · "
                f"{by_id[item].value or '(빈칸)'}"
            ),
        )
        render_cell_detail(job, page, by_id[cell_id])


def render_review_tab(job: Job | None) -> tuple[Page, Template, PageReview] | None:
    if job is None:
        st.info("1단계에서 양식을 고르고 사진을 올려 주세요.")
        return

    ready = [page for page in job.pages if page.status in (PAGE_ALIGNED, PAGE_RECOGNIZED, PAGE_OCR_FAILED)]
    if not ready:
        st.warning("정합에 성공한 쪽이 없습니다. 1단계의 안내를 확인해 주세요.")
        return

    template = job_template(job)
    st.caption(f"{template.label} · 정합 성공 {len(ready)}/{len(job.pages)}쪽")
    pending = [page for page in ready if page.status != PAGE_RECOGNIZED]
    label = "셀 인식 시작" if any(page.status == PAGE_ALIGNED for page in pending) else "인식 실패 쪽 다시 시도"
    if pending and st.button(label, type="primary"):
        run_recognition(job)
    for page in job.pages:
        if page.status == PAGE_OCR_FAILED:
            st.error(f"{page.number}쪽: {page.message}")

    done = [page for page in job.pages if page.status == PAGE_RECOGNIZED]
    if not done:
        return None

    page = st.selectbox("검수·저장할 페이지", done, format_func=lambda item: f"{item.number}쪽 · {item.file_name}")
    review = inspect_page(job, page, template)
    render_results(job, page, template, review)
    return page, template, review


def render_export_tab(job: Job | None, selection: tuple[Page, Template, PageReview] | None) -> None:
    if job is None or selection is None:
        st.info("2단계에서 페이지를 인식한 뒤 저장할 수 있습니다.")
        return

    page, template, review = selection
    st.subheader(f"{page.number}쪽 XLSX 저장")
    st.caption(f"{page.file_name} · {template.label} · 데이터 {len(review.rows)}행")
    if review.rows:
        preview = pd.DataFrame(
            [[cell.value for cell in cells] for cells in review.rows.values()],
            columns=[column["label"] for column in template.columns],
            index=list(review.rows),
        )
        preview.index.name = "원본 행"
        st.dataframe(preview)

    duplicates_confirmed = False
    if review.duplicates:
        for rows in review.duplicates.values():
            st.warning(f"원본 {', '.join(map(str, rows))}행의 업무 키가 같습니다.")
        duplicates_confirmed = st.checkbox(
            "원본을 대조했고 중복 행을 그대로 저장합니다.",
            key=f"duplicates_{job.job_id}_{template.label}_{page.number}_{list(review.duplicates.items())}",
        )

    remaining = sum(bool(cell.issues) for cell in review.cells)
    if remaining:
        st.warning(f"검수가 필요한 셀이 {remaining}개 남았습니다. 2단계에서 원본과 대조하고 저장해 주세요.")
        return
    try:
        data = xlsx_bytes(review, template, duplicates_confirmed=duplicates_confirmed)
    except ReviewError as error:
        st.info(str(error))
        return

    st.download_button(
        "확정 XLSX 다운로드", data, file_name=f"{Path(page.file_name).stem}_검수.xlsx",
        mime=XLSX_MIME, type="primary", on_click="ignore",
    )


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
        selection = render_review_tab(job)
    with export:
        render_export_tab(job, selection)
    with settings:
        render_settings_tab(templates)


if __name__ == "__main__":
    main()
