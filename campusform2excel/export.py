"""검수를 마친 한 페이지를 XLSX로 내보낸다 (FR-17)."""
from __future__ import annotations

from decimal import Decimal
from io import BytesIO

from openpyxl import Workbook

from .review import PageReview, ReviewError
from .templates import NUMERIC_TYPES, Template

XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def xlsx_bytes(review: PageReview, template: Template, *, duplicates_confirmed: bool = False) -> bytes:
    if any(cell.issues for cell in review.cells):
        raise ReviewError("남은 검수 항목을 해결해 주세요.")
    if review.duplicates and not duplicates_confirmed:
        raise ReviewError("중복 행을 확인해 주세요.")
    if not review.rows:
        raise ReviewError("저장할 데이터 행이 없습니다.")

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "검수 결과"
    for col, column in enumerate(template.columns, start=1):
        sheet.cell(row=1, column=col, value=column["label"]).data_type = "s"

    for output_row, cells in enumerate(review.rows.values(), start=2):
        for cell, column in zip(cells, template.columns):
            target = sheet.cell(row=output_row, column=cell.box.col)
            if column["type"] == "student_id":
                target.number_format = "@"
            if cell.value:
                if column["type"] in NUMERIC_TYPES:
                    target.value = Decimal(cell.value)
                else:
                    target.value = cell.value
                    target.data_type = "s"

    sheet.freeze_panes = "A2"
    sheet.auto_filter.ref = sheet.dimensions
    with BytesIO() as output:
        workbook.save(output)
        return output.getvalue()
