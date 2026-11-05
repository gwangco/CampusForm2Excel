"""양식(템플릿) 설정의 등록·조회·검증 (FR-01, FR-02, FR-05)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .paths import TEMPLATE_DIR

COLUMN_TYPES = {"student_id", "name", "date", "text", "equipment", "int", "score", "total", "check"}
NUMERIC_TYPES = {"int", "score", "total"}


class TemplateError(ValueError):
    """양식 설정이 규칙에 맞지 않을 때 발생한다. errors에 사유 목록을 담는다."""

    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass(frozen=True)
class CellSpec:
    """양식에 정의된 데이터 셀 하나. row·col은 1부터 시작한다."""

    row: int
    col: int
    column_key: str
    column_type: str
    bbox: tuple[int, int, int, int]  # 정규 페이지 좌표 (x0, y0, x1, y1)


@dataclass(frozen=True)
class Template:
    """설정 파일 하나에 대응하는 양식. raw가 저장되는 원본이다."""

    raw: dict[str, Any]

    @property
    def template_id(self) -> str:
        return self.raw["template_id"]

    @property
    def name(self) -> str:
        return self.raw["name"]

    @property
    def version(self) -> str:
        return self.raw["version"]

    @property
    def label(self) -> str:
        return f"{self.name} v{self.version}"

    @property
    def page_size(self) -> tuple[int, int]:
        return self.raw["page"]["width"], self.raw["page"]["height"]

    @property
    def columns(self) -> list[dict[str, Any]]:
        return self.raw["columns"]

    @property
    def x_edges(self) -> list[int]:
        return self.raw["grid"]["x_edges"]

    @property
    def y_edges(self) -> list[int]:
        return self.raw["grid"]["y_edges"]

    @property
    def header_rows(self) -> int:
        return self.raw["grid"]["header_rows"]

    @property
    def data_row_count(self) -> int:
        return len(self.y_edges) - 1 - self.header_rows

    @property
    def table_bbox(self) -> tuple[int, int, int, int]:
        return self.x_edges[0], self.y_edges[0], self.x_edges[-1], self.y_edges[-1]

    def cells(self, x_edges: list[int] | None = None, y_edges: list[int] | None = None) -> list[CellSpec]:
        """데이터 셀 목록을 만든다. 정합에서 보정한 격자선을 넘기면 그 좌표를 쓴다."""
        xs = x_edges or self.x_edges
        ys = y_edges or self.y_edges
        specs = []
        for row in range(1, self.data_row_count + 1):
            top = ys[self.header_rows + row - 1]
            bottom = ys[self.header_rows + row]
            for col, column in enumerate(self.columns, start=1):
                specs.append(
                    CellSpec(row, col, column["key"], column["type"], (xs[col - 1], top, xs[col], bottom))
                )
        return specs


def validate_template(raw: dict[str, Any]) -> list[str]:
    """설정의 구조와 규칙을 검사해 오류 목록을 돌려준다. 빈 목록이면 정상이다."""
    required = ("template_id", "name", "version", "page", "grid", "columns", "business_key")
    missing = [key for key in required if key not in raw]
    if missing:
        return [f"필수 항목 누락: {', '.join(missing)}"]

    errors: list[str] = []
    errors += _validate_grid(raw)
    errors += _validate_columns(raw)
    return errors


def _validate_grid(raw: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    page, grid, columns = raw["page"], raw["grid"], raw["columns"]
    xs, ys = grid.get("x_edges", []), grid.get("y_edges", [])
    header_rows = grid.get("header_rows", 0)

    if any(b <= a for a, b in zip(xs, xs[1:])) or any(b <= a for a, b in zip(ys, ys[1:])):
        errors.append("격자선 좌표는 오름차순이어야 합니다")
    if len(xs) != len(columns) + 1:
        errors.append(f"열 수({len(columns)})와 세로 격자선 수({len(xs)})가 맞지 않습니다")
    if len(ys) - 1 <= header_rows:
        errors.append("데이터 행이 없습니다")
    if xs and ys and (xs[0] < 0 or ys[0] < 0 or xs[-1] > page["width"] or ys[-1] > page["height"]):
        errors.append("표 영역이 페이지 밖으로 나갑니다")
    return errors


def _validate_columns(raw: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    columns = raw["columns"]
    keys = [column.get("key") for column in columns]
    types = {column.get("key"): column.get("type") for column in columns}

    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        errors.append(f"중복된 열: {', '.join(duplicates)}")

    for column in columns:
        key, kind = column.get("key"), column.get("type")
        if kind not in COLUMN_TYPES:
            errors.append(f"{key}: 알 수 없는 열 타입 {kind}")
        if kind == "student_id" and not (isinstance(column.get("length"), int) and column["length"] > 0):
            errors.append(f"{key}: 학번 길이 규칙이 필요합니다")
        if kind == "date" and not column.get("date_formats"):
            errors.append(f"{key}: 날짜 형식 규칙이 필요합니다")
        if kind in ("score", "total") and column.get("min", 0) > column.get("max", -1):
            errors.append(f"{key}: 점수 범위가 모순됩니다")
        if kind == "total":
            sources = column.get("sum_of") or []
            if not sources:
                errors.append(f"{key}: 합계 대상 열이 필요합니다")
            for source in sources:
                if types.get(source) not in NUMERIC_TYPES:
                    errors.append(f"{key}: 합계 대상 {source}가 숫자 열이 아닙니다")

    for key in raw["business_key"]:
        if key not in keys:
            errors.append(f"업무 키 {key}가 열 목록에 없습니다")
    return errors


def template_path(template_id: str, version: str, directory: Path = TEMPLATE_DIR) -> Path:
    return directory / f"{template_id}_v{version}.json"


def save_template(raw: dict[str, Any], directory: Path = TEMPLATE_DIR) -> Template:
    """양식을 검증한 뒤 등록한다. 같은 버전을 다른 내용으로 덮어쓰지 않는다."""
    errors = validate_template(raw)
    if errors:
        raise TemplateError(errors)

    path = template_path(raw["template_id"], raw["version"], directory)
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != raw:
        raise TemplateError([f"이미 등록된 버전입니다: {path.name}. 변경하려면 버전을 올려 주세요"])

    directory.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return Template(raw)


def load_template(template_id: str, version: str, directory: Path = TEMPLATE_DIR) -> Template:
    """선택한 양식과 버전을 읽는다. 없거나 잘못된 설정은 거절한다."""
    path = template_path(template_id, version, directory)
    if not path.exists():
        raise TemplateError([f"등록되지 않은 양식입니다: {template_id} v{version}"])
    raw = json.loads(path.read_text(encoding="utf-8"))
    errors = validate_template(raw)
    if errors:
        raise TemplateError(errors)
    return Template(raw)


def list_templates(directory: Path = TEMPLATE_DIR) -> list[Template]:
    """등록된 양식을 파일 이름순으로 돌려준다."""
    templates = []
    for path in sorted(directory.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        templates.append(load_template(raw["template_id"], raw["version"], directory))
    return templates
