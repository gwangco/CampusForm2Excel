"""T-01 양식 등록·조회, T-02 열 규칙 저장."""
from __future__ import annotations

import copy

import pytest

from campusform2excel.templates import (
    TemplateError,
    list_templates,
    load_template,
    save_template,
    validate_template,
)


def test_t01_three_forms_registered(templates):
    assert {t.template_id for t in templates} == {"attendance", "equipment_rental", "lab_evaluation"}
    assert {t.name for t in templates} == {"출석표", "장비 대여 장부", "실습 평가표"}


def test_t01_saved_form_reads_back_identical(templates, tmp_path):
    for template in templates:
        saved = save_template(copy.deepcopy(template.raw), tmp_path)
        loaded = load_template(saved.template_id, saved.version, tmp_path)
        assert (loaded.name, loaded.version) == (template.name, template.version)
        assert loaded.x_edges == template.x_edges
        assert loaded.y_edges == template.y_edges
        assert [c.bbox for c in loaded.cells()] == [c.bbox for c in template.cells()]
    assert len(list_templates(tmp_path)) == 3


def test_t01_cell_grid_covers_every_data_cell(templates):
    for template in templates:
        cells = template.cells()
        assert len(cells) == template.data_row_count * len(template.columns)
        assert len({(c.row, c.col) for c in cells}) == len(cells)
        x0, y0, x1, y1 = template.table_bbox
        assert all(x0 <= c.bbox[0] < c.bbox[2] <= x1 and y0 < c.bbox[1] < c.bbox[3] <= y1 for c in cells)


def test_t01_unknown_form_rejected():
    with pytest.raises(TemplateError):
        load_template("no_such_form", "1")


def test_t01_same_version_cannot_be_overwritten(templates, tmp_path):
    raw = copy.deepcopy(templates[0].raw)
    save_template(raw, tmp_path)
    raw["name"] = "바뀐 이름"
    with pytest.raises(TemplateError, match="이미 등록된 버전"):
        save_template(raw, tmp_path)


def test_t02_rules_survive_save_and_reload(templates, tmp_path):
    for template in templates:
        save_template(copy.deepcopy(template.raw), tmp_path)
        loaded = load_template(template.template_id, template.version, tmp_path)
        assert loaded.columns == template.columns


def test_t02_required_rule_types_present(templates):
    by_id = {t.template_id: t for t in templates}
    columns = {c["key"]: c for c in by_id["lab_evaluation"].columns}
    assert columns["student_id"]["length"] == 7
    assert columns["score1"]["max"] == 25
    assert columns["total"]["sum_of"] == ["score1", "score2", "score3", "score4"]
    assert columns["eval_date"]["date_formats"]
    assert any(c["type"] == "check" for c in by_id["attendance"].columns)
    assert any(c.get("dictionary") == "equipment" for c in by_id["equipment_rental"].columns)


def _broken(templates, change):
    raw = copy.deepcopy(next(t for t in templates if t.template_id == "lab_evaluation").raw)
    change(raw)
    return validate_template(raw)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda r: r["columns"][1].update(key="student_id"), "중복된 열"),
        (lambda r: r["columns"][0].pop("length"), "학번 길이"),
        (lambda r: r["columns"][2].pop("date_formats"), "날짜 형식"),
        (lambda r: r["columns"][3].update(min=30), "점수 범위"),
        (lambda r: r["columns"][7].update(sum_of=["name"]), "숫자 열"),
        (lambda r: r["columns"][0].update(type="unknown"), "알 수 없는 열 타입"),
        (lambda r: r.update(business_key=["no_such"]), "업무 키"),
        (lambda r: r["grid"]["x_edges"].pop(), "세로 격자선"),
        (lambda r: r["grid"]["y_edges"].reverse(), "오름차순"),
        (lambda r: r["page"].update(width=1000), "페이지 밖"),
    ],
)
def test_t02_contradictory_rules_rejected(templates, change, message):
    assert any(message in error for error in _broken(templates, change))
