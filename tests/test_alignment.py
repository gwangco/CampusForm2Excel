"""T-05 양식 선택, T-06 셀 매핑 (합성 사진 사전 점검)."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from support import quad_iou, simulate_photo

from campusform2excel.alignment import AlignmentError, align_page, crop_cell, make_cell_id
from campusform2excel.preprocess import normalize

SEEDS = range(10, 16)


def align(photo, template):
    normalized = normalize(photo)
    return align_page(normalized.image, template, 1, normalized.scale)


def true_quads(template, matrix):
    """양식의 정답 셀 좌표를 사진 좌표로 옮긴다."""
    corners = []
    for cell in template.cells():
        x0, y0, x1, y1 = cell.bbox
        corners += [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    points = np.array(corners, dtype=np.float64).reshape(1, -1, 2)
    return cv2.perspectiveTransform(points, matrix).reshape(-1, 4, 2)


def test_t05_each_form_can_be_selected(templates, blank_pages):
    for template in templates:
        aligned = align(simulate_photo(blank_pages[template.template_id], seed=3).image, template)
        assert len(aligned.cells) == len(template.cells())


def test_t05_wrong_selection_is_rejected_with_guidance(templates, blank_pages):
    for actual in templates:
        photo = simulate_photo(blank_pages[actual.template_id], seed=4).image
        for selected in templates:
            if selected.template_id == actual.template_id:
                continue
            with pytest.raises(AlignmentError, match="양식 선택을 확인") as caught:
                align(photo, selected)
            assert caught.value.code == "grid_mismatch"


@pytest.mark.parametrize("seed", SEEDS)
def test_t06_cells_map_to_page_row_col_with_iou(templates, blank_pages, seed):
    """셀 전체의 95% 이상이 정답 위치와 IoU 0.5 이상으로 대응해야 한다."""
    for template in templates:
        photo = simulate_photo(blank_pages[template.template_id], seed=seed)
        aligned = align(photo.image, template)

        expected = true_quads(template, photo.matrix)
        scores = [quad_iou(cell.source_quad, truth) for cell, truth in zip(aligned.cells, expected)]
        rate = sum(score >= 0.5 for score in scores) / len(scores)
        assert rate >= 0.95, f"{template.template_id}: 매핑 {rate:.3f}, 최저 IoU {min(scores):.3f}"


def test_t06_cell_ids_are_unique_and_ordered(templates, blank_pages):
    template = templates[0]
    aligned = align(simulate_photo(blank_pages[template.template_id], seed=5).image, template)
    ids = [cell.cell_id for cell in aligned.cells]
    assert len(set(ids)) == len(ids)
    assert ids[0] == make_cell_id(1, 1, 1)
    assert ids[-1] == make_cell_id(1, template.data_row_count, len(template.columns))


@pytest.mark.parametrize(
    ("rotate", "expected"),
    [(cv2.ROTATE_90_CLOCKWISE, 270), (cv2.ROTATE_180, 180), (cv2.ROTATE_90_COUNTERCLOCKWISE, 90)],
)
def test_t06_rotated_photo_is_turned_upright(templates, blank_pages, rotate, expected):
    template = templates[1]
    photo = simulate_photo(blank_pages[template.template_id], seed=6)
    aligned = align(cv2.rotate(photo.image, rotate), template)
    assert aligned.rotation == expected
    assert aligned.grid_score >= 0.9


def test_t06_source_location_traces_back_through_rotation_and_scaling(templates, blank_pages):
    """회전·축소가 함께 일어나도 셀의 원본 위치를 되찾아야 한다 (NFR-05)."""
    template = templates[0]
    photo = simulate_photo(blank_pages[template.template_id], seed=7, canvas=(2600, 3510))
    upright = photo.image
    rotated = cv2.rotate(upright, cv2.ROTATE_90_CLOCKWISE)
    enlarged = cv2.resize(rotated, None, fx=1.3, fy=1.3)

    aligned = align(enlarged, template)
    height = upright.shape[0]
    for cell, truth in zip(aligned.cells, true_quads(template, photo.matrix)):
        rotated_truth = np.column_stack((height - 1 - truth[:, 1], truth[:, 0])) * 1.3
        assert quad_iou(cell.source_quad, rotated_truth) >= 0.5


def test_t06_page_without_table_fails_with_guidance():
    empty = np.full((2000, 1500, 3), 235, np.uint8)
    with pytest.raises(AlignmentError, match="다시 촬영") as caught:
        align(empty, _any_template())
    assert caught.value.code == "table_not_found"


def test_t06_crop_excludes_grid_lines(templates, blank_pages):
    template = templates[0]
    page = blank_pages[template.template_id]
    for cell in template.cells()[:10]:
        assert crop_cell(page, cell.bbox, margin=7).min() == 255


def _any_template():
    from campusform2excel.templates import list_templates

    return list_templates()[0]
