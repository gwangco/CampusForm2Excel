"""FR-07·AI-01 출력 계약. 가짜 엔진으로 실패 경로까지 결정적으로 확인한다."""
from __future__ import annotations

import cv2
import numpy as np
import pytest
from conftest import FakeEngine
from run_t07 import contract_violations

from campusform2excel.ocr import (
    REASON_DUPLICATE_CELL_ID,
    REASON_INFERENCE_ERROR,
    REASON_INVALID_CELL_ID,
    REASON_INVALID_IMAGE,
    REASON_LOW_CONFIDENCE,
    REASON_MARK_PENDING,
    REASON_NO_SCORE,
    REASON_NO_TEXT,
    STATUS_DEFERRED,
    STATUS_EMPTY,
    STATUS_FAILED,
    STATUS_RECOGNIZED,
    STATUS_UNREADABLE,
    CellInput,
    ModelError,
    extract_lines,
    load_settings,
    recognize_cells,
    verify_model,
)

SETTINGS = load_settings()


def blank_cell(height=78, width=286):
    return np.full((height, width, 3), 255, np.uint8)


def written_cell(text="2400731", height=78, width=286):
    image = blank_cell(height, width)
    cv2.putText(image, text, (20, height // 2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 3)
    return image


def cell(cell_id="p01-r01-c01", image=None, column_type="student_id"):
    image = written_cell() if image is None else image
    return CellInput(cell_id, 1, 1, 1, (0, 0, 10, 10), image, column_type, "attendance", "1")


def recognize(cells, engine=None):
    return recognize_cells(cells, engine or FakeEngine(), SETTINGS)


def test_one_result_per_input_in_order():
    cells = [cell(f"id-{i}", written_cell() if i % 2 else blank_cell()) for i in range(30)]
    results = recognize(cells)
    assert [r.cell_id for r in results] == [c.cell_id for c in cells]
    assert not [problem for c, r in zip(cells, results) for problem in contract_violations(c.cell_id, r)]


def test_recognized_keeps_text_score_and_traceability():
    result = recognize([cell()], FakeEngine(default=("2400731", 0.91)))[0]
    assert (result.raw_text, result.confidence, result.status) == ("2400731", 0.91, STATUS_RECOGNIZED)
    assert not result.review_required
    assert result.model_name == "fake-model" and result.model_sha256 == "fake-sha256"
    assert result.config_version == SETTINGS["config_version"]
    assert (result.template_id, result.template_version) == ("attendance", "1")


def test_blank_cell_is_empty_without_calling_the_model():
    engine = FakeEngine()
    result = recognize([cell(image=blank_cell())], engine)[0]
    assert (result.status, result.confidence, result.raw_text) == (STATUS_EMPTY, None, "")
    assert not result.review_required
    assert engine.calls == []


def test_ink_without_text_is_unreadable_and_goes_to_review():
    result = recognize([cell()], FakeEngine(default=("", 0.0)))[0]
    assert result.status == STATUS_UNREADABLE
    assert result.review_required and REASON_NO_TEXT in result.reasons


@pytest.mark.parametrize("score", [None, float("nan")])
def test_missing_score_is_null_and_goes_to_review(score):
    result = recognize([cell()], FakeEngine(default=("가나", score)))[0]
    assert result.status == STATUS_RECOGNIZED and result.confidence is None
    assert result.review_required and REASON_NO_SCORE in result.reasons


def test_low_confidence_goes_to_review_but_keeps_text():
    result = recognize([cell()], FakeEngine(default=("240O731", 0.62)))[0]
    assert result.raw_text == "240O731"
    assert result.review_required and REASON_LOW_CONFIDENCE in result.reasons


def test_batch_failure_retries_each_cell_and_only_bad_cell_fails():
    engine = FakeEngine(responses=[RuntimeError("batch"), ("하나", 0.9), RuntimeError("bad"), ("셋", 0.9)])
    results = recognize([cell("a"), cell("b"), cell("c")], engine)
    assert [r.status for r in results] == [STATUS_RECOGNIZED, STATUS_FAILED, STATUS_RECOGNIZED]
    assert results[1].reasons == [REASON_INFERENCE_ERROR]
    assert results[1].confidence is None and results[1].review_required
    assert engine.calls == [3, 1, 1, 1]


def test_wrong_output_count_is_recovered():
    class ShortEngine(FakeEngine):
        def predict(self, images):
            self.calls.append(len(images))
            return [("하나", 0.9)]

    results = recognize([cell("a"), cell("b")], ShortEngine())
    assert [r.status for r in results] == [STATUS_RECOGNIZED, STATUS_RECOGNIZED]


def test_total_engine_failure_keeps_every_cell():
    engine = FakeEngine(default=RuntimeError("down"))
    results = recognize([cell(f"id-{i}") for i in range(5)], engine)
    assert len(results) == 5
    assert all(r.status == STATUS_FAILED for r in results)


@pytest.mark.parametrize(
    "bad_image",
    [None, np.zeros((0, 0, 3), np.uint8), np.zeros((10, 10), np.uint8), "not an array"],
)
def test_invalid_image_fails_without_losing_the_cell(bad_image):
    bad = CellInput("bad", 1, 1, 1, (0, 0, 1, 1), bad_image, "text", "attendance", "1")
    results = recognize([bad, cell("good")])
    assert [r.cell_id for r in results] == ["bad", "good"]
    assert results[0].status == STATUS_FAILED and results[0].reasons == [REASON_INVALID_IMAGE]
    assert results[1].status == STATUS_RECOGNIZED


def test_invalid_and_duplicate_cell_ids_are_reported():
    results = recognize([cell(""), cell("same"), cell("same")])
    assert results[0].reasons == [REASON_INVALID_CELL_ID]
    assert results[1].status == STATUS_RECOGNIZED
    assert results[2].reasons == [REASON_DUPLICATE_CELL_ID]


def test_check_column_is_deferred_and_stays_in_review():
    engine = FakeEngine()
    result = recognize([cell(column_type="check")], engine)[0]
    assert result.status == STATUS_DEFERRED and result.mark_state is None
    assert result.review_required and result.reasons == [REASON_MARK_PENDING]
    assert engine.calls == []


def test_candidates_are_empty_until_dictionary_feature():
    assert recognize([cell()])[0].candidates == []


def test_two_lines_are_read_separately_and_joined_in_order():
    image = blank_cell(120, 286)
    cv2.putText(image, "AAAA", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 3)
    cv2.putText(image, "BBBB", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 3)
    assert len(extract_lines(image, SETTINGS)) == 2

    engine = FakeEngine(responses=[("윗줄", 0.9), ("아랫줄", 0.7)])
    result = recognize([cell(image=image)], engine)[0]
    assert result.raw_text == "윗줄 아랫줄"
    assert result.confidence == 0.7
    assert REASON_LOW_CONFIDENCE in result.reasons


def test_one_failed_line_fails_the_whole_cell():
    image = blank_cell(120, 286)
    cv2.putText(image, "AAAA", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 3)
    cv2.putText(image, "BBBB", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (20, 20, 20), 3)
    engine = FakeEngine(responses=[RuntimeError("batch"), ("윗줄", 0.9), RuntimeError("line")])
    result = recognize([cell(image=image)], engine)[0]
    assert result.status == STATUS_FAILED and result.raw_text == ""


def test_stacked_strokes_of_one_character_are_not_split_into_lines():
    """'를'처럼 위아래로 떨어진 획을 여러 줄로 나누지 않는다."""
    image = blank_cell(78, 286)
    for top in (15, 33, 51):
        cv2.rectangle(image, (120, top), (170, top + 7), (20, 20, 20), -1)
    assert len(extract_lines(image, SETTINGS)) == 1


def test_grid_line_residue_is_not_mistaken_for_writing():
    image = blank_cell()
    image[:3, :] = 20  # 잘림 경계에 남은 윗 격자선
    image[:, -3:] = 20  # 잘림 경계에 남은 오른쪽 격자선
    assert extract_lines(image, SETTINGS) == []


def test_specks_are_not_mistaken_for_writing():
    image = blank_cell()
    for x, y in ((40, 20), (90, 50), (200, 30)):
        image[y:y + 2, x:x + 2] = 30
    assert extract_lines(image, SETTINGS) == []


def test_tampered_model_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("CF2E_MODEL_HOME", str(tmp_path))
    with pytest.raises(ModelError, match="setup_model"):
        verify_model()
