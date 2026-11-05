"""T-07: 정상·빈칸·흐림 30셀의 응답을 비교해 FR-07 계약을 검사한다.

    python scripts/run_t07.py

합격 기준은 '셀 ID·원본 문자열·점수/null·상태·모델 정보 누락 0건'이다.
입력이 글꼴 합성 사진이므로 reference_exact_match는 참고값이며 손글씨 성능이 아니다.
"""
from __future__ import annotations

import json
import platform
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from campusform2excel.alignment import align_page, crop_cell  # noqa: E402
from campusform2excel.imaging import read_image  # noqa: E402
from campusform2excel.ocr import (  # noqa: E402
    STATUS_DEFERRED,
    STATUS_EMPTY,
    STATUS_FAILED,
    STATUS_RECOGNIZED,
    STATUS_UNREADABLE,
    CellInput,
    CellResult,
    PaddleEngine,
    load_model_lock,
    load_settings,
    recognize_cells,
)
from campusform2excel.preprocess import normalize  # noqa: E402
from campusform2excel.templates import load_template  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
RECORD_PATH = ROOT / "docs" / "시험기록" / "T-07_30셀_계약시험.json"
VALID_STATUS = {STATUS_RECOGNIZED, STATUS_EMPTY, STATUS_UNREADABLE, STATUS_FAILED, STATUS_DEFERRED}


def contract_violations(cell_id: str, result: CellResult) -> list[str]:
    """FR-07·AI-01의 출력 계약을 어긴 항목을 돌려준다."""
    problems = []
    if result.cell_id != cell_id:
        problems.append("cell_id 불일치")
    if not isinstance(result.raw_text, str):
        problems.append("raw_text 누락")
    if result.confidence is not None and not 0.0 <= result.confidence <= 1.0:
        problems.append("confidence 범위 오류")
    if result.status not in VALID_STATUS:
        problems.append("status 오류")
    if result.status == STATUS_RECOGNIZED and not result.raw_text:
        problems.append("recognized인데 문자열 없음")
    for name in ("model_name", "model_sha256", "config_version", "template_id", "template_version"):
        if not getattr(result, name):
            problems.append(f"{name} 누락")
    return problems


def run(engine: PaddleEngine | None = None) -> dict[str, Any]:
    settings = load_settings()
    manifest = json.loads((FIXTURES / "t07_manifest.json").read_text(encoding="utf-8"))

    load_seconds = 0.0
    if engine is None:
        started = time.perf_counter()
        engine = PaddleEngine(settings)
        load_seconds = time.perf_counter() - started

    inputs: list[CellInput] = []
    kinds: dict[str, dict[str, Any]] = {}
    pages = {}
    for item in manifest:
        key = item["file"]
        if key not in pages:
            template = load_template(item["template_id"], item["version"])
            normalized = normalize(read_image(FIXTURES / key))
            pages[key] = (template, align_page(normalized.image, template, 1, normalized.scale))
        template, aligned = pages[key]
        cell = next(box for box in aligned.cells if box.cell_id == item["cell_id"])
        unique_id = f"{item['template_id']}/{cell.cell_id}"
        inputs.append(CellInput(
            unique_id, cell.page, cell.row, cell.col, cell.bbox,
            crop_cell(aligned.image, cell.bbox, settings["cell_margin_px"]),
            cell.column_type, template.template_id, template.version,
        ))
        kinds[unique_id] = item

    started = time.perf_counter()
    results = recognize_cells(inputs, engine, settings)
    inference_seconds = time.perf_counter() - started

    rows = []
    violations: list[dict[str, Any]] = []
    for cell, result in zip(inputs, results):
        item = kinds[cell.cell_id]
        problems = contract_violations(cell.cell_id, result)
        if problems:
            violations.append({"cell_id": cell.cell_id, "problems": problems})
        rows.append({
            "cell_id": cell.cell_id, "kind": item["kind"], "truth": item["truth"],
            "raw_text": result.raw_text, "confidence": result.confidence,
            "status": result.status, "reasons": result.reasons,
        })

    by_kind = {}
    for kind in ("normal", "blur", "blank"):
        subset = [row for row in rows if row["kind"] == kind]
        by_kind[kind] = {
            "cells": len(subset),
            "status": dict(Counter(row["status"] for row in subset)),
            "reference_exact_match": sum(row["raw_text"] == row["truth"] for row in subset),
        }

    missing = len(inputs) - len(results)
    lock = load_model_lock()
    return {
        "test_id": "T-07",
        "requirement": "FR-07",
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "input_note": "글꼴 합성 사진. 손글씨가 아니며 개발·최종·외부 데이터가 아니다.",
        "total_cells": len(inputs),
        "returned_results": len(results),
        "missing_results": missing,
        "contract_violations": violations,
        "passed": missing == 0 and not violations and len(inputs) == 30,
        "by_kind": by_kind,
        "model": {"name": lock["model_name"], "revision": lock["revision"],
                  "weights_sha256": lock["files"]["inference.pdiparams"]["sha256"]},
        "config_version": settings["config_version"],
        "timing_seconds": {"model_load": round(load_seconds, 2), "inference_30_cells": round(inference_seconds, 2)},
        "environment": {"python": platform.python_version(), "machine": platform.processor(),
                        "cpu_threads": settings["cpu_threads"]},
        "cells": rows,
    }


def main() -> int:
    report = run()
    RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECORD_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"T-07 {'합격' if report['passed'] else '불합격'}: "
          f"{report['returned_results']}/{report['total_cells']}셀, 계약 위반 {len(report['contract_violations'])}건")
    for kind, summary in report["by_kind"].items():
        print(f"  {kind}: {summary}")
    print(f"  시간: {report['timing_seconds']}")
    print(f"  기록: {RECORD_PATH}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
