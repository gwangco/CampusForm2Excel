"""T-07용 합성 사진과 정답 목록을 만든다. 생성 결과는 tests/fixtures에 저장되어 있다.

    python tests/make_fixtures.py

글꼴은 이 PC의 휴먼편지체·휴먼아미체를 쓴다. 다른 PC에서는 재생성이 필요 없다.
모든 이름과 학번은 가상이다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from support import simulate_photo  # noqa: E402

from campusform2excel.alignment import make_cell_id  # noqa: E402
from campusform2excel.forms import render_blank_form  # noqa: E402
from campusform2excel.templates import load_template  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
FONTS = ["C:/Windows/Fonts/HMFMPYUN.TTF", "C:/Windows/Fonts/HMKMAMI.TTF"]
INK = (28, 32, 60)
BLUR_SIGMA = 2.2

ATTENDANCE_ROWS = [
    ["2400731", "김가람", "2026.11.03", "", ""],
    ["2400842", "이나래", "2026.11.03", "", "지각"],
    ["2400953", "박다온", "2026.11.03", "", ""],
    ["2401064", "최라온", "2026.11.03", "", "공결"],
]
RENTAL_ROWS = [
    ["11/03", "2400731", "김가람", "오실로스코프", "1", "11/05", ""],
    ["11/04", "2400842", "이나래", "멀티미터", "2", "", ""],
    ["11/04", "2400953", "박다온", "납땜인두", "1", "11/06", ""],
]
EVALUATION_ROWS = [
    ["2400731", "김가람", "11/03", "22", "20", "24", "18", "84"],
    ["2400842", "이나래", "11/03", "25", "19", "21", "23", "88"],
    ["2400953", "박다온", "11/03", "17", "22", "20", "16", "75"],
]

# 양식별 시험 셀 (행, 열). 합계 정상 10·흐림 10·빈칸 10 = 30셀이다.
SELECTION = {
    "attendance": {
        "rows": ATTENDANCE_ROWS,
        "normal": [(1, 1), (1, 2), (1, 3), (2, 5)],
        "blur": [(3, 1), (3, 2), (3, 3)],
        "blank": [(8, 1), (8, 3), (9, 5)],
    },
    "equipment_rental": {
        "rows": RENTAL_ROWS,
        "normal": [(1, 2), (1, 4), (1, 5)],
        "blur": [(2, 2), (2, 3), (2, 4)],
        "blank": [(7, 1), (7, 4), (8, 2), (8, 6)],
    },
    "lab_evaluation": {
        "rows": EVALUATION_ROWS,
        "normal": [(1, 1), (1, 4), (1, 8)],
        "blur": [(2, 1), (2, 2), (2, 4), (2, 8)],
        "blank": [(7, 1), (7, 5), (8, 8)],
    },
}


def fill_page(template_id: str, spec: dict, rng: np.random.Generator) -> np.ndarray:
    template = load_template(template_id, "1")
    image = render_blank_form(template)
    draw = ImageDraw.Draw(image)
    boxes = {(cell.row, cell.col): cell.bbox for cell in template.cells()}

    for row, values in enumerate(spec["rows"], start=1):
        for col, text in enumerate(values, start=1):
            if not text:
                continue
            x0, y0, x1, y1 = boxes[(row, col)]
            font = ImageFont.truetype(FONTS[int(rng.integers(len(FONTS)))], int(rng.integers(40, 47)))
            center = ((x0 + x1) / 2 + rng.uniform(-8, 8), (y0 + y1) / 2 + rng.uniform(-5, 5))
            draw.text(center, text, font=font, fill=INK, anchor="mm")

    page = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    for row, col in spec["blur"]:
        x0, y0, x1, y1 = boxes[(row, col)]
        inner = page[y0 + 5:y1 - 5, x0 + 5:x1 - 5]
        page[y0 + 5:y1 - 5, x0 + 5:x1 - 5] = cv2.GaussianBlur(inner, (0, 0), BLUR_SIGMA)
    return page


def main() -> None:
    FIXTURES.mkdir(exist_ok=True)
    manifest = []
    for seed, (template_id, spec) in enumerate(SELECTION.items(), start=7):
        rng = np.random.default_rng(seed)
        photo = simulate_photo(fill_page(template_id, spec, rng), seed=seed)
        file_name = f"{template_id}_synthetic.jpg"
        ok, encoded = cv2.imencode(".jpg", photo.image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        assert ok
        (FIXTURES / file_name).write_bytes(encoded.tobytes())

        for kind in ("normal", "blur", "blank"):
            for row, col in spec[kind]:
                truth = "" if kind == "blank" else spec["rows"][row - 1][col - 1]
                manifest.append({
                    "file": file_name, "template_id": template_id, "version": "1",
                    "cell_id": make_cell_id(1, row, col), "kind": kind, "truth": truth,
                })

    path = FIXTURES / "t07_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    counts = {kind: sum(item["kind"] == kind for item in manifest) for kind in ("normal", "blur", "blank")}
    print(f"합계 {len(manifest)}셀: {counts}")


if __name__ == "__main__":
    main()
