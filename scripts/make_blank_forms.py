"""등록된 양식 설정으로 인쇄용 빈 양식(PNG·PDF)을 만든다.

    python scripts/make_blank_forms.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from campusform2excel.forms import render_blank_form  # noqa: E402
from campusform2excel.paths import FORMS_DIR  # noqa: E402
from campusform2excel.templates import list_templates  # noqa: E402


def main() -> None:
    FORMS_DIR.mkdir(exist_ok=True)
    pages = []
    for template in list_templates():
        image = render_blank_form(template)
        path = FORMS_DIR / f"{template.template_id}_v{template.version}.png"
        image.save(path, dpi=(200, 200))
        pages.append(image)
        print(path.name)

    pdf = FORMS_DIR / "빈양식_3종_인쇄용.pdf"
    pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=200)
    print(pdf.name)


if __name__ == "__main__":
    main()
