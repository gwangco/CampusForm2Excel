"""양식 설정으로 인쇄용 빈 양식을 그린다. 좌표의 기준은 언제나 설정 파일이다."""
from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from .templates import Template

FONT_REGULAR = "C:/Windows/Fonts/malgun.ttf"
FONT_BOLD = "C:/Windows/Fonts/malgunbd.ttf"
OUTER_LINE_PX = 6
INNER_LINE_PX = 3


def render_blank_form(template: Template) -> Image.Image:
    width, height = template.page_size
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    x0, y0, x1, y1 = template.table_bbox

    title = ImageFont.truetype(FONT_BOLD, 62)
    body = ImageFont.truetype(FONT_REGULAR, 28)
    header = ImageFont.truetype(FONT_BOLD, 34)
    draw.text((width / 2, 120), template.name, font=title, fill="black", anchor="mm")
    draw.text(
        (width / 2, 215),
        "칸 안에 또박또박 적어 주세요. 촬영할 때는 표의 네 모서리가 모두 보이게 해 주세요.",
        font=body, fill="black", anchor="mm",
    )

    for x in template.x_edges[1:-1]:
        draw.line([(x, y0), (x, y1)], fill="black", width=INNER_LINE_PX)
    for y in template.y_edges[1:-1]:
        draw.line([(x0, y), (x1, y)], fill="black", width=INNER_LINE_PX)
    draw.rectangle([x0, y0, x1, y1], outline="black", width=OUTER_LINE_PX)

    header_bottom = template.y_edges[template.header_rows]
    for left, right, column in zip(template.x_edges, template.x_edges[1:], template.columns):
        draw.text(((left + right) / 2, (y0 + header_bottom) / 2), column["label"],
                  font=header, fill="black", anchor="mm")

    draw.text(
        (width / 2, height - 45),
        f"CampusForm2Excel · {template.label} · 가상의 이름과 학번으로 작성",
        font=body, fill="black", anchor="mm",
    )
    return image
