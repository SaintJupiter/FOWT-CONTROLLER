from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "paper_figures_current" / "chapter1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FONT_PATH = Path("/System/Library/Fonts/STHeiti Medium.ttc")


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size=size)


def draw_centered_lines(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill: str = "#111827",
    spacing: int = 10,
) -> None:
    x, y, w, h = box
    lines = text.split("\n")
    heights = []
    widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=fnt)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    total_h = sum(heights) + spacing * (len(lines) - 1)
    cy = y + (h - total_h) / 2
    for line, lw, lh in zip(lines, widths, heights):
        draw.text((x + (w - lw) / 2, cy), line, font=fnt, fill=fill)
        cy += lh + spacing


def rounded_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill: str,
    outline: str = "#52616f",
    width: int = 4,
    radius: int = 26,
) -> None:
    x, y, w, h = box
    draw.rounded_rectangle((x, y, x + w, y + h), radius=radius, fill=fill, outline=outline, width=width)
    draw_centered_lines(draw, box, text, fnt)


def arrowhead(draw: ImageDraw.ImageDraw, x1: float, y1: float, x2: float, y2: float, color: str, size: int = 22) -> None:
    angle = math.atan2(y2 - y1, x2 - x1)
    pts = [
        (x2, y2),
        (x2 - size * math.cos(angle - math.pi / 6), y2 - size * math.sin(angle - math.pi / 6)),
        (x2 - size * math.cos(angle + math.pi / 6), y2 - size * math.sin(angle + math.pi / 6)),
    ]
    draw.polygon(pts, fill=color)


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str = "#374151",
    width: int = 5,
    dashed: bool = False,
) -> None:
    x1, y1 = start
    x2, y2 = end
    if dashed:
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length
        step = 28
        dash = 16
        t = 0
        while t < length - 24:
            a = t
            b = min(t + dash, length - 24)
            draw.line((x1 + ux * a, y1 + uy * a, x1 + ux * b, y1 + uy * b), fill=color, width=width)
            t += step
    else:
        draw.line((x1, y1, x2, y2), fill=color, width=width)
    arrowhead(draw, x1, y1, x2, y2, color)


def make_png() -> Path:
    w, h = 2400, 1450
    img = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(img)

    title_f = font(56)
    box_f = font(38)
    small_f = font(31)
    note_f = font(34)

    title = "短时风况预测驱动的主动压载监督调节框架"
    tw = draw.textbbox((0, 0), title, font=title_f)[2]
    draw.text(((w - tw) / 2, 58), title, font=title_f, fill="#111827")

    blue = "#eaf3fb"
    green = "#ecf7ee"
    warm = "#f7f2ea"
    gray = "#f5f6f7"
    line = "#3f4d5a"
    accent = "#2563eb"

    top1 = (120, 210, 460, 160)
    top2 = (720, 210, 460, 160)
    top3 = (1320, 210, 580, 160)
    supervisor = (760, 525, 880, 285)
    fb = (120, 1040, 430, 160)
    ctrl = (690, 1040, 470, 160)
    exec_box = (1300, 1040, 500, 160)
    plant = (1940, 1040, 380, 160)

    rounded_box(draw, top1, "历史风况与平台状态\n风速、风向、姿态信息", box_f, blue)
    rounded_box(draw, top2, "短时风况预测\n未来60分钟风矢量", box_f, blue)
    rounded_box(draw, top3, "风险识别与压力代理\n0—20 / 20—40 / 40—60分钟", box_f, blue)

    rounded_box(draw, supervisor, "预测监督层", font(44), green, outline="#3f7a4c", width=4, radius=30)
    pill_y = 625
    pill_w, pill_h = 210, 68
    for x, label in [(845, "经济保持"), (1095, "提前调节"), (1345, "安全响应")]:
        draw.rounded_rectangle((x, pill_y, x + pill_w, pill_y + pill_h), radius=24, fill="#ffffff", outline="#3f7a4c", width=3)
        draw_centered_lines(draw, (x, pill_y, pill_w, pill_h), label, small_f, fill="#14532d")
    draw_centered_lines(
        draw,
        (835, 720, 730, 72),
        "输出外层调节参数：姿态响应死区、动作预算、安全/经济权重",
        small_f,
        fill="#1f2937",
    )

    rounded_box(draw, fb, "姿态反馈\n横摇、纵摇偏差", box_f, gray)
    rounded_box(draw, ctrl, "基础闭环控制器\n生成压载调节需求", box_f, gray)
    rounded_box(draw, exec_box, "压载分配与水泵执行\n三舱目标水量与泵送动作", box_f, warm)
    rounded_box(draw, plant, "浮式平台\n姿态响应更新", box_f, gray)

    arrow(draw, (580, 290), (720, 290), line)
    arrow(draw, (1180, 290), (1320, 290), line)
    arrow(draw, (1610, 370), (1320, 525), accent, dashed=True)
    arrow(draw, (1115, 810), (945, 1040), accent, dashed=True)
    arrow(draw, (550, 1120), (690, 1120), line)
    arrow(draw, (1160, 1120), (1300, 1120), line)
    arrow(draw, (1800, 1120), (1940, 1120), line)

    # Feedback line from platform response to posture feedback.
    draw.line((2130, 1200, 2130, 1320, 335, 1320, 335, 1200), fill=line, width=4)
    arrowhead(draw, 335, 1320, 335, 1200, line, size=20)

    # Labels on key supervisory arrows.
    draw.text((1380, 430), "分段风险概率、代表风况量", font=note_f, fill=accent)
    draw.text((1010, 915), "调整触发条件与释放程度", font=note_f, fill=accent)

    # Light background bands.
    draw.rounded_rectangle((70, 175, 1950, 405), radius=22, outline="#d7dee7", width=2)
    draw.rounded_rectangle((70, 1005, 2350, 1240), radius=22, outline="#d7dee7", width=2)
    draw.text((95, 168), "预测与风险识别", font=small_f, fill="#4b5563")
    draw.text((95, 998), "闭环控制与压载执行", font=small_f, fill="#4b5563")

    out = OUT_DIR / "fig1_prediction_supervised_ballast_framework.png"
    img.save(out, dpi=(300, 300))
    return out


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def make_svg() -> Path:
    # Matching vector copy for Word insertion. The PNG is the visual reference.
    out = OUT_DIR / "fig1_prediction_supervised_ballast_framework.svg"
    svg = """<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="1450" viewBox="0 0 2400 1450">
<defs>
  <marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth">
    <path d="M2,2 L10,6 L2,10 Z" fill="#3f4d5a"/>
  </marker>
  <marker id="arrowBlue" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth">
    <path d="M2,2 L10,6 L2,10 Z" fill="#2563eb"/>
  </marker>
  <style>
    text { font-family: 'STHeiti', 'PingFang SC', 'Microsoft YaHei', sans-serif; fill: #111827; }
    .title { font-size: 56px; }
    .box { font-size: 38px; }
    .small { font-size: 31px; }
    .note { font-size: 34px; fill: #2563eb; }
    .muted { fill: #4b5563; }
  </style>
</defs>
<text x="1200" y="105" text-anchor="middle" class="title">短时风况预测驱动的主动压载监督调节框架</text>
<rect x="70" y="175" width="1880" height="230" rx="22" fill="none" stroke="#d7dee7" stroke-width="2"/>
<text x="95" y="170" class="small muted">预测与风险识别</text>
<rect x="70" y="1005" width="2280" height="235" rx="22" fill="none" stroke="#d7dee7" stroke-width="2"/>
<text x="95" y="1000" class="small muted">闭环控制与压载执行</text>
"""

    def box(x: int, y: int, w: int, h: int, lines: list[str], fill: str, stroke: str = "#52616f") -> str:
        lines_svg = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="26" fill="{fill}" stroke="{stroke}" stroke-width="4"/>']
        start_y = y + h / 2 - (len(lines) - 1) * 24
        for i, line in enumerate(lines):
            lines_svg.append(f'<text x="{x+w/2}" y="{start_y+i*50}" text-anchor="middle" dominant-baseline="middle" class="box">{esc(line)}</text>')
        return "\n".join(lines_svg)

    svg += box(120, 210, 460, 160, ["历史风况与平台状态", "风速、风向、姿态信息"], "#eaf3fb")
    svg += box(720, 210, 460, 160, ["短时风况预测", "未来60分钟风矢量"], "#eaf3fb")
    svg += box(1320, 210, 580, 160, ["风险识别与压力代理", "0—20 / 20—40 / 40—60分钟"], "#eaf3fb")
    svg += box(760, 525, 880, 285, ["预测监督层"], "#ecf7ee", "#3f7a4c")
    for x, label in [(845, "经济保持"), (1095, "提前调节"), (1345, "安全响应")]:
        svg += f'<rect x="{x}" y="625" width="210" height="68" rx="24" fill="#ffffff" stroke="#3f7a4c" stroke-width="3"/>'
        svg += f'<text x="{x+105}" y="660" text-anchor="middle" dominant-baseline="middle" class="small" fill="#14532d">{label}</text>'
    svg += '<text x="1200" y="758" text-anchor="middle" dominant-baseline="middle" class="small">输出外层调节参数：姿态响应死区、动作预算、安全/经济权重</text>'
    svg += box(120, 1040, 430, 160, ["姿态反馈", "横摇、纵摇偏差"], "#f5f6f7")
    svg += box(690, 1040, 470, 160, ["基础闭环控制器", "生成压载调节需求"], "#f5f6f7")
    svg += box(1300, 1040, 500, 160, ["压载分配与水泵执行", "三舱目标水量与泵送动作"], "#f7f2ea")
    svg += box(1940, 1040, 380, 160, ["浮式平台", "姿态响应更新"], "#f5f6f7")
    svg += """
<line x1="580" y1="290" x2="720" y2="290" stroke="#3f4d5a" stroke-width="5" marker-end="url(#arrow)"/>
<line x1="1180" y1="290" x2="1320" y2="290" stroke="#3f4d5a" stroke-width="5" marker-end="url(#arrow)"/>
<line x1="1610" y1="370" x2="1320" y2="525" stroke="#2563eb" stroke-width="5" stroke-dasharray="16 12" marker-end="url(#arrowBlue)"/>
<line x1="1115" y1="810" x2="945" y2="1040" stroke="#2563eb" stroke-width="5" stroke-dasharray="16 12" marker-end="url(#arrowBlue)"/>
<line x1="550" y1="1120" x2="690" y2="1120" stroke="#3f4d5a" stroke-width="5" marker-end="url(#arrow)"/>
<line x1="1160" y1="1120" x2="1300" y2="1120" stroke="#3f4d5a" stroke-width="5" marker-end="url(#arrow)"/>
<line x1="1800" y1="1120" x2="1940" y2="1120" stroke="#3f4d5a" stroke-width="5" marker-end="url(#arrow)"/>
<path d="M2130,1200 L2130,1320 L335,1320 L335,1200" fill="none" stroke="#3f4d5a" stroke-width="4" marker-end="url(#arrow)"/>
<text x="1380" y="455" class="note">分段风险概率、代表风况量</text>
<text x="1010" y="940" class="note">调整触发条件与释放程度</text>
</svg>
"""
    out.write_text(svg, encoding="utf-8")
    return out


if __name__ == "__main__":
    png = make_png()
    svg = make_svg()
    print(png)
    print(svg)
