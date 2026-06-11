from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "paper_figures_current" / "chapter1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FONT_PATH = Path("/System/Library/Fonts/STHeiti Medium.ttc")


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size=size)


def center_text(draw: ImageDraw.ImageDraw, box, text: str, fnt, fill="#111111", spacing=8):
    x, y, w, h = box
    lines = text.split("\n")
    dims = [draw.textbbox((0, 0), line, font=fnt) for line in lines]
    heights = [d[3] - d[1] for d in dims]
    widths = [d[2] - d[0] for d in dims]
    total_h = sum(heights) + spacing * (len(lines) - 1)
    yy = y + (h - total_h) / 2
    for line, tw, th in zip(lines, widths, heights):
        draw.text((x + (w - tw) / 2, yy), line, font=fnt, fill=fill)
        yy += th + spacing


def box(draw, xywh, text, fnt, fill="#f8fbfc", outline="#111111", width=3):
    x, y, w, h = xywh
    draw.rectangle((x, y, x + w, y + h), fill=fill, outline=outline, width=width)
    center_text(draw, xywh, text, fnt)


def arrow(draw, start, end, width=3, fill="#111111"):
    draw.line((*start, *end), fill=fill, width=width)
    x1, y1 = start
    x2, y2 = end
    # simple triangular arrowhead
    if abs(x2 - x1) >= abs(y2 - y1):
        s = 13 if x2 >= x1 else -13
        pts = [(x2, y2), (x2 - s, y2 - 7), (x2 - s, y2 + 7)]
    else:
        s = 13 if y2 >= y1 else -13
        pts = [(x2, y2), (x2 - 7, y2 - s), (x2 + 7, y2 - s)]
    draw.polygon(pts, fill=fill)


def dashed_arrow(draw, start, end, width=3, fill="#333333", dash=14, gap=10):
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    length = (dx * dx + dy * dy) ** 0.5
    ux, uy = dx / length, dy / length
    t = 0
    while t < length - 18:
        a = t
        b = min(t + dash, length - 18)
        draw.line((x1 + ux * a, y1 + uy * a, x1 + ux * b, y1 + uy * b), fill=fill, width=width)
        t += dash + gap
    arrow(draw, (int(x1 + ux * (length - 18)), int(y1 + uy * (length - 18))), end, width=width, fill=fill)


def make():
    img = Image.new("RGB", (1800, 1160), "white")
    draw = ImageDraw.Draw(img)

    title_f = font(44)
    module_f = font(35)
    box_f = font(31)
    label_f = font(25)

    title = "短时风况预测驱动的主动压载监督调节框架"
    tw = draw.textbbox((0, 0), title, font=title_f)[2]
    draw.text(((1800 - tw) / 2, 32), title, font=title_f, fill="#111111")

    blue = "#d8f0f4"
    pink = "#fde8e1"
    green = "#d9efc5"
    cream = "#fff6ce"
    white = "#f9fbfc"

    left = (110, 210, 900, 800)
    right = (1080, 210, 620, 800)
    top = (700, 110, 410, 80)

    draw.rectangle((left[0], left[1], left[0] + left[2], left[1] + left[3]), fill=blue)
    draw.rectangle((right[0], right[1], right[0] + right[2], right[1] + right[3]), fill=pink)
    box(draw, top, "姿态与压载状态", module_f, fill=green, width=3)

    # Left panel boxes
    l_in1 = (155, 285, 305, 90)
    l_in2 = (155, 495, 305, 90)
    l_in3 = (155, 755, 305, 90)
    l_p1 = (580, 285, 340, 90)
    l_p2 = (580, 445, 340, 90)
    l_p3 = (580, 605, 340, 90)
    l_p4 = (580, 765, 340, 90)

    box(draw, l_in1, "历史风况数据", box_f)
    box(draw, l_in2, "当前平台状态", box_f)
    box(draw, l_in3, "压力代理信号", box_f, fill=cream)
    box(draw, l_p1, "短时风况预测", box_f)
    box(draw, l_p2, "风险窗口识别", box_f)
    box(draw, l_p3, "监督状态判别", box_f)
    box(draw, l_p4, "外层参数调度", box_f)

    # Right panel boxes
    r1 = (1190, 285, 400, 90)
    r2 = (1190, 435, 400, 90)
    r3 = (1190, 585, 400, 90)
    r4 = (1190, 735, 400, 90)
    r5 = (1190, 885, 400, 90)

    box(draw, r1, "姿态误差计算", box_f)
    box(draw, r2, "基础反馈控制", box_f)
    box(draw, r3, "压载需求修正", box_f)
    box(draw, r4, "压载分配执行", box_f)
    box(draw, r5, "平台响应更新", box_f)

    # Left panel internal arrows
    arrow(draw, (460, 330), (580, 330))
    arrow(draw, (750, 375), (750, 445))
    arrow(draw, (750, 535), (750, 605))
    arrow(draw, (750, 695), (750, 765))

    # Inputs feeding decision chain
    draw.line((460, 540, 520, 540, 520, 650, 580, 650), fill="#111111", width=3)
    arrow(draw, (520, 650), (580, 650))
    draw.line((460, 800, 520, 800, 520, 810, 580, 810), fill="#111111", width=3)
    arrow(draw, (520, 810), (580, 810))

    # Right panel vertical arrows
    arrow(draw, (1390, 375), (1390, 435))
    arrow(draw, (1390, 525), (1390, 585))
    arrow(draw, (1390, 675), (1390, 735))
    arrow(draw, (1390, 825), (1390, 885))

    # Cross link from supervision to demand correction.
    dashed_arrow(draw, (920, 810), (1190, 630), width=3, fill="#333333")
    draw.text((980, 690), "死区、预算、权重", font=label_f, fill="#111111")

    # Feedback from platform response to top state box.
    draw.line((1590, 930, 1660, 930, 1660, 150, 1110, 150), fill="#111111", width=3)
    arrow(draw, (1660, 150), (1110, 150))

    # Top state feeds current platform state and posture error.
    # Route around the historical wind input so the state feedback is not
    # mistaken as wind-data input.
    draw.line((700, 150, 95, 150, 95, 540, 155, 540), fill="#111111", width=3)
    arrow(draw, (95, 540), (155, 540))
    draw.line((1110, 150, 1390, 150, 1390, 285), fill="#111111", width=3)
    arrow(draw, (1390, 150), (1390, 285))

    # Panel captions.
    center_text(draw, (left[0], 1035, left[2], 60), "预测监督层", module_f)
    center_text(draw, (right[0], 1035, right[2], 60), "闭环压载层", module_f)

    out = OUT_DIR / "fig1_supervised_ballast_flow_v3.png"
    img.save(out, dpi=(300, 300))
    print(out)


if __name__ == "__main__":
    make()
