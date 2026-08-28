# -*- coding: utf-8 -*-
"""生成写入 PPT 第10页的成品图 career_plan.png。

相对预览版去掉：页标题「职业规划」（PPT 标题占位符已有）、版式脚注。
保留上升阶梯 + 多色分层色块。
"""

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os

OUT = "career_plan.png"
SRC = "preview/slide02.png"

F_BOLD = "C:/Windows/Fonts/msyhbd.ttc"
F_REG = "C:/Windows/Fonts/msyh.ttc"

INK = (36, 40, 48)
SOFT = (92, 98, 110)
MUTED = (122, 128, 140)
FAINT = (168, 174, 184)
ACCENT = (176, 108, 52)
EDGE = (208, 212, 218)

CHIP_COOL = (230, 236, 244)
CHIP_COOL_L = (170, 185, 205)
CHIP_COOL_INK = (70, 100, 140)

CHIP_L = (232, 242, 238)
CHIP_L_L = (170, 195, 185)
CHIP_L_INK = (45, 105, 90)

CHIP_R = (245, 232, 214)
CHIP_R_L = (218, 172, 118)
CHIP_R_INK = ACCENT

CHIP_WARN = (248, 236, 228)
CHIP_WARN_L = (210, 150, 110)
CHIP_WARN_INK = ACCENT


def font(size, bold=False):
    return ImageFont.truetype(F_BOLD if bold else F_REG, size)


def text(d, xy, s, f, fill, anchor="lt"):
    d.text(xy, s, font=f, fill=fill, anchor=anchor)


def clean_bg(size):
    src = Image.open(SRC).convert("RGB")
    bg = src.resize(size, Image.Resampling.LANCZOS)
    bg = bg.filter(ImageFilter.GaussianBlur(radius=40))
    white = Image.new("RGB", size, (250, 248, 245))
    return Image.blend(bg, white, 0.45).convert("RGBA")


def card(d, box, fill=(255, 255, 255, 230), radius=16):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=EDGE, width=1)


def chip(d, box, fill, outline, title, sub, title_color, title_size=16, sub_size=12):
    d.rounded_rectangle(box, radius=10, fill=fill, outline=outline, width=1)
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2
    if sub:
        text(d, (cx, y0 + 16), title, font(title_size, True), title_color, anchor="mt")
        text(d, (cx, y0 + 42), sub, font(sub_size), SOFT, anchor="mt")
    else:
        text(d, (cx, (y0 + y1) / 2), title, font(title_size, True), title_color, anchor="mm")


def main():
    # 内容区约 12.2in × 5.7in @ 160dpi
    W, H = 1952, 912
    im = clean_bg((W, H))
    # 透明底：不要整幅实底，贴模板渐变
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)

    # 题眼（无「职业规划」页标题）
    text(d, (W / 2, 8), "从跟随前沿，到定义前沿", font(26, True), ACCENT, anchor="mt")
    text(d, (W / 2, 48), "成为能做出行业领先成果的研究员", font(16), MUTED, anchor="mt")

    widths = [560, 560, 560]
    heights = [300, 390, 480]
    gap = 36
    base_y = 620
    total_w = sum(widths) + 2 * gap
    x = (W - total_w) / 2

    stages = [
        {
            "label": "短期",
            "head": "复现  →  独立",
            "fill": (255, 255, 255, 235),
            "chips": [
                (CHIP_COOL, CHIP_COOL_L, CHIP_COOL_INK, "判断方向价值", None),
                (CHIP_L, CHIP_L_L, CHIP_L_INK, "设计实验路径", None),
                (CHIP_WARN, CHIP_WARN_L, CHIP_WARN_INK, "经得起推敲的结果", None),
            ],
        },
        {
            "label": "中期",
            "head": "行业领先成果",
            "fill": (255, 255, 255, 235),
            "chips": [
                (CHIP_COOL, CHIP_COOL_L, CHIP_COOL_INK, "达到 SOTA", "在具体问题上做到领先"),
                (CHIP_R, CHIP_R_L, CHIP_R_INK, "真实场景落地", "推动成果走出实验"),
            ],
        },
        {
            "label": "长期",
            "head": "原创方法 / 理论",
            "fill": (255, 252, 248, 235),
            "chips": [
                (CHIP_L, CHIP_L_L, CHIP_L_INK, "形成方向影响力", "在所处方向上被看见"),
                (CHIP_WARN, CHIP_WARN_L, CHIP_WARN_INK, "成为他人参照", "工作成为别人的参照"),
            ],
        },
    ]

    xs = []
    for i, st in enumerate(stages):
        w = widths[i]
        h = heights[i]
        y1 = base_y
        y0 = base_y - h
        xs.append((x, y0, x + w, y1))
        card(d, (x, y0, x + w, y1), fill=st["fill"])

        tip_fill, tip_line, tip_ink = st["chips"][-1][0], st["chips"][-1][1], st["chips"][-1][2]
        d.rounded_rectangle([x + 28, y0 + 22, x + 110, y0 + 54],
                            radius=8, fill=tip_fill, outline=tip_line, width=1)
        text(d, (x + 69, y0 + 38), st["label"], font(16, True), tip_ink, anchor="mm")
        text(d, (x + 128, y0 + 26), st["head"], font(20, True), INK)

        chips = st["chips"]
        n = len(chips)
        area_top = y0 + 78
        area_bot = y1 - 24
        chip_gap = 12
        chip_h = (area_bot - area_top - chip_gap * (n - 1)) / n
        for j, (cf, cl, ci, title, sub) in enumerate(chips):
            cy0 = area_top + j * (chip_h + chip_gap)
            chip(d, (x + 28, cy0, x + w - 28, cy0 + chip_h),
                 cf, cl, title, sub, ci,
                 title_size=18 if sub else 17, sub_size=13)
        x += w + gap

    for i in range(2):
        x0, y0, x1, _ = xs[i]
        nx0, ny0, _, _ = xs[i + 1]
        d.line([(x1, y0), (nx0, ny0)], fill=FAINT, width=1)

    # 行动方案
    card(d, (36, 650, W - 36, 900), fill=(255, 255, 255, 230))
    text(d, (64, 672), "行动方案", font(17, True), ACCENT)

    actions = [
        (CHIP_L, CHIP_L_L, CHIP_L_INK, "主线高频跟进", "快速判断新工作价值"),
        (CHIP_COOL, CHIP_COOL_L, CHIP_COOL_INK, "沉淀实验基建", "缩短论文到验证周期"),
        (CHIP_WARN, CHIP_WARN_L, CHIP_WARN_INK, "成果对外呈现", "接受同行检验"),
    ]
    n = len(actions)
    bar_l, bar_r = 64, W - 64
    bar_gap = 20
    bar_w = (bar_r - bar_l - bar_gap * (n - 1)) / n
    by, bh = 720, 140
    for i, (cf, cl, ci, title, sub) in enumerate(actions):
        bx0 = bar_l + i * (bar_w + bar_gap)
        d.rounded_rectangle([bx0, by, bx0 + bar_w, by + bh],
                            radius=12, fill=cf, outline=cl, width=1)
        cx = bx0 + bar_w / 2
        text(d, (cx, by + 40), title, font(20, True), ci, anchor="mt")
        text(d, (cx, by + 88), sub, font(15), SOFT, anchor="mt")

    im.save(OUT)
    print(f"wrote {OUT} ({W}x{H})")


if __name__ == "__main__":
    main()
