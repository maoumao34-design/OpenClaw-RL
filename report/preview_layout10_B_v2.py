# -*- coding: utf-8 -*-
"""第10页版式 B v2：上升阶梯 + 色块突出（对齐第8页风格）。"""

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os

OUT = "preview/layout10_B_v2.png"
KEEP = "preview/keep/slide10_layoutB_v2.png"
SRC = "preview/slide02.png"

F_BOLD = "C:/Windows/Fonts/msyhbd.ttc"
F_REG = "C:/Windows/Fonts/msyh.ttc"

INK = (36, 40, 48)
SOFT = (88, 94, 106)
MUTED = (120, 126, 138)
FAINT = (150, 156, 168)
ACCENT = (176, 108, 52)
GREEN = (45, 105, 90)
EDGE = (200, 206, 214)

CHIP_L = (232, 242, 238)
CHIP_R = (245, 232, 214)
CHIP_COOL = (230, 236, 244)
CHIP_WARN = (248, 236, 228)
OUTLINE_L = (170, 195, 185)
OUTLINE_R = (218, 172, 118)
OUTLINE_COOL = (170, 185, 205)
OUTLINE_WARN = (210, 150, 110)


def font(size, bold=False):
    return ImageFont.truetype(F_BOLD if bold else F_REG, size)


def text(d, xy, s, f, fill, anchor="lt"):
    d.text(xy, s, font=f, fill=fill, anchor=anchor)


def clean_bg():
    src = Image.open(SRC).convert("RGB")
    bg = src.filter(ImageFilter.GaussianBlur(radius=40))
    white = Image.new("RGB", bg.size, (250, 248, 245))
    return Image.blend(bg, white, 0.35).convert("RGBA")


def card(d, box, fill=(255, 255, 255, 220), radius=14):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=EDGE, width=1)


def chip(d, box, fill, outline, title, sub, title_color):
    d.rounded_rectangle(box, radius=10, fill=fill, outline=outline, width=1)
    x0, y0, x1, _ = box
    cx = (x0 + x1) / 2
    text(d, (cx, y0 + 18), title, font(15, True), title_color, anchor="mt")
    if sub:
        text(d, (cx, y0 + 46), sub, font(12), SOFT, anchor="mt")


def main():
    im = clean_bg()
    d = ImageDraw.Draw(im)
    W, _ = im.size

    text(d, (W / 2, 36), "职业规划", font(34, True), INK, anchor="mt")
    text(d, (W / 2, 88), "从跟随前沿，到定义前沿", font(22, True), ACCENT, anchor="mt")
    text(d, (W / 2, 124), "成为能做出行业领先成果的研究员", font(15), MUTED, anchor="mt")

    # 三阶段卡片（等高，更利于塞色块；用编号体现递进）
    stages = [
        {
            "label": "短期",
            "head": "从「复现」走向「独立」",
            "chips": [
                (CHIP_COOL, OUTLINE_COOL, (70, 100, 140), "判断方向价值", "自己判断值不值得做"),
                (CHIP_COOL, OUTLINE_COOL, (70, 100, 140), "设计实验路径", "能独立设计验证路径"),
                (CHIP_WARN, OUTLINE_WARN, ACCENT, "经得起推敲", "做出可站得住的结果"),
            ],
        },
        {
            "label": "中期",
            "head": "做出行业领先的成果",
            "chips": [
                (CHIP_R, OUTLINE_R, ACCENT, "达到 SOTA", "在具体问题上做到领先"),
                (CHIP_WARN, OUTLINE_WARN, ACCENT, "真实场景落地", "不止停在实验里"),
            ],
        },
        {
            "label": "长期",
            "head": "提出原创方法或理论",
            "chips": [
                (CHIP_R, OUTLINE_R, ACCENT, "形成影响力", "在所处方向上被看见"),
                (CHIP_WARN, OUTLINE_WARN, ACCENT, "成为他人参照", "工作成为别人的参照"),
            ],
        },
    ]

    gap, cw = 22, 478
    top, bot = 165, 560
    for i, st in enumerate(stages):
        x0 = 56 + i * (cw + gap)
        x1 = x0 + cw
        fill = (255, 255, 255, 220) if i < 2 else (245, 232, 214, 200)
        card(d, (x0, top, x1, bot), fill=fill)

        # 标签
        d.rounded_rectangle([x0 + 22, top + 18, x0 + 92, top + 46],
                            radius=8, fill=CHIP_R, outline=ACCENT, width=1)
        text(d, (x0 + 57, top + 32), st["label"], font(14, True), ACCENT, anchor="mm")
        text(d, (x0 + 108, top + 24), st["head"], font(16, True), INK)

        chips = st["chips"]
        n = len(chips)
        chip_gap = 12
        area_top = top + 70
        area_bot = bot - 22
        chip_h = (area_bot - area_top - chip_gap * (n - 1)) / n
        for j, (cf, co, tc, title, sub) in enumerate(chips):
            cy0 = area_top + j * (chip_h + chip_gap)
            chip(d, (x0 + 22, cy0, x1 - 22, cy0 + chip_h), cf, co, title, sub, tc)

        # 阶段箭头
        if i < 2:
            ax = x1 + 4
            ay = (top + bot) / 2
            d.polygon([(ax + 12, ay), (ax, ay - 7), (ax, ay + 7)], fill=FAINT)

    # 行动方案：四色块
    card(d, (56, 585, 1544, 830))
    text(d, (88, 605), "行动方案", font(16, True), ACCENT)

    # 行动方案：三色块（对齐定稿三条）
    actions = [
        (CHIP_L, OUTLINE_L, GREEN, "主线高频跟进", "快速判断新工作价值"),
        (CHIP_COOL, OUTLINE_COOL, (70, 100, 140), "沉淀实验基建", "缩短论文到验证周期"),
        (CHIP_R, OUTLINE_R, ACCENT, "成果对外呈现", "接受同行检验"),
    ]

    n = len(actions)
    bar_l, bar_r = 88, 1512
    bar_gap = 18
    bar_w = (bar_r - bar_l - bar_gap * (n - 1)) / n
    by, bh = 650, 140
    for i, (cf, co, tc, title, sub) in enumerate(actions):
        bx0 = bar_l + i * (bar_w + bar_gap)
        bx1 = bx0 + bar_w
        d.rounded_rectangle([bx0, by, bx1, by + bh], radius=12,
                            fill=cf, outline=co, width=1)
        cx = (bx0 + bx1) / 2
        text(d, (cx, by + 42), title, font(18, True), tc, anchor="mt")
        text(d, (cx, by + 86), sub, font(14), SOFT, anchor="mt")

    text(d, (W / 2, 858), "版式 B · v2（色块突出）", font(13), FAINT, anchor="mt")

    rgb = im.convert("RGB")
    rgb.save(OUT, quality=95)
    os.makedirs(os.path.dirname(KEEP), exist_ok=True)
    rgb.save(KEEP, quality=95)
    print("wrote", OUT)
    print("kept ", KEEP)


if __name__ == "__main__":
    main()
