# -*- coding: utf-8 -*-
"""第10页版式 B v4：保留上升阶梯；色块用不同颜色分层突显重点。"""

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os

OUT = "preview/layout10_B_v4.png"
KEEP = "preview/keep/slide10_layoutB_v4.png"
SRC = "preview/slide02.png"

F_BOLD = "C:/Windows/Fonts/msyhbd.ttc"
F_REG = "C:/Windows/Fonts/msyh.ttc"

INK = (36, 40, 48)
SOFT = (92, 98, 110)
MUTED = (122, 128, 140)
FAINT = (168, 174, 184)
ACCENT = (176, 108, 52)
EDGE = (208, 212, 218)

# 分层色：过程 / 能力 / 成果·重点（与第8页同族）
CHIP_COOL = (230, 236, 244)      # 蓝灰：过程、方法
CHIP_COOL_L = (170, 185, 205)
CHIP_COOL_INK = (70, 100, 140)

CHIP_L = (232, 242, 238)         # 绿：路径、基建
CHIP_L_L = (170, 195, 185)
CHIP_L_INK = (45, 105, 90)

CHIP_R = (245, 232, 214)         # 暖沙：能力定位
CHIP_R_L = (218, 172, 118)
CHIP_R_INK = ACCENT

CHIP_WARN = (248, 236, 228)      # 浅橙：结果 / 最重点
CHIP_WARN_L = (210, 150, 110)
CHIP_WARN_INK = ACCENT


def font(size, bold=False):
    return ImageFont.truetype(F_BOLD if bold else F_REG, size)


def text(d, xy, s, f, fill, anchor="lt"):
    d.text(xy, s, font=f, fill=fill, anchor=anchor)


def clean_bg():
    src = Image.open(SRC).convert("RGB")
    bg = src.filter(ImageFilter.GaussianBlur(radius=40))
    white = Image.new("RGB", bg.size, (250, 248, 245))
    return Image.blend(bg, white, 0.35).convert("RGBA")


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
    im = clean_bg()
    d = ImageDraw.Draw(im)
    W, _ = im.size

    text(d, (W / 2, 36), "职业规划", font(34, True), INK, anchor="mt")
    text(d, (W / 2, 92), "从跟随前沿，到定义前沿", font(24, True), ACCENT, anchor="mt")
    text(d, (W / 2, 132), "成为能做出行业领先成果的研究员", font(16), MUTED, anchor="mt")

    # 上升阶梯：同底边、从矮到高
    widths = [430, 430, 430]
    heights = [228, 300, 372]
    gap = 28
    base_y = 598
    total_w = sum(widths) + 2 * gap
    x = (W - total_w) / 2

    # 每张卡内色块颜色不同：前段过程色 → 末段结果色（暖橙）突显落点
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
            # 蓝灰 → 暖沙：跟短期/长期都拉开
            "chips": [
                (CHIP_COOL, CHIP_COOL_L, CHIP_COOL_INK, "达到 SOTA", "在具体问题上做到领先"),
                (CHIP_R, CHIP_R_L, CHIP_R_INK, "真实场景落地", "推动成果走出实验"),
            ],
        },
        {
            "label": "长期",
            "head": "原创方法 / 理论",
            "fill": (255, 252, 248, 235),
            # 绿 → 浅橙：与中期蓝/沙明显不同
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

        # 标签用该卡「落点色」勾一下层级
        tip_fill, tip_line, tip_ink = st["chips"][-1][0], st["chips"][-1][1], st["chips"][-1][2]
        d.rounded_rectangle([x + 22, y0 + 18, x + 88, y0 + 46],
                            radius=8, fill=tip_fill, outline=tip_line, width=1)
        text(d, (x + 55, y0 + 32), st["label"], font(14, True), tip_ink, anchor="mm")
        text(d, (x + 102, y0 + 22), st["head"], font(18, True), INK)

        chips = st["chips"]
        n = len(chips)
        area_top = y0 + 66
        area_bot = y1 - 20
        chip_gap = 10
        chip_h = (area_bot - area_top - chip_gap * (n - 1)) / n
        for j, (cf, cl, ci, title, sub) in enumerate(chips):
            cy0 = area_top + j * (chip_h + chip_gap)
            chip(d, (x + 22, cy0, x + w - 22, cy0 + chip_h),
                 cf, cl, title, sub, ci,
                 title_size=16 if sub else 15, sub_size=12)
        x += w + gap

    for i in range(2):
        x0, y0, x1, _ = xs[i]
        nx0, ny0, _, _ = xs[i + 1]
        d.line([(x1, y0), (nx0, ny0)], fill=FAINT, width=1)

    # 行动方案：三块也分层——跟进(绿) / 基建(蓝) / 呈现(暖，对外检验是落点)
    card(d, (70, 628, 1530, 818), fill=(255, 255, 255, 230))
    text(d, (100, 648), "行动方案", font(15, True), ACCENT)

    actions = [
        (CHIP_L, CHIP_L_L, CHIP_L_INK, "主线高频跟进", "快速判断新工作价值"),
        (CHIP_COOL, CHIP_COOL_L, CHIP_COOL_INK, "沉淀实验基建", "缩短论文到验证周期"),
        (CHIP_WARN, CHIP_WARN_L, CHIP_WARN_INK, "成果对外呈现", "接受同行检验"),
    ]
    n = len(actions)
    bar_l, bar_r = 100, 1500
    bar_gap = 16
    bar_w = (bar_r - bar_l - bar_gap * (n - 1)) / n
    by, bh = 688, 100
    for i, (cf, cl, ci, title, sub) in enumerate(actions):
        bx0 = bar_l + i * (bar_w + bar_gap)
        d.rounded_rectangle([bx0, by, bx0 + bar_w, by + bh],
                            radius=12, fill=cf, outline=cl, width=1)
        cx = bx0 + bar_w / 2
        text(d, (cx, by + 28), title, font(17, True), ci, anchor="mt")
        text(d, (cx, by + 62), sub, font(13), SOFT, anchor="mt")

    text(d, (W / 2, 858), "版式 B · v4（阶梯 + 多色分层突显）", font(13), FAINT, anchor="mt")

    rgb = im.convert("RGB")
    rgb.save(OUT, quality=95)
    os.makedirs(os.path.dirname(KEEP), exist_ok=True)
    rgb.save(KEEP, quality=95)
    print("wrote", OUT)
    print("kept ", KEEP)


if __name__ == "__main__":
    main()
