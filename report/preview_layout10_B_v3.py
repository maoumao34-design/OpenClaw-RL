# -*- coding: utf-8 -*-
"""第10页版式 B v3：保留初版上升阶梯，仅在内部加色块；配色收束。"""

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os

OUT = "preview/layout10_B_v3.png"
KEEP = "preview/keep/slide10_layoutB_v3.png"
SRC = "preview/slide02.png"

F_BOLD = "C:/Windows/Fonts/msyhbd.ttc"
F_REG = "C:/Windows/Fonts/msyh.ttc"

INK = (36, 40, 48)
SOFT = (92, 98, 110)
MUTED = (122, 128, 140)
FAINT = (168, 174, 184)
ACCENT = (176, 108, 52)
EDGE = (208, 212, 218)

# 三阶段各用一族色，避免一页里绿蓝橙乱跳
C_SHORT = (236, 241, 246)       # 冷灰蓝：起步
C_SHORT_LINE = (168, 186, 204)
C_SHORT_INK = (70, 98, 128)

C_MID = (244, 236, 226)         # 暖沙：出成果
C_MID_LINE = (214, 184, 150)
C_MID_INK = ACCENT

C_LONG = (247, 232, 214)        # 更饱和暖色：长期
C_LONG_LINE = (210, 160, 110)
C_LONG_INK = (156, 88, 36)

C_ACT = (238, 242, 246)
C_ACT_LINE = (176, 190, 204)
C_ACT_INK = (70, 98, 128)


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

    # —— 页眉：与初版一致 ——
    text(d, (W / 2, 36), "职业规划", font(34, True), INK, anchor="mt")
    text(d, (W / 2, 92), "从跟随前沿，到定义前沿", font(24, True), ACCENT, anchor="mt")
    text(d, (W / 2, 132), "成为能做出行业领先成果的研究员", font(16), MUTED, anchor="mt")

    # —— 上升阶梯（同底边、从矮到高）——
    # 短 / 中 / 长 高度递增；底边对齐，视觉上向右抬升
    widths = [430, 430, 430]
    heights = [228, 300, 372]
    gap = 28
    base_y = 598
    total_w = sum(widths) + 2 * gap
    x = (W - total_w) / 2

    stages = [
        {
            "label": "短期",
            "head": "复现  →  独立",
            "fill": (255, 255, 255, 235),
            "chip_fill": C_SHORT,
            "chip_line": C_SHORT_LINE,
            "chip_ink": C_SHORT_INK,
            "chips": [
                ("判断方向价值", None),
                ("设计实验路径", None),
                ("经得起推敲的结果", None),
            ],
        },
        {
            "label": "中期",
            "head": "行业领先成果",
            "fill": (255, 255, 255, 235),
            "chip_fill": C_MID,
            "chip_line": C_MID_LINE,
            "chip_ink": C_MID_INK,
            "chips": [
                ("达到 SOTA", "在具体问题上做到领先"),
                ("真实场景落地", "推动成果走出实验"),
            ],
        },
        {
            "label": "长期",
            "head": "原创方法 / 理论",
            "fill": (247, 236, 220, 230),
            "chip_fill": C_LONG,
            "chip_line": C_LONG_LINE,
            "chip_ink": C_LONG_INK,
            "chips": [
                ("形成方向影响力", "在所处方向上被看见"),
                ("成为他人参照", "工作成为别人的参照"),
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

        # 阶段标签
        d.rounded_rectangle([x + 22, y0 + 18, x + 88, y0 + 46],
                            radius=8, fill=st["chip_fill"], outline=st["chip_line"], width=1)
        text(d, (x + 55, y0 + 32), st["label"], font(14, True), st["chip_ink"], anchor="mm")
        text(d, (x + 102, y0 + 22), st["head"], font(18, True), INK)

        chips = st["chips"]
        n = len(chips)
        area_top = y0 + 66
        area_bot = y1 - 20
        chip_gap = 10
        chip_h = (area_bot - area_top - chip_gap * (n - 1)) / n
        for j, (title, sub) in enumerate(chips):
            cy0 = area_top + j * (chip_h + chip_gap)
            chip(d, (x + 22, cy0, x + w - 22, cy0 + chip_h),
                 st["chip_fill"], st["chip_line"], title, sub, st["chip_ink"],
                 title_size=16 if sub else 15, sub_size=12)
        x += w + gap

    # 阶梯连接细线（顶边到下一级顶边）
    for i in range(2):
        x0, y0, x1, _ = xs[i]
        nx0, ny0, _, _ = xs[i + 1]
        d.line([(x1, y0), (nx0, ny0)], fill=FAINT, width=1)

    # —— 行动方案：三块同色系，不另开一套配色 ——
    card(d, (70, 628, 1530, 818), fill=(255, 255, 255, 230))
    text(d, (100, 648), "行动方案", font(15, True), ACCENT)

    actions = [
        ("主线高频跟进", "快速判断新工作价值"),
        ("沉淀实验基建", "缩短论文到验证周期"),
        ("成果对外呈现", "接受同行检验"),
    ]
    n = len(actions)
    bar_l, bar_r = 100, 1500
    bar_gap = 16
    bar_w = (bar_r - bar_l - bar_gap * (n - 1)) / n
    by, bh = 688, 100
    for i, (title, sub) in enumerate(actions):
        bx0 = bar_l + i * (bar_w + bar_gap)
        d.rounded_rectangle([bx0, by, bx0 + bar_w, by + bh],
                            radius=12, fill=C_ACT, outline=C_ACT_LINE, width=1)
        cx = bx0 + bar_w / 2
        text(d, (cx, by + 28), title, font(17, True), C_ACT_INK, anchor="mt")
        text(d, (cx, by + 62), sub, font(13), SOFT, anchor="mt")

    text(d, (W / 2, 858), "版式 B · v3（阶梯保留 + 内部色块）", font(13), FAINT, anchor="mt")

    rgb = im.convert("RGB")
    rgb.save(OUT, quality=95)
    os.makedirs(os.path.dirname(KEEP), exist_ok=True)
    rgb.save(KEEP, quality=95)
    print("wrote", OUT)
    print("kept ", KEEP)


if __name__ == "__main__":
    main()
