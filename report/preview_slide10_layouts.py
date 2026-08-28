# -*- coding: utf-8 -*-
"""第10页「职业规划」三种版式预览（不上 PPT）。

检索归纳的适配备选：
  A  愿景横幅 + 短中长期三阶段轴 + 底栏行动（最经典，结构贴近第8页版式B）
  B  上升阶梯三步（成长隐喻：短→中→长逐级抬高）
  C  左愿景右路径（大题眼占左，右侧竖向时间轴）
"""

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
import sys

FINAL = "--final-b" in sys.argv    # 正式产出 layout_b：不带开发水印

OUT_DIR = "preview"
SRC = os.path.join(OUT_DIR, "slide02.png")

F_BOLD = "C:/Windows/Fonts/msyhbd.ttc"
F_REG = "C:/Windows/Fonts/msyh.ttc"

INK = (36, 40, 48)
SOFT = (88, 94, 106)
MUTED = (120, 126, 138)
FAINT = (150, 156, 168)
ACCENT = (176, 108, 52)
EDGE = (200, 206, 214)
WARM_FILL = (245, 232, 214)
COOL_FILL = (232, 238, 244)


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


def title(d, w):
    text(d, (w / 2, 42), "职业规划", font(34, True), INK, anchor="mt")


# ---------- A ----------
def layout_a():
    im = clean_bg()
    d = ImageDraw.Draw(im)
    W, _ = im.size
    title(d, W)

    # 愿景横幅
    card(d, (56, 100, 1544, 230))
    text(d, (88, 120), "职业目标", font(15, True), ACCENT)
    text(d, (88, 152), "成为能在所处方向上做出行业领先成果的研究员",
         font(20, True), INK)
    text(d, (88, 190), "从跟随前沿，到定义前沿", font(22, True), ACCENT)

    # 三阶段
    stages = [
        ("短期", "从「复现」走向「独立」",
         "判断方向价值\n设计实验路径\n做出经得起推敲的结果"),
        ("中期", "做出行业领先的成果",
         "具体问题上达到 SOTA\n推动真实场景落地"),
        ("长期", "提出原创方法或理论",
         "形成方向影响力\n让工作成为别人的参照"),
    ]
    gap, cw, top, bot = 22, 478, 256, 620
    for i, (label, head, body) in enumerate(stages):
        x0 = 56 + i * (cw + gap)
        card(d, (x0, top, x0 + cw, bot))
        # 阶段标签条
        d.rounded_rectangle([x0 + 24, top + 24, x0 + 100, top + 54],
                            radius=8, fill=WARM_FILL, outline=ACCENT, width=1)
        text(d, (x0 + 62, top + 39), label, font(15, True), ACCENT, anchor="mm")
        text(d, (x0 + 28, top + 78), head, font(18, True), INK)
        y = top + 130
        for line in body.split("\n"):
            text(d, (x0 + 28, y), "·  " + line, font(15), SOFT)
            y += 36
        # 阶段间箭头（除最后一张）
        if i < 2:
            ax = x0 + cw + 4
            ay = (top + bot) / 2
            d.polygon([(ax + 12, ay), (ax, ay - 8), (ax, ay + 8)], fill=FAINT)

    # 底栏行动
    card(d, (56, 650, 1544, 820), fill=COOL_FILL)
    text(d, (88, 672), "行动方案", font(15, True), ACCENT)
    actions = [
        "①  自进化主线高频跟进，快速判断新工作价值",
        "②  把复现能力沉淀为实验基建，缩短论文 → 验证周期",
        "③  成果尽量对外呈现，接受同行检验",
    ]
    for j, a in enumerate(actions):
        text(d, (88, 708 + j * 32), a, font(15), SOFT)

    text(d, (W / 2, 858), "版式 A · 愿景横幅 + 三阶段轴 + 底栏行动",
         font(13), FAINT, anchor="mt")
    path = os.path.join(OUT_DIR, "layout10_A_vision_stages.png")
    im.convert("RGB").save(path, quality=95)
    print("wrote", path)


# ---------- B：上升阶梯 ----------
def layout_b():
    im = clean_bg()
    d = ImageDraw.Draw(im)
    W, _ = im.size
    title(d, W)

    # 题眼
    text(d, (W / 2, 105), "从跟随前沿，到定义前沿", font(24, True), ACCENT, anchor="mt")
    text(d, (W / 2, 145), "成为能做出行业领先成果的研究员", font(16), MUTED, anchor="mt")

    # 三阶：高度递增
    stages = [
        ("短期", "复现 → 独立", "判断方向 · 设计实验 · 经得起推敲", 0),
        ("中期", "行业领先成果", "具体问题 SOTA · 真实场景落地", 1),
        ("长期", "原创方法 / 理论", "形成影响力 · 成为他人参照", 2),
    ]
    base_y = 620
    widths = [420, 420, 420]
    heights = [200, 260, 320]
    gap = 30
    total_w = sum(widths) + 2 * gap
    x = (W - total_w) / 2

    for i, (label, head, body, _) in enumerate(stages):
        h = heights[i]
        y1 = base_y
        y0 = base_y - h
        card(d, (x, y0, x + widths[i], y1),
             fill=(255, 255, 255, 230) if i < 2 else WARM_FILL)
        text(d, (x + 24, y0 + 24), label, font(15, True), ACCENT)
        text(d, (x + 24, y0 + 60), head, font(20, True), INK)
        text(d, (x + 24, y0 + 110), body, font(15), SOFT)
        # 阶梯顶边示意线
        if i < 2:
            d.line([(x + widths[i], y0), (x + widths[i] + gap, base_y - heights[i + 1])],
                   fill=FAINT, width=1)
        x += widths[i] + gap

    # 底行动条
    card(d, (80, 660, 1520, 800), fill=COOL_FILL)
    text(d, (110, 690), "行动方案", font(15, True), ACCENT)
    text(d, (110, 735),
         "主线高频跟进  ·  沉淀实验基建、缩短验证周期  ·  成果对外呈现、接受同行检验",
         font(16), SOFT)

    if FINAL:
        path = "slide10_final.png"
    else:
        text(d, (W / 2, 858), "版式 B · 上升阶梯（短→中→长）", font(13), FAINT, anchor="mt")
        path = os.path.join(OUT_DIR, "layout10_B_ascending_steps.png")
    im.convert("RGB").save(path, quality=95)
    print("wrote", path)


# ---------- C：左愿景右路径 ----------
def layout_c():
    im = clean_bg()
    d = ImageDraw.Draw(im)
    W, _ = im.size
    title(d, W)

    # 左大卡片：题眼
    card(d, (56, 110, 620, 800), fill=WARM_FILL)
    text(d, (90, 160), "职业目标", font(16, True), ACCENT)
    text(d, (90, 230), "从跟随前沿", font(28, True), INK)
    text(d, (90, 280), "到定义前沿", font(28, True), ACCENT)
    d.line([(90, 350), (560, 350)], fill=(220, 180, 130), width=1)
    text(d, (90, 390), "成为能在所处方向上", font(17), SOFT)
    text(d, (90, 430), "做出行业领先成果的", font(17), SOFT)
    text(d, (90, 470), "研究员", font(22, True), INK)

    text(d, (90, 580), "行动", font(14, True), ACCENT)
    for j, a in enumerate(["高频跟进主线", "沉淀实验基建", "对外呈现成果"]):
        text(d, (90, 620 + j * 40), "·  " + a, font(16), SOFT)

    # 右：竖向三阶段
    stages = [
        ("01  短期", "从「复现」走向「独立」",
         "判断方向价值 · 设计实验路径 · 经得起推敲的结果"),
        ("02  中期", "做出行业领先的成果",
         "具体问题达到 SOTA · 推动真实场景落地"),
        ("03  长期", "提出原创方法或理论",
         "形成方向影响力 · 成为别人的参照"),
    ]
    y = 110
    for label, head, body in stages:
        card(d, (660, y, 1544, y + 200))
        text(d, (700, y + 30), label, font(16, True), ACCENT)
        text(d, (700, y + 75), head, font(22, True), INK)
        text(d, (700, y + 125), body, font(16), SOFT)
        y += 230

    text(d, (W / 2, 858), "版式 C · 左愿景右路径", font(13), FAINT, anchor="mt")
    path = os.path.join(OUT_DIR, "layout10_C_vision_path.png")
    im.convert("RGB").save(path, quality=95)
    print("wrote", path)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    if FINAL:
        layout_b()
    else:
        layout_a()
        layout_b()
        layout_c()
