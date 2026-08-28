# -*- coding: utf-8 -*-
"""第8页三种版式预览（干净底，无串页文字）。"""

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os

OUT_DIR = "preview"
SRC = os.path.join(OUT_DIR, "slide02.png")

F_BOLD = "C:/Windows/Fonts/msyhbd.ttc"
F_REG = "C:/Windows/Fonts/msyh.ttc"

INK = (36, 40, 48)
SOFT = (100, 106, 118)
FAINT = (140, 146, 158)
ACCENT = (176, 108, 52)
GREEN = (55, 110, 95)
CARD_EDGE = (205, 210, 218)


def font(size, bold=False):
    return ImageFont.truetype(F_BOLD if bold else F_REG, size)


def t(d, xy, s, f, fill, anchor="lt"):
    d.text(xy, s, font=f, fill=fill, anchor=anchor)


def clean_bg():
    """从模板页取样氛围，强模糊 + 提亮，去掉原文字。"""
    src = Image.open(SRC).convert("RGB")
    bg = src.filter(ImageFilter.GaussianBlur(radius=40))
    # 再铺一层半透明白，保证可读
    white = Image.new("RGB", bg.size, (250, 248, 245))
    bg = Image.blend(bg, white, 0.35)
    return bg.convert("RGBA")


def card(d, box, radius=14, fill=(255, 255, 255, 210)):
    d.rounded_rectangle(box, radius=radius, fill=fill, outline=CARD_EDGE, width=1)


def title(d):
    t(d, (800, 52), "实习收获", font(36, True), INK, anchor="mt")


def layout_a():
    im = clean_bg()
    d = ImageDraw.Draw(im)
    title(d)

    card(d, (70, 120, 1040, 800))
    t(d, (100, 150), "收获一", font(18, True), ACCENT)
    t(d, (100, 185), "对 Agent 自进化形成了系统认知", font(24, True), INK)

    items = [
        ("结论", "通往 AGI 的必要路径 · 多方案并行探索"),
        ("必要性", "上线冻结 × 任务漂移 → 能力衰减必然"),
        ("技术谱系", "非参数化（记忆/技能）  ↔  参数化（权重）"),
        ("我的落点", "OpenClaw-RL · 边服务边进化"),
    ]
    y = 260
    for h, b in items:
        d.ellipse([100, y + 8, 114, y + 22], fill=ACCENT)
        t(d, (136, y), h, font(20, True), INK)
        t(d, (136, y + 34), b, font(17), SOFT)
        y += 105

    card(d, (1080, 120, 1530, 800), fill=(240, 248, 244, 210))
    t(d, (1110, 150), "收获二", font(18, True), GREEN)
    t(d, (1110, 185), "试用 Multica", font(24, True), INK)
    t(d, (1110, 235), "非参数化路线对照", font(16), SOFT)
    d.line([(1110, 280), (1500, 280)], fill=(190, 205, 195), width=1)
    for j, s in enumerate(["完整产品开发", "印证两条路互补", "反哺平台改进"]):
        t(d, (1110, 330 + j * 80), "·  " + s, font(18), SOFT)

    t(d, (800, 850), "版式 A · 主次双栏（主 65% / 次 35%）", font(14), FAINT, anchor="mt")
    path = os.path.join(OUT_DIR, "layout_A_primary_secondary.png")
    im.convert("RGB").save(path, quality=95)
    print("wrote", path)


def layout_b():
    im = clean_bg()
    d = ImageDraw.Draw(im)
    title(d)

    card(d, (70, 115, 1530, 240))
    t(d, (100, 138), "收获一 · 核心判断", font(16, True), ACCENT)
    t(d, (100, 178), "自进化是通往 AGI 的必要路径；没有完备路线，值得并行探索",
      font(22, True), INK)

    bodies = [
        ("01  必要性", ["上线即冻结", "任务持续漂移", "→ 能力衰减必然"]),
        ("02  技术谱系", ["非参数化 · 记忆/技能", "↔", "参数化 · 权重更新"]),
        ("03  我的落点", ["OpenClaw-RL", "参数化一端", "边服务边进化"]),
    ]
    gap, cw = 24, 470
    for i, (h, lines) in enumerate(bodies):
        x = 70 + i * (cw + gap)
        card(d, (x, 270, x + cw, 620))
        t(d, (x + 28, 300), h, font(20, True), ACCENT)
        by = 360
        for line in lines:
            t(d, (x + 28, by), line, font(18), FAINT if line == "↔" else SOFT)
            by += 40

    card(d, (70, 650, 1530, 800), fill=(240, 248, 244, 210))
    t(d, (100, 680), "收获二 · 旁支", font(16, True), GREEN)
    t(d, (100, 720), "试用 Multica", font(22, True), INK)
    t(d, (360, 725), "完整产品开发  ·  印证两条路互补  ·  反哺平台改进",
      font(17), SOFT)

    t(d, (800, 850), "版式 B · 结论置顶 + 三卡片 + 底栏旁支", font(14), FAINT, anchor="mt")
    path = os.path.join(OUT_DIR, "layout_B_headline_cards.png")
    im.convert("RGB").save(path, quality=95)
    print("wrote", path)


def layout_c():
    im = clean_bg()
    d = ImageDraw.Draw(im)
    title(d)

    t(d, (90, 120), "收获一", font(18, True), ACCENT)
    t(d, (90, 155), "对 Agent 自进化形成了系统认知", font(24, True), INK)

    rows = [
        ("01", "结论", "必要路径 · 多方案并行"),
        ("02", "必要性", "冻结 × 漂移 → 衰减必然"),
        ("03", "技术谱系", "非参数化  ↔  参数化"),
        ("04", "落点", "OpenClaw-RL · 边服务边进化"),
    ]
    y = 230
    for num, h, b in rows:
        d.rounded_rectangle([90, y, 168, y + 68], radius=10,
                            fill=(245, 232, 214, 230), outline=ACCENT, width=1)
        t(d, (129, y + 34), num, font(20, True), ACCENT, anchor="mm")
        t(d, (200, y + 10), h, font(22, True), INK)
        t(d, (200, y + 42), b, font(17), SOFT)
        y += 105

    d.rounded_rectangle([1180, 120, 1530, 800], radius=16,
                        fill=(236, 244, 240, 220), outline=(170, 195, 185), width=1)
    t(d, (1210, 160), "收获二", font(16, True), GREEN)
    t(d, (1210, 200), "Multica", font(28, True), INK)
    t(d, (1210, 255), "非参数化对照", font(16), SOFT)
    d.line([(1210, 300), (1500, 300)], fill=(180, 200, 190), width=1)
    for j, s in enumerate(["完整产品", "印证互补", "反哺平台"]):
        t(d, (1210, 350 + j * 90), s, font(20), INK)

    t(d, (800, 850), "版式 C · 编号精简列表 + 右侧旁支条", font(14), FAINT, anchor="mt")
    path = os.path.join(OUT_DIR, "layout_C_numbered_strip.png")
    im.convert("RGB").save(path, quality=95)
    print("wrote", path)


if __name__ == "__main__":
    layout_a()
    layout_b()
    layout_c()
