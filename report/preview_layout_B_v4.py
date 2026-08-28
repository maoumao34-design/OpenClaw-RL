# -*- coding: utf-8 -*-
"""版式 B v4：收获二去掉「旁支」字样，改为正式标题 + 多色块要点。"""

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os

import sys

FINAL = "--final" in sys.argv    # 正式产出：不带开发水印、不加版式标注文字

OUT = "slide8_final.png" if FINAL else "preview/layout_B_v4.png"
KEEP = "preview/keep/slide8_layoutB_v4.png"
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
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2
    text(d, (cx, y0 + 22), title, font(17, True), title_color, anchor="mt")
    if sub:
        text(d, (cx, y0 + 52), sub, font(13), SOFT, anchor="mt")


def main():
    im = clean_bg()
    d = ImageDraw.Draw(im)
    W, _ = im.size

    text(d, (W / 2, 36), "实习收获", font(34, True), INK, anchor="mt")

    # 顶栏略收一点高度，给底栏色块腾空间
    card(d, (56, 90, 1544, 200))
    text(d, (88, 108), "收获一 ｜ 对 Agent 自进化形成了系统认知",
         font(15, True), ACCENT)
    text(d, (88, 142),
         "自进化是通往 AGI 的必要路径；当前没有完备路线，值得多方案并行探索",
         font(20, True), INK)

    gap, cw, top, bot = 22, 478, 220, 560
    chip_h = 78

    # —— 01 ——
    x0 = 56
    x1 = x0 + cw
    card(d, (x0, top, x1, bot))
    text(d, (x0 + 28, top + 22), "01", font(20, True), ACCENT)
    text(d, (x0 + 72, top + 24), "必要性", font(20, True), INK)

    cy1 = top + 72
    half_w = (cw - 24 * 2 - 12) / 2
    chip(d, (x0 + 24, cy1, x0 + 24 + half_w, cy1 + chip_h),
         CHIP_COOL, OUTLINE_COOL, "上线即冻结", "模型能力固定", (70, 100, 140))
    chip(d, (x0 + 24 + half_w + 12, cy1, x1 - 24, cy1 + chip_h),
         CHIP_COOL, OUTLINE_COOL, "任务持续漂移", "分布不断变化", (70, 100, 140))
    text(d, ((x0 + x1) / 2, cy1 + chip_h + 16), "导  致", font(12, True), FAINT, anchor="mm")
    cy2 = top + 190
    chip(d, (x0 + 24, cy2, x1 - 24, cy2 + chip_h),
         CHIP_WARN, OUTLINE_WARN, "能力衰减是必然", "自进化因此成为必要路径", ACCENT)

    # —— 02 ——
    x0 = 56 + cw + gap
    x1 = x0 + cw
    card(d, (x0, top, x1, bot))
    text(d, (x0 + 28, top + 22), "02", font(20, True), ACCENT)
    text(d, (x0 + 72, top + 24), "技术谱系", font(20, True), INK)

    cy1 = top + 72
    chip(d, (x0 + 24, cy1, x1 - 24, cy1 + chip_h),
         CHIP_L, OUTLINE_L, "非参数化", "记忆 / 技能库 / 零停机", GREEN)
    text(d, ((x0 + x1) / 2, cy1 + chip_h + 16), "互  补", font(12, True), FAINT, anchor="mm")
    cy2 = top + 190
    chip(d, (x0 + 24, cy2, x1 - 24, cy2 + chip_h),
         CHIP_R, OUTLINE_R, "参数化", "权重更新 / 真正长能力", ACCENT)

    # —— 03 ——
    x0 = 56 + 2 * (cw + gap)
    x1 = x0 + cw
    card(d, (x0, top, x1, bot))
    text(d, (x0 + 28, top + 22), "03", font(20, True), ACCENT)
    text(d, (x0 + 72, top + 24), "我的落点", font(20, True), INK)

    cy1 = top + 72
    chip(d, (x0 + 24, cy1, x1 - 24, cy1 + chip_h),
         CHIP_R, OUTLINE_R, "OpenClaw-RL", "复现并验证核心机制", ACCENT)
    text(d, ((x0 + x1) / 2, cy1 + chip_h + 16), "定  位", font(12, True), FAINT, anchor="mm")
    cy2 = top + 190
    chip(d, (x0 + 24, cy2, x1 - 24, cy2 + chip_h),
         CHIP_WARN, OUTLINE_WARN, "参数化一端", "边服务边进化", ACCENT)

    # —— 收获二：正式标题 + 四色块 ——
    card(d, (56, 580, 1544, 830), fill=(255, 255, 255, 220))
    text(d, (88, 600), "收获二 ｜ 试用 Multica，体验了另一条技术路线",
         font(16, True), GREEN)

    # 四个并排色块
    items = [
        (CHIP_L, OUTLINE_L, GREEN, "非参数化对照", "与收获一互为印证"),
        (CHIP_COOL, OUTLINE_COOL, (70, 100, 140), "完整产品开发", "完成可视化产品展示"),
        (CHIP_R, OUTLINE_R, ACCENT, "多智能体协作优势", "任务分工与协同执行"),
        (CHIP_WARN, OUTLINE_WARN, ACCENT, "反哺平台改进", "提出多项改进意见"),
    ]
    n = len(items)
    bar_l, bar_r = 88, 1512
    bar_gap = 16
    bar_w = (bar_r - bar_l - bar_gap * (n - 1)) / n
    by = 650
    bh = 140
    for i, (fill, outline, tc, title, sub) in enumerate(items):
        bx0 = bar_l + i * (bar_w + bar_gap)
        bx1 = bx0 + bar_w
        d.rounded_rectangle([bx0, by, bx1, by + bh], radius=12,
                            fill=fill, outline=outline, width=1)
        cx = (bx0 + bx1) / 2
        text(d, (cx, by + 40), title, font(17, True), tc, anchor="mt")
        text(d, (cx, by + 82), sub, font(13), SOFT, anchor="mt")

    if not FINAL:
        text(d, (W / 2, 858), "版式 B · v4（收获二正式标题 + 色块）", font(13), FAINT, anchor="mt")

    rgb = im.convert("RGB")
    rgb.save(OUT, quality=95)
    if not FINAL:
        os.makedirs(os.path.dirname(KEEP), exist_ok=True)
        rgb.save(KEEP, quality=95)
        print("kept ", KEEP)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
