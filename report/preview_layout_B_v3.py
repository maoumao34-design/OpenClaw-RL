# -*- coding: utf-8 -*-
"""版式 B v3：01/03 也用色块框突出重点（对齐 02 的双芯片风格）。"""

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os

OUT = "preview/layout_B_v3.png"
KEEP = "preview/keep/slide8_layoutB_v3.png"
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
CHIP_L = (232, 242, 238)       # 绿：非参数化
CHIP_R = (245, 232, 214)       # 暖：参数化 / 落点
CHIP_COOL = (230, 236, 244)    # 蓝灰：问题/现状
CHIP_WARN = (248, 236, 228)    # 浅橙：结果强调
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

    text(d, (W / 2, 42), "实习收获", font(34, True), INK, anchor="mt")

    # 顶栏
    card(d, (56, 100, 1544, 228))
    text(d, (88, 118), "收获一 ｜ 对 Agent 自进化形成了系统认知",
         font(15, True), ACCENT)
    text(d, (88, 152), "核心判断", font(14), MUTED)
    text(d, (88, 178),
         "自进化是通往 AGI 的必要路径；当前没有完备路线，值得多方案并行探索",
         font(21, True), INK)

    gap, cw, top, bot = 22, 478, 252, 640
    chip_h = 88

    # —— 01 必要性 ——
    x0 = 56
    x1 = x0 + cw
    card(d, (x0, top, x1, bot))
    text(d, (x0 + 28, top + 28), "01", font(22, True), ACCENT)
    text(d, (x0 + 78, top + 30), "必要性", font(22, True), INK)

    cy1 = top + 90
    # 上：两个并排「现状」小芯片
    half_w = (cw - 24 * 2 - 12) / 2
    chip(d, (x0 + 24, cy1, x0 + 24 + half_w, cy1 + chip_h),
         CHIP_COOL, OUTLINE_COOL, "上线即冻结", "模型能力固定", (70, 100, 140))
    chip(d, (x0 + 24 + half_w + 12, cy1, x1 - 24, cy1 + chip_h),
         CHIP_COOL, OUTLINE_COOL, "任务持续漂移", "分布不断变化", (70, 100, 140))

    text(d, ((x0 + x1) / 2, cy1 + chip_h + 18), "导  致", font(13, True), FAINT, anchor="mm")

    cy2 = top + 220
    chip(d, (x0 + 24, cy2, x1 - 24, cy2 + chip_h),
         CHIP_WARN, OUTLINE_WARN, "能力衰减是必然", "自进化因此成为必要路径", ACCENT)

    # —— 02 技术谱系 ——
    x0 = 56 + cw + gap
    x1 = x0 + cw
    card(d, (x0, top, x1, bot))
    text(d, (x0 + 28, top + 28), "02", font(22, True), ACCENT)
    text(d, (x0 + 78, top + 30), "技术谱系", font(22, True), INK)

    cy1 = top + 90
    chip(d, (x0 + 24, cy1, x1 - 24, cy1 + chip_h),
         CHIP_L, OUTLINE_L, "非参数化", "记忆 / 技能库 · 零停机", GREEN)
    text(d, ((x0 + x1) / 2, cy1 + chip_h + 18), "互  补", font(13, True), FAINT, anchor="mm")
    cy2 = top + 220
    chip(d, (x0 + 24, cy2, x1 - 24, cy2 + chip_h),
         CHIP_R, OUTLINE_R, "参数化", "权重更新 · 真正长能力", ACCENT)

    # —— 03 我的落点 ——
    x0 = 56 + 2 * (cw + gap)
    x1 = x0 + cw
    card(d, (x0, top, x1, bot))
    text(d, (x0 + 28, top + 28), "03", font(22, True), ACCENT)
    text(d, (x0 + 78, top + 30), "我的落点", font(22, True), INK)

    cy1 = top + 90
    chip(d, (x0 + 24, cy1, x1 - 24, cy1 + chip_h),
         CHIP_R, OUTLINE_R, "OpenClaw-RL", "复现并验证核心机制", ACCENT)
    text(d, ((x0 + x1) / 2, cy1 + chip_h + 18), "定  位", font(13, True), FAINT, anchor="mm")
    cy2 = top + 220
    chip(d, (x0 + 24, cy2, x1 - 24, cy2 + chip_h),
         CHIP_WARN, OUTLINE_WARN, "参数化一端", "边服务边进化", ACCENT)

    # 底栏
    card(d, (56, 668, 1544, 820), fill=(236, 244, 240, 220))
    text(d, (88, 690), "收获二 · 旁支", font(14, True), GREEN)
    text(d, (88, 722), "试用 Multica", font(22, True), INK)
    text(d, (88, 762),
         "自研多智能体平台  ·  完整产品开发  ·  印证两条路互补  ·  反哺平台改进",
         font(16), SOFT)
    text(d, (1480, 744), "非参数化对照", font(15, True), GREEN, anchor="rm")

    text(d, (W / 2, 858), "版式 B · v3（01/03 也用色块强调）", font(13), FAINT, anchor="mt")

    rgb = im.convert("RGB")
    rgb.save(OUT, quality=95)
    os.makedirs(os.path.dirname(KEEP), exist_ok=True)
    rgb.save(KEEP, quality=95)
    print("wrote", OUT)
    print("kept ", KEEP)


if __name__ == "__main__":
    main()
