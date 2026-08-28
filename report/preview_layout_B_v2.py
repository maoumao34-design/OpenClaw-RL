# -*- coding: utf-8 -*-
"""版式 B 内容/字号 refinement 预览（不上 PPT）。"""

from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os

OUT = "preview/layout_B_v2.png"
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


def main():
    im = clean_bg()
    d = ImageDraw.Draw(im)
    W, H = im.size

    # 标题
    text(d, (W / 2, 42), "实习收获", font(34, True), INK, anchor="mt")

    # —— 顶栏：结论 ——
    card(d, (56, 100, 1544, 228))
    text(d, (88, 118), "收获一 ｜ 对 Agent 自进化形成了系统认知",
         font(15, True), ACCENT)
    text(d, (88, 152), "核心判断", font(14), MUTED)
    text(d, (88, 178),
         "自进化是通往 AGI 的必要路径；当前没有完备路线，值得多方案并行探索",
         font(21, True), INK)

    # —— 三卡片 ——
    # 内容比 v1 稍充实，字号分层更清楚，减少空荡感
    specs = [
        {
            "num": "01",
            "title": "必要性",
            "lines": [
                (True, "模型上线即冻结"),
                (False, "用户任务分布持续漂移"),
                (True, "→  能力衰减是必然"),
            ],
        },
        {
            "num": "02",
            "title": "技术谱系",
            "dual": True,  # 特殊：双芯片
        },
        {
            "num": "03",
            "title": "我的落点",
            "lines": [
                (True, "OpenClaw-RL"),
                (False, "复现并验证核心机制"),
                (True, "参数化 · 边服务边进化"),
            ],
        },
    ]

    gap, cw, top, bot = 22, 478, 252, 640
    for i, spec in enumerate(specs):
        x0 = 56 + i * (cw + gap)
        x1 = x0 + cw
        card(d, (x0, top, x1, bot))

        # 编号 + 标题同一行
        text(d, (x0 + 28, top + 28), spec["num"], font(22, True), ACCENT)
        text(d, (x0 + 78, top + 30), spec["title"], font(22, True), INK)

        if spec.get("dual"):
            # 上下两个轻量芯片，中间写「互补」
            chip_h = 88
            cy1 = top + 90
            cy2 = top + 220
            # 上：非参数化
            d.rounded_rectangle([x0 + 24, cy1, x1 - 24, cy1 + chip_h],
                                radius=10, fill=CHIP_L, outline=(170, 195, 185), width=1)
            text(d, ((x0 + x1) / 2, cy1 + 28), "非参数化", font(18, True), GREEN, anchor="mt")
            text(d, ((x0 + x1) / 2, cy1 + 58), "记忆 / 技能库 · 零停机", font(14), SOFT, anchor="mt")

            text(d, ((x0 + x1) / 2, cy1 + chip_h + 18), "互  补", font(13, True), FAINT, anchor="mm")

            # 下：参数化
            d.rounded_rectangle([x0 + 24, cy2, x1 - 24, cy2 + chip_h],
                                radius=10, fill=CHIP_R, outline=(218, 172, 118), width=1)
            text(d, ((x0 + x1) / 2, cy2 + 28), "参数化", font(18, True), ACCENT, anchor="mt")
            text(d, ((x0 + x1) / 2, cy2 + 58), "权重更新 · 真正长能力", font(14), SOFT, anchor="mt")
        else:
            y = top + 100
            for bold, line in spec["lines"]:
                text(d, (x0 + 28, y), line,
                     font(18, True) if bold else font(16),
                     INK if bold else SOFT)
                y += 48 if bold else 40

    # —— 底栏：收获二 ——
    card(d, (56, 668, 1544, 820), fill=(236, 244, 240, 220))
    text(d, (88, 690), "收获二 · 旁支", font(14, True), GREEN)
    text(d, (88, 722), "试用 Multica", font(22, True), INK)
    text(d, (88, 762),
         "自研多智能体平台  ·  完整产品开发  ·  印证两条路互补  ·  反哺平台改进",
         font(16), SOFT)
    # 右侧一句定位
    text(d, (1480, 744), "非参数化对照", font(15, True), GREEN, anchor="rm")

    text(d, (W / 2, 858), "版式 B · v2（内容与字号调整）", font(13), FAINT, anchor="mt")

    im.convert("RGB").save(OUT, quality=95)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
