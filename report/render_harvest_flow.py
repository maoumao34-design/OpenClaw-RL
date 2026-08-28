# -*- coding: utf-8 -*-
"""第8页流程图 v2：时间轴 + 少字 + 大留白。

用法：python render_harvest_flow.py
"""

from PIL import Image, ImageDraw, ImageFont

OUT = "harvest_flow.png"
W, H = 2100, 900

F_BOLD = "C:/Windows/Fonts/msyhbd.ttc"
F_REG = "C:/Windows/Fonts/msyh.ttc"

INK = (38, 42, 50)
SOFT = (108, 114, 126)
FAINT = (168, 174, 186)
LINE = (198, 204, 212)
ACCENT = (176, 108, 52)
ACCENT_SOFT = (245, 232, 214)
CHIP_L = (236, 242, 240)
CHIP_R = (245, 232, 214)
CHIP_L_INK = (50, 100, 88)
CHIP_R_INK = ACCENT
SIDE_EDGE = (175, 198, 188)


def font(size, bold=False):
    return ImageFont.truetype(F_BOLD if bold else F_REG, size)


def text(d, xy, s, f, fill, anchor="lt"):
    d.text(xy, s, font=f, fill=fill, anchor=anchor)


def main():
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)

    # 页眉
    text(d, (64, 28), "收获一", font(30, True), ACCENT)
    text(d, (64 + 118, 34), "对 Agent 自进化形成了系统认知", font(22), SOFT)

    spine_x = 100
    y0 = 100
    step_h = 170
    node_r = 26

    # 竖轴
    d.line([(spine_x, y0 + node_r),
            (spine_x, y0 + 3 * step_h + node_r)],
           fill=LINE, width=2)

    # —— 01 必要性 ——
    cy = y0 + node_r
    d.ellipse([spine_x - node_r, cy - node_r, spine_x + node_r, cy + node_r],
              fill=(255, 255, 255, 220), outline=ACCENT, width=2)
    text(d, (spine_x, cy), "01", font(15, True), ACCENT, anchor="mm")
    tx = spine_x + 64
    text(d, (tx, cy - 26), "必要性", font(26, True), INK)
    text(d, (tx, cy + 14), "上线即冻结  ×  任务持续漂移", font(18), SOFT)
    text(d, (tx, cy + 44), "能力衰减是必然", font(15), FAINT)

    # —— 02 技术谱系 ——
    cy = y0 + step_h + node_r
    d.ellipse([spine_x - node_r, cy - node_r, spine_x + node_r, cy + node_r],
              fill=(255, 255, 255, 220), outline=ACCENT, width=2)
    text(d, (spine_x, cy), "02", font(15, True), ACCENT, anchor="mm")
    text(d, (tx, cy - 26), "技术谱系", font(26, True), INK)

    chip_y = cy + 16
    chip_h, chip_w, gap = 72, 300, 72
    lx, rx = tx, tx + chip_w + gap

    d.rounded_rectangle([lx, chip_y, lx + chip_w, chip_y + chip_h],
                        radius=12, fill=CHIP_L, outline=SIDE_EDGE, width=1)
    text(d, (lx + chip_w / 2, chip_y + 20), "非参数化",
         font(19, True), CHIP_L_INK, anchor="mt")
    text(d, (lx + chip_w / 2, chip_y + 48), "记忆 / 技能库",
         font(14), SOFT, anchor="mt")

    d.rounded_rectangle([rx, chip_y, rx + chip_w, chip_y + chip_h],
                        radius=12, fill=CHIP_R, outline=(218, 172, 118), width=1)
    text(d, (rx + chip_w / 2, chip_y + 20), "参数化",
         font(19, True), CHIP_R_INK, anchor="mt")
    text(d, (rx + chip_w / 2, chip_y + 48), "权重更新",
         font(14), SOFT, anchor="mt")

    mx0, mx1 = lx + chip_w + 8, rx - 8
    my = chip_y + chip_h / 2
    d.line([(mx0, my), (mx1, my)], fill=FAINT, width=1)
    text(d, ((mx0 + mx1) / 2, my - 12), "互补", font(12), FAINT, anchor="mm")

    # —— 03 我的落点 ——
    cy = y0 + 2 * step_h + node_r
    d.ellipse([spine_x - node_r, cy - node_r, spine_x + node_r, cy + node_r],
              fill=(255, 255, 255, 220), outline=ACCENT, width=2)
    text(d, (spine_x, cy), "03", font(15, True), ACCENT, anchor="mm")
    text(d, (tx, cy - 26), "我的落点", font(26, True), INK)

    # 「主线工作」标签贴在标题右侧
    tag = "主线工作"
    tag_f = font(12, True)
    tag_w = d.textlength(tag, font=tag_f) + 18
    tag_x = tx + 130
    tag_y = cy - 28
    d.rounded_rectangle([tag_x, tag_y, tag_x + tag_w, tag_y + 26],
                        radius=8, fill=ACCENT_SOFT, outline=ACCENT, width=1)
    text(d, (tag_x + 9, tag_y + 13), tag, tag_f, ACCENT, anchor="lm")

    text(d, (tx, cy + 14), "OpenClaw-RL  ·  参数化一端", font(18), SOFT)
    text(d, (tx, cy + 44), "边服务边进化", font(15), FAINT)

    # —— 04 结论 ——
    cy = y0 + 3 * step_h + node_r
    d.ellipse([spine_x - node_r, cy - node_r, spine_x + node_r, cy + node_r],
              fill=(255, 255, 255, 220), outline=ACCENT, width=2)
    text(d, (spine_x, cy), "04", font(15, True), ACCENT, anchor="mm")
    text(d, (tx, cy - 26), "结论", font(26, True), INK)
    text(d, (tx, cy + 14), "通往 AGI 的必要路径", font(18), SOFT)
    text(d, (tx, cy + 44), "没有完备路线，值得并行探索", font(15), FAINT)

    # —— 旁支卡片（小、靠上，对齐谱系）——
    side_l, side_t = 1520, 200
    side_w, side_h = 480, 230
    d.rounded_rectangle([side_l, side_t, side_l + side_w, side_t + side_h],
                        radius=14, fill=(255, 255, 255, 150),
                        outline=SIDE_EDGE, width=1)
    text(d, (side_l + 28, side_t + 22), "收获二 · 旁支", font(14, True), CHIP_L_INK)
    text(d, (side_l + 28, side_t + 52), "试用 Multica", font(22, True), INK)
    text(d, (side_l + 28, side_t + 90), "非参数化路线的对照", font(15), SOFT)
    d.line([(side_l + 28, side_t + 122), (side_l + side_w - 28, side_t + 122)],
           fill=(210, 220, 215), width=1)
    for j, s in enumerate(["完整产品开发", "印证两条路互补", "反哺平台改进"]):
        text(d, (side_l + 28, side_t + 140 + j * 26), "·  " + s, font(15), SOFT)

    # 连线：从非参数化芯片底边中点 → 下 → 右 → 卡片
    # 避免横穿「参数化」芯片
    ax = lx + chip_w / 2
    ay = chip_y + chip_h
    drop_y = ay + 28
    mid_x = side_l - 40
    card_y = side_t + side_h / 2

    d.line([(ax, ay + 2), (ax, drop_y)], fill=SIDE_EDGE, width=1)
    d.line([(ax, drop_y), (mid_x, drop_y)], fill=SIDE_EDGE, width=1)
    d.line([(mid_x, drop_y), (mid_x, card_y)], fill=SIDE_EDGE, width=1)
    d.line([(mid_x, card_y), (side_l - 2, card_y)], fill=SIDE_EDGE, width=1)
    d.polygon([(side_l, card_y), (side_l - 9, card_y - 5), (side_l - 9, card_y + 5)],
              fill=SIDE_EDGE)
    text(d, ((ax + mid_x) / 2, drop_y - 14), "对照", font(12), CHIP_L_INK, anchor="mm")

    im.save(OUT)
    print(f"wrote {OUT} ({W}x{H})")


if __name__ == "__main__":
    main()
