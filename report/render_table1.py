# -*- coding: utf-8 -*-
"""生成第7页用的 Table 1 对照图 table1_with_ours.png。

上半部分：直接裁自 MetaClaw 论文 PDF 第 7 页 Table 1 的 **Part I** 区域（30 days, 346 Q，
与本次迁移实验的设置完全一致；Part II 不在本次范围内，裁掉）。
下半部分：本次迁移的两行结果，用同一套排版接在论文表格下方，浅色底纹区分。

依赖：poppler 的 pdftoppm、Pillow。
用法：python render_table1.py
"""

import glob
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PDF = Path("D:/MAO/Paper/MetaClaw/MetaClaw_2603.17187.pdf")
PAGE = 7                        # Table 1 所在页
TMP = "_mc7"
OUT = "table1_with_ours.png"

CROP = (300, 235, 1010, 585)    # 页面坐标：Table 1 的 Part I 区域

# 以下均为裁剪后坐标（宽 710），全部由论文表格实测得出，不要凭感觉调：
#   列起点/中心 —— 对 Kimi 各行做逐列暗像素扫描
#   行距 39、字号 26 —— 实测「Baseline」墨迹宽 89px、高 18px，Times 26px 为 90/18
#   行与横线的间距 19 —— 论文每条分隔线到下一行文字顶部的距离
COL_MODEL, COL_COND = 12, 162   # 前两列左对齐起点
COL_ACC, COL_COMPL = 452, 619   # 后两列数字的居中基准
ROW_H = 39                      # 论文表格行距
RULE_GAP = 19                   # 横线到下一行文字顶部
FONT_PX = 26

SERIF = "C:/Windows/Fonts/times.ttf"
SERIF_BD = "C:/Windows/Fonts/timesbd.ttf"

INK = (0, 0, 0)
RULE = (0, 0, 0)
OURS_BG = (255, 244, 214)       # 本次结果底纹，跟论文行区分
HILITE = (176, 42, 42)          # 训练后一行的数字用红色

# (模型, 条件, Acc., Compl., 是否为最终结果行)
OURS = [
    ("Qwen3-4B", "Baseline", "17.8", "0.0", False),
    ("Qwen3-4B", "Hybrid RL (Ours)", "37.3", "13.9", True),
]


def render_page():
    hits = glob.glob(f"{TMP}-*.png")
    if not hits:
        subprocess.run(["pdftoppm.exe", "-f", str(PAGE), "-l", str(PAGE),
                        "-r", "200", "-png", str(PDF), TMP], check=True)
        hits = glob.glob(f"{TMP}-*.png")
    return Image.open(hits[0])


def last_rule_y(img):
    """找论文表格最下面那条横线的 y，本次结果从它下方接着排。"""
    g = img.convert("L")
    w, h = g.size
    px = g.load()
    for y in range(h - 1, -1, -1):
        if sum(1 for x in range(w) if px[x, y] < 128) > w * 0.5:
            return y
    raise RuntimeError("没找到表格底线")


def main():
    table = render_page().crop(CROP).convert("RGB")
    w = table.width
    base = last_rule_y(table)                    # 论文表格底线
    first_top = base + RULE_GAP                  # 本次结果第一行的墨迹顶部

    canvas = Image.new("RGB", (w, first_top + ROW_H * len(OURS) + 28), "white")
    canvas.paste(table, (0, 0))

    d = ImageDraw.Draw(canvas)
    f = ImageFont.truetype(SERIF, FONT_PX)
    fb = ImageFont.truetype(SERIF_BD, FONT_PX)

    def put(text, font, fill, ink_top, x=None, cx=None):
        """按墨迹顶部对齐，而不是按行框顶部——后者随字体内部留白浮动。"""
        x0, y0, x1, _ = d.textbbox((0, 0), text, font=font)
        left = x if x is not None else cx - (x1 - x0) / 2 - x0
        d.text((left, ink_top - y0), text, font=font, fill=fill)

    tint_top = first_top - RULE_GAP + 4
    tint_bottom = first_top + ROW_H * (len(OURS) - 1) + 26
    d.rectangle([0, tint_top, w, tint_bottom], fill=OURS_BG)

    for i, (model, cond, acc, compl, final) in enumerate(OURS):
        top = first_top + ROW_H * i
        face = fb if final else f
        color = HILITE if final else INK
        put(model, f, INK, top, x=COL_MODEL)
        put(cond, face, color, top, x=COL_COND)
        put(acc, face, color, top, cx=COL_ACC)
        put(compl, face, color, top, cx=COL_COMPL)

    d.rectangle([0, tint_bottom + 1, w, tint_bottom + 2], fill=RULE)

    canvas.save(OUT)
    print(f"已生成 {OUT}  {canvas.size}（论文底线 y={base}）")


if __name__ == "__main__":
    main()
