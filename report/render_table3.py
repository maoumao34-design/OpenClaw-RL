# -*- coding: utf-8 -*-
"""从论文 PDF 裁出 Table 3，并高亮 Separate / Student / Hybrid RL 的 19.2。

该格即本次复现（Separate-Student）对应的论文数值。
输出：table3_highlighted.png

依赖：poppler 的 pdftoppm（渲染 PDF 页）、Pillow。
用法：python render_table3.py
"""

import subprocess
from pathlib import Path

from PIL import Image, ImageDraw

PDF = Path("../openclaw-rl-paper.pdf")
PAGE = 11                      # Table 3 所在页
TMP = "_p11"
OUT = "table3_highlighted.png"

CROP = (160, 440, 1500, 825)   # 原图坐标：表格本体（不含正文说明段）
BOX = (766, 174, 856, 216)     # 裁剪后坐标：19.2 所在单元格
RED = (200, 42, 42)


def render_page():
    src = Path(f"{TMP}-{PAGE}.png")
    if not src.exists():
        subprocess.run(["pdftoppm.exe", "-f", str(PAGE), "-l", str(PAGE),
                        "-r", "200", "-png", str(PDF), TMP], check=True)
    return Image.open(src)


def main():
    table = render_page().crop(CROP).convert("RGB")

    # 半透明底色，避免盖住数字
    overlay = Image.new("RGBA", table.size, (0, 0, 0, 0))
    ImageDraw.Draw(overlay).rectangle(BOX, fill=(255, 214, 214, 150))
    table = Image.alpha_composite(table.convert("RGBA"), overlay).convert("RGB")

    ImageDraw.Draw(table).rectangle(BOX, outline=RED, width=4)

    table.save(OUT)
    print(f"已生成 {OUT}  {table.size}")


if __name__ == "__main__":
    main()
