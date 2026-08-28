# -*- coding: utf-8 -*-
"""把 实习小结_毛泽辉.pptx 每页导出为 preview/slideNN.png，用于检查排版。

依赖本机安装的 PowerPoint（通过 COM 调用）。用法：python preview_ppt.py
"""

import os

import win32com.client

SRC = os.path.abspath("实习小结_毛泽辉.pptx")
OUT = os.path.abspath("preview")


def main():
    os.makedirs(OUT, exist_ok=True)
    # 只清 slide 导出图；保留 preview/keep/ 里的版式预览归档
    for name in os.listdir(OUT):
        path = os.path.join(OUT, name)
        if os.path.isfile(path) and name.startswith("slide") and name.endswith(".png"):
            os.remove(path)

    app = win32com.client.Dispatch("PowerPoint.Application")
    pres = app.Presentations.Open(SRC, WithWindow=False)
    try:
        for i, slide in enumerate(pres.Slides, 1):
            slide.Export(os.path.join(OUT, f"slide{i:02d}.png"), "PNG", 1600, 900)
    finally:
        pres.Close()
        app.Quit()

    print(f"已导出 {len(os.listdir(OUT))} 页到 {OUT}")


if __name__ == "__main__":
    main()
