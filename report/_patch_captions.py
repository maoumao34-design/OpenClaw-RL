# -*- coding: utf-8 -*-
"""Strip process-narrative captions on slide 4. Leave other slides unchanged."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pptx.util import Pt

import rebuild_share_v2 as m

HERE = m.HERE
PPT = m.PPT
BAK = HERE / "Openclaw-RL复现及迁移内容分享_backup_更新实验页前_20260917_1001.pptx"

BANNED = re.compile(
    r"修复后|环境修复|09-05|训练窗|61\.4|45\.0|定版"
)


def snapshot(prs, idxs):
    return [etree.tostring(prs.slides[i]._element) for i in idxs]


def para_text(para):
    return "".join(r.text or "" for r in para.runs) or para.text


def set_para(para, text, size, bold=None, color=None):
    if para.runs:
        para.runs[0].text = text
        for r in para.runs[1:]:
            r.text = ""
        m.style_run(para.runs[0], size, bold=bold, color=color)
    else:
        run = para.add_run()
        run.text = text
        m.style_run(run, size, bold=bold, color=color)


def main():
    m.render_ours_table()
    prs = Presentation(str(PPT))
    assert len(prs.slides) == 6
    before_other = snapshot(prs, [0, 1, 2, 4, 5])

    slide = prs.slides[3]
    replaced = 0
    for shape in list(slide.shapes):
        if shape.shape_type == MSO_SHAPE_TYPE.PICTURE and 2.7 < shape.top.inches < 3.3:
            left, top, width = shape.left, shape.top, shape.width
            shape._element.getparent().remove(shape._element)
            slide.shapes.add_picture(str(m.OURS_TBL), left, top, width=width)
            replaced += 1
    assert replaced == 1, replaced

    for shape in slide.shapes:
        if not shape.has_text_frame:
            continue
        for para in shape.text_frame.paragraphs:
            raw = para_text(para)
            if not raw:
                continue
            if "环境修复后定版" in raw or ("上表为本次按题隔离" in raw and "修复" in raw):
                set_para(para, "上表为本次按题隔离，不与论文 Table 1 数字直接对比", Pt(10))
                print("caption ->", para_text(para))
            elif "训练窗" in raw or "61.4" in raw or "09-05" in raw:
                set_para(
                    para,
                    "day 6 相对同路径 day 0：Acc +2.9pt / Compl +1.8pt",
                    Pt(11.5),
                )
                print("body ->", para_text(para))

    m.disable_proofing(slide)

    after_other = snapshot(prs, [0, 1, 2, 4, 5])
    if before_other != after_other:
        raise RuntimeError("非第 4 页 XML 变了")
    bak = Presentation(str(BAK))
    for i in range(3):
        if etree.tostring(prs.slides[i]._element) != etree.tostring(bak.slides[i]._element):
            raise RuntimeError(f"第 {i+1} 页相对原版备份变了")

    tmp = HERE / "_share_caption_tmp.pptx"
    prs.save(str(tmp))
    try:
        shutil.copy2(tmp, PPT)
        tmp.unlink(missing_ok=True)
        print("saved", PPT.name)
    except PermissionError:
        alt = HERE / "_share_caption_ready.pptx"
        shutil.copy2(tmp, alt)
        print("locked, wrote", alt.name)
        return

    prs = Presentation(str(PPT))
    print("===== SLIDE 4 TEXT =====")
    hits = []
    for sh in prs.slides[3].shapes:
        if sh.has_text_frame and sh.text_frame.text.strip():
            t = sh.text_frame.text.replace("\n", " | ")
            print(t)
            hits.extend(BANNED.findall(t))
    print("===== SLIDE 5 TEXT =====")
    for sh in prs.slides[4].shapes:
        if sh.has_text_frame and sh.text_frame.text.strip():
            t = sh.text_frame.text.replace("\n", " | ")
            print(t)
            hits.extend(BANNED.findall(t))
    print("BANNED leftover", hits)


if __name__ == "__main__":
    main()
