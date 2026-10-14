# -*- coding: utf-8 -*-
"""Rewrite slide 5 with user-supplied day01–07 numbers. Do not touch slides 1-4 or THANKS."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from lxml import etree
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

import rebuild_share_v2 as m

HERE = m.HERE
PPT = m.PPT
DAY_TBL = HERE / "day7_compare.png"
PAIR_TBL = HERE / "day_pair.png"
GREEN = (39, 128, 62)

# User-supplied day01–08. Do not alter. Collapse Acc from day09 is not in docs.
DAYS = [
    # day, base_acc, train_acc, d_acc, base_c, train_c, d_c
    ("day01", "68.3", "60.0", "−8.3", "30.0", "20.0", "−10.0"),
    ("day02", "80.3", "71.2", "−9.1", "45.5", "36.4", "−9.1"),
    ("day03", "33.3", "62.5", "+29.2", "33.3", "33.3", "0"),
    ("day04", "66.7", "55.0", "−11.7", "30.0", "20.0", "−10.0"),
    ("day05", "88.5", "87.2", "−1.3", "53.8", "53.8", "0"),
    ("day06", "43.6", "43.6", "0", "10.0", "10.0", "0"),
    ("day07", "36.4", "83.0", "+46.7", "36.4", "63.6", "+27.3"),
    ("day08", "54.1", "49.4", "−4.7", "25.0", "25.0", "0"),
]
SUM_ROW = ("合计 day01–08", "59.2", "64.7", "+5.5", "33.7", "33.7", "0")


def delta_color(s: str):
    if s.startswith("+"):
        return GREEN
    if s.startswith("−") or s.startswith("-"):
        return m.HILITE
    return m.INK


def snapshot(prs, idxs):
    return [etree.tostring(prs.slides[i]._element) for i in idxs]


def render_day7():
    W, H = 2000, 560
    img = Image.new("RGB", (W, H), m.BG)
    d = ImageDraw.Draw(img)
    title = ImageFont.truetype(m.FB, 30)
    cn = ImageFont.truetype(m.F, 24)
    cnb = ImageFont.truetype(m.FB, 24)
    num = ImageFont.truetype(m.SERIF, 24)
    numb = ImageFont.truetype(m.SERIF_BD, 24)
    note = ImageFont.truetype(m.F, 20)

    d.text((16, 8), "前八天对比", font=title, fill=m.INK)
    d.line([(16, 48), (W - 16, 48)], fill=m.RULE, width=2)

    heads = [
        (16, "Day", numb),
        (250, "基线 Acc", cnb),
        (520, "训跑 Acc", cnb),
        (790, "ΔAcc", numb),
        (1040, "基线 Compl", cnb),
        (1380, "训跑 Compl", cnb),
        (1720, "ΔCompl", numb),
    ]
    for x, t, fnt in heads:
        d.text((x, 58), t, font=fnt, fill=m.INK)
    d.line([(16, 94), (W - 16, 94)], fill=m.RULE, width=1)

    xs = (16, 250, 520, 790, 1040, 1380, 1720)
    y = 108
    for row in DAYS:
        d.text((xs[0], y), row[0], font=cn, fill=m.INK)
        d.text((xs[1], y), row[1], font=num, fill=m.INK)
        d.text((xs[2], y), row[2], font=num, fill=m.INK)
        d.text((xs[3], y), row[3], font=numb, fill=delta_color(row[3]))
        d.text((xs[4], y), row[4], font=num, fill=m.INK)
        d.text((xs[5], y), row[5], font=num, fill=m.INK)
        d.text((xs[6], y), row[6], font=numb, fill=delta_color(row[6]))
        y += 40
    d.line([(16, y - 4), (W - 16, y - 4)], fill=m.RULE, width=1)
    d.rectangle([8, y, W - 8, y + 42], fill=m.OURS_BG)
    d.text((xs[0], y + 6), SUM_ROW[0], font=cnb, fill=m.HILITE)
    d.text((xs[1], y + 6), SUM_ROW[1], font=numb, fill=m.HILITE)
    d.text((xs[2], y + 6), SUM_ROW[2], font=numb, fill=m.HILITE)
    d.text((xs[3], y + 6), SUM_ROW[3], font=numb, fill=GREEN)
    d.text((xs[4], y + 6), SUM_ROW[4], font=numb, fill=m.HILITE)
    d.text((xs[5], y + 6), SUM_ROW[5], font=numb, fill=m.HILITE)
    d.text((xs[6], y + 6), SUM_ROW[6], font=numb, fill=m.INK)
    y += 50
    d.line([(16, y), (W - 16, y)], fill=m.RULE, width=2)
    img.save(DAY_TBL)
    print("wrote", DAY_TBL.name, img.size)


UP_BG = (232, 245, 233)
DOWN_BG = (252, 228, 228)


def render_pair():
    """同窗四行对照，列与第 4 页一致：Model | Condition | Acc | Compl."""
    W, H = 1140, 340
    img = Image.new("RGB", (W, H), m.BG)
    d = ImageDraw.Draw(img)
    title = ImageFont.truetype(m.FB, 28)
    cn = ImageFont.truetype(m.F, 26)
    cnb = ImageFont.truetype(m.FB, 26)
    num = ImageFont.truetype(m.SERIF, 26)
    numb = ImageFont.truetype(m.SERIF_BD, 26)
    note = ImageFont.truetype(m.F, 22)

    d.text((12, 8), "本次按天隔离（同窗对照）", font=title, fill=m.INK)
    d.line([(12, 48), (W - 12, 48)], fill=m.RULE, width=2)

    heads = [(12, "Model"), (300, "Condition"), (700, "Acc. (%)"), (920, "Compl. (%)")]
    for x, t in heads:
        d.text((x, 58), t, font=numb, fill=m.INK)
    d.line([(12, 92), (W - 12, 92)], fill=m.RULE, width=1)

    # kind: None | "up" | "down"
    rows = [
        ("Qwen3-4B", "day 0  day01–08", "59.2", "33.7", None),
        ("Qwen3-4B", "Hybrid RL  day01–08", "64.7", "33.7", "up"),
        ("Qwen3-4B", "day 0  day09–20", "48.5", "21.0", None),
        ("Qwen3-4B", "Hybrid RL  day09–20", "15.5", "8.0", "down"),
    ]
    y = 104
    for i, (model, cond, acc, compl, kind) in enumerate(rows):
        if i == 2:
            d.line([(12, y - 6), (W - 12, y - 6)], fill=m.RULE, width=1)
            y += 6
        if kind == "up":
            d.rectangle([8, y - 6, W - 8, y + 36], fill=UP_BG)
            color = GREEN
        elif kind == "down":
            d.rectangle([8, y - 6, W - 8, y + 36], fill=DOWN_BG)
            color = m.HILITE
        else:
            color = m.INK
        hi = kind is not None
        cond_font = cnb if hi else cn
        num_font = numb if hi else num
        acc_color = m.MUTED if acc == "—" else color
        compl_color = m.MUTED if compl == "—" else color
        d.text((12, y), model, font=cond_font, fill=color)
        d.text((300, y), cond, font=cond_font, fill=color)
        d.text((730, y), acc, font=num_font, fill=acc_color)
        d.text((960, y), compl, font=num_font, fill=compl_color)
        y += 42
    d.line([(12, y - 4), (W - 12, y - 4)], fill=m.RULE, width=2)
    d.text((12, y + 8), "前八段 Acc +5.5pt；day09–20 Acc −33pt / Compl −13pt", font=note, fill=m.MUTED)
    img.save(PAIR_TBL)
    print("wrote", PAIR_TBL.name, img.size)


def main():
    render_day7()
    render_pair()
    prs = Presentation(str(PPT))
    assert len(prs.slides) == 6
    before = snapshot(prs, [0, 1, 2, 3, 5])

    slide = prs.slides[4]
    for shape in list(slide.shapes):
        if shape.is_placeholder:
            continue
        shape._element.getparent().remove(shape._element)

    title = next(sh for sh in slide.placeholders if sh.placeholder_format.idx == 0)
    assert "按天隔离" in title.text_frame.text

    # 底栏收进右侧结论，避免两处重复
    body = next(sh for sh in slide.placeholders if sh.placeholder_format.idx == 1)
    body.left, body.top = Inches(6.60), Inches(4.62)
    body.width, body.height = Inches(6.13), Inches(2.20)
    tf = body.text_frame
    tf.clear()
    tf.word_wrap = True

    def conc_p(first=False):
        para = tf.paragraphs[0] if first else tf.add_paragraph()
        para.level = 0
        m.set_no_bullet(para)
        para.line_spacing = 1.12
        para.space_before = Pt(0)
        return para

    p0 = conc_p(True)
    p0.space_after = Pt(6)
    m.add_run(p0, "结论", Pt(18), bold=True)

    p1 = conc_p()
    p1.space_after = Pt(5)
    p1.line_spacing = 1.12
    m.add_run(p1, "① 前八天 Acc 升、Compl 持平；day09–20 相对基线 Acc/Compl 都掉（+5.5pt → −33pt）。", Pt(14))

    p2 = conc_p()
    p2.space_after = Pt(5)
    p2.line_spacing = 1.12
    m.add_run(p2, "② 后半崩，是因为 day09 起大部分题目 timeout。", Pt(14))

    p3 = conc_p()
    p3.space_after = Pt(0)
    p3.line_spacing = 1.12
    m.add_run(p3, "③ timeout 基本是重复 thinking 把 context 顶满。day09 一次 MAIN turn：同一句 ×1388，thinking 15.9 万字，finish_reason=length。", Pt(14))

    # full-width daily table, then pair card at the same 5.90" width as slide 4
    slide.shapes.add_picture(str(DAY_TBL), Inches(0.53), Inches(1.10), width=Inches(12.20))
    slide.shapes.add_picture(str(PAIR_TBL), Inches(0.53), Inches(4.62), width=Inches(5.90))

    foot = slide.shapes.add_textbox(Inches(0.53), Inches(6.98), Inches(12.20), Inches(0.40))
    tf = foot.text_frame
    tf.word_wrap = False
    tf.clear()
    fp = tf.paragraphs[0]
    m.set_no_bullet(fp)
    fp.space_before = Pt(0)
    fp.space_after = Pt(0)
    m.add_run(
        fp,
        "The user's message is a user instruction, "
        "so the assistant's response is a tool call to the write tool.   ",
        Pt(13),
    )
    m.add_run(fp, "×1388", Pt(15), bold=True, color=m.HILITE)

    m.disable_proofing(slide)

    after = snapshot(prs, [0, 1, 2, 3, 5])
    if before != after:
        raise RuntimeError("非第 5 页 XML 变了")

    tmp = HERE / "_share_s5_tmp.pptx"
    prs.save(str(tmp))
    try:
        shutil.copy2(tmp, PPT)
        tmp.unlink(missing_ok=True)
        print("saved", PPT.name)
    except PermissionError:
        alt = HERE / "_share_s5_ready.pptx"
        shutil.copy2(tmp, alt)
        print("locked, wrote", alt.name)
        return

    prs = Presentation(str(PPT))
    print("===== SLIDE 5 TEXT =====")
    for sh in prs.slides[4].shapes:
        if sh.has_text_frame and sh.text_frame.text.strip():
            print(sh.text_frame.text.replace("\n", " | "))
    blob = "\n".join(sh.text_frame.text for sh in prs.slides[4].shapes if sh.has_text_frame)
    print("K leftover", re.findall(r"[Kk]\s*[=＝]\s*[06]", blob))


if __name__ == "__main__":
    main()
