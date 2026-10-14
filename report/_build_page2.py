# -*- coding: utf-8 -*-
"""Fill slide 1's placeholders with the updated content, and add slide 2
covering 已做的小修改 / 存在的问题 / 下一步计划. Run from report/."""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_TICK_LABEL_POSITION
from pptx.oxml.ns import qn
import copy

SRC = "work_progress_src.pptx"
OUT = "work_progress_out.pptx"

# Theme accents (from theme1.xml, Office default palette already used elsewhere in this deck)
ORANGE = RGBColor(0xF7, 0x96, 0x46)   # accent6 -- matches "已替换/橙框" in the architecture diagram
GREEN = RGBColor(0x9B, 0xBB, 0x59)    # accent3
RED = RGBColor(0xC0, 0x50, 0x4D)      # accent2
BLUE = RGBColor(0x4F, 0x81, 0xBD)     # accent1
DARK = RGBColor(0x40, 0x40, 0x40)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_ORANGE = RGBColor(0xFC, 0xE4, 0xD6)
LIGHT_GREEN = RGBColor(0xEB, 0xF1, 0xDD)
LIGHT_RED = RGBColor(0xF2, 0xDC, 0xDB)

FONT = "Segoe UI"
EA_FONT = "微软雅黑"

prs = Presentation(SRC)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def set_run(run, text, size=11.5, bold=False, color=None, font=FONT):
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = font
    if color is not None:
        run.font.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    ea = rPr.find(qn('a:ea'))
    if ea is None:
        ea = rPr.makeelement(qn('a:ea'), {})
        rPr.append(ea)
    ea.set('typeface', EA_FONT)
    # 关闭拼写检查，否则中文会按英文校对显示红色波浪线
    rPr.set('lang', 'zh-CN')
    rPr.set('altLang', 'en-US')
    rPr.set('noProof', '1')


def clear_paragraphs(text_frame):
    tf = text_frame
    for p in list(tf.paragraphs[1:]):
        p._p.getparent().remove(p._p)
    tf.paragraphs[0].clear()
    return tf


def add_bullet(tf, header_text=None, bullets=None, header_color=None):
    """header_text: bold section header (①/②/③ style). bullets: list of
    (text, bold_spans) where bold_spans is a list of substrings to bold, or
    just plain strings for uniform-weight bullets."""
    first = len(tf.paragraphs) == 1 and not tf.paragraphs[0].runs
    if header_text is not None:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.line_spacing = 1.03
        p.space_after = Pt(4)
        r = p.add_run()
        set_run(r, header_text, size=13.5, bold=True, color=header_color)
    for item in (bullets or []):
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = 1
        p.line_spacing = 1.03
        p.space_after = Pt(4)
        pPr = p._pPr if p._pPr is not None else p._p.get_or_add_pPr()
        pPr.set('marL', '285750')
        pPr.set('indent', '-285750')
        for old in pPr.findall(qn('a:buChar')) + pPr.findall(qn('a:buNone')) + pPr.findall(qn('a:buAutoNum')):
            pPr.remove(old)
        buFont = pPr.makeelement(qn('a:buFont'), {'typeface': 'Arial'})
        buChar = pPr.makeelement(qn('a:buChar'), {'char': '–'})
        pPr.append(buFont)
        pPr.append(buChar)
        if isinstance(item, str):
            r = p.add_run()
            set_run(r, item, size=11.5)
        else:
            for seg_text, seg_bold in item:
                r = p.add_run()
                set_run(r, seg_text, size=11.5, bold=seg_bold)


def add_rrect(slide, x, y, w, h, fill, text_lines, line_color=None, text_color=DARK,
              align=PP_ALIGN.LEFT, font_size=10.5, anchor_ctr=False):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    shape.adjustments[0] = 0.06
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    if line_color is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line_color
        shape.line.width = Pt(1)
    shape.shadow.inherit = False
    tf = shape.text_frame
    tf.word_wrap = True
    if anchor_ctr:
        from pptx.enum.text import MSO_ANCHOR
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Inches(0.12)
    tf.margin_right = Inches(0.12)
    tf.margin_top = Inches(0.08)
    tf.margin_bottom = Inches(0.08)
    first = True
    for line_text, bold, size, color in text_lines:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.alignment = align
        p.space_after = Pt(2)
        r = p.add_run()
        set_run(r, line_text, size=size, bold=bold, color=color)
    return shape


def set_chart_white_bg(chart):
    """Solid white fill behind the whole chart area, so the slide's gradient
    background doesn't bleed through and wash out the gridlines/data."""
    NS_C = 'http://schemas.openxmlformats.org/drawingml/2006/chart'
    NS_A = 'http://schemas.openxmlformats.org/drawingml/2006/main'
    chart_space = chart._chartSpace
    chart_el = chart_space.find(f'{{{NS_C}}}chart')
    spPr = chart_space.makeelement(f'{{{NS_C}}}spPr', {})
    fill = chart_space.makeelement(f'{{{NS_A}}}solidFill', {})
    clr = chart_space.makeelement(f'{{{NS_A}}}srgbClr', {'val': 'FFFFFF'})
    fill.append(clr)
    ln = chart_space.makeelement(f'{{{NS_A}}}ln', {})
    ln.append(chart_space.makeelement(f'{{{NS_A}}}noFill', {}))
    spPr.append(fill)
    spPr.append(ln)
    # spPr must sit right after <c:chart> and before <c:externalData>, per schema
    chart_el.addnext(spPr)


def style_chart_axes(chart, val_fmt=None, cat_min_max=None):
    chart.has_title = True
    cat_ax = chart.category_axis
    val_ax = chart.value_axis
    cat_ax.tick_labels.font.size = Pt(9)
    val_ax.tick_labels.font.size = Pt(9)
    cat_ax.format.line.color.rgb = RGBColor(0xBF, 0xBF, 0xBF)
    val_ax.has_major_gridlines = True
    val_ax.major_gridlines.format.line.color.rgb = RGBColor(0xE7, 0xE6, 0xE6)
    val_ax.major_gridlines.format.line.width = Pt(0.5)
    if val_fmt:
        val_ax.tick_labels.number_format = val_fmt
        val_ax.tick_labels.number_format_is_linked = False


# ---------------------------------------------------------------------------
# Slide 1 -- fill in existing placeholders (整体框架 + 目前进展)
# ---------------------------------------------------------------------------
slide1 = prs.slides[0]
shapes1 = {s.name: s for s in slide1.shapes}

# Title -- standalone report title, no reference to any earlier internship-deck section.
shapes1['Title 1'].width = Inches(5.85)  # widen slightly (safe up to ~6.05in before the diagram at x=6.58in)
title_tf = clear_paragraphs(shapes1['Title 1'].text_frame)
p = title_tf.paragraphs[0]
p.line_spacing = 1.10
r = p.add_run()
set_run(r, "MetaClaw 方法泛化实验进展（一）", size=24, bold=True)

body_tf = clear_paragraphs(shapes1['Content Placeholder 2'].text_frame)
add_bullet(
    body_tf,
    "① 整体框架",
    [
        [("训练方法基于已验证过的 Hybrid RL（GRPO+OPD 混合损失），原样应用在 ", False),
         ("MetaClaw-Bench", True), ("（30 天连续任务流的评测基准）上验证泛化性——训练主干、损失公式、反向传播均未改动，", False)],
        [("只替换了信号来源：确定性 ", False), ("checker", True), (" 取代主观判官、新增任务无关的中间步骤判官（右图橙/绿框）", False)],
    ],
)
add_bullet(
    body_tf,
    "② 目前进展",
    [
        [("只训练到 day 6 实验验证方法本身有效：Acc. ", False), ("17.8%→37.3%", True),
         ("，Compl. ", False), ("0%→13.9%", True), ("（首次非零）——但只训到 day 6，是权宜之计，", False),
         ("终极目标是训满完整 30 天", True)],
    ],
)

caption_tf = clear_paragraphs(shapes1['TextBox 6'].text_frame)
p = caption_tf.paragraphs[0]
p.line_spacing = 1.08
r = p.add_run()
set_run(r, "本次仅迁移 RL 训练方法，不含 MetaClaw 技能库；模型与评测环境均不同，与论文各行为数量级参照", size=10)

# ---------------------------------------------------------------------------
# Slide 2 -- 已做的小修改 / 存在的问题 / 下一步计划
# ---------------------------------------------------------------------------
layout_title_only = prs.slide_masters[0].slide_layouts[5]
slide2 = prs.slides.add_slide(layout_title_only)

t2 = clear_paragraphs(slide2.shapes.title.text_frame)
p = t2.paragraphs[0]
p.line_spacing = 1.10
r = p.add_run()
set_run(r, "MetaClaw 方法泛化实验进展（二）", size=24, bold=True)

MARGIN_L = 0.5
COL1_W = 4.85   # left text column
COL2_X = 5.55   # right visual column
COL2_W = 7.28

# --- Section ① 已做的小修改 (y = 1.25 - 3.55) ---
box1 = slide2.shapes.add_textbox(Inches(MARGIN_L), Inches(1.25), Inches(COL1_W), Inches(2.3))
tf1 = box1.text_frame
tf1.word_wrap = True
add_bullet(
    tf1,
    "① 已做的小修改",
    [
        [("OPD 反馈修复实验：训练流程从此前 day12 左右直接崩溃，", False),
         ("延后到能持续跑完 day22", True), ("（进程不再提前终止）", False)],
        [("进一步定位到确定性 checker 的一类结构性缺陷：部分检查是", False),
         ("累计跨轮次计数", True), ("，本轮做对也可能因更早轮次欠账被判负", False)],
        [("已设计并实现修复（", False), ("只看本轮净增", True),
         ("的判定口径），真实训练核实打分完全正确：", False), ("4 个升级案例全部命中、0 个误判", True)],
    ],
)

# before/after mini-diagram, right of the text in this section
d_y = 1.55
d_h = 1.55
old_box = add_rrect(
    slide2, COL2_X, d_y, 3.1, d_h, LIGHT_ORANGE,
    [
        ("旧逻辑：累计计数", True, 11, DARK),
        ("round2 本轮写对", False, 9.5, DARK),
        ("但历史欠账未补 → 判 FAIL ✗", False, 9.5, RED),
    ],
    line_color=ORANGE,
)
arrow = slide2.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(COL2_X + 3.2), Inches(d_y + d_h/2 - 0.18), Inches(0.55), Inches(0.36))
arrow.fill.solid(); arrow.fill.fore_color.rgb = RGBColor(0xBF, 0xBF, 0xBF); arrow.line.fill.background()
arrow.shadow.inherit = False
new_box = add_rrect(
    slide2, COL2_X + 3.9, d_y, 3.1, d_h, LIGHT_GREEN,
    [
        ("新逻辑：本轮净增", True, 11, DARK),
        ("round2 本轮写对", False, 9.5, DARK),
        ("只看本轮新增文件 → 判 PASS ✓", False, 9.5, RGBColor(0x4F, 0x7A, 0x1E)),
    ],
    line_color=GREEN,
)
cap1 = slide2.shapes.add_textbox(Inches(COL2_X), Inches(d_y + d_h + 0.08), Inches(COL2_W), Inches(0.35))
p = cap1.text_frame.paragraphs[0]
r = p.add_run()
set_run(r, "官方 Acc./Compl. 判定口径不变，只改训练用的奖励信号", size=9.5, color=RGBColor(0x80, 0x80, 0x80))

# --- Section ② 存在的问题 (y = 3.75 - 6.35) ---
box2 = slide2.shapes.add_textbox(Inches(MARGIN_L), Inches(3.75), Inches(COL1_W), Inches(2.5))
tf2 = box2.text_frame
tf2.word_wrap = True
add_bullet(
    tf2,
    "② 存在的问题",
    [
        [("但", False), ("跑得更久不等于效果更好", True),
         ("：OPD 反馈修复实验 day01-22 整体 Acc. 均值 ", False), ("31.0%", True),
         ("，明显低于只训练到 day 6 实验的 ", False), ("41.7%", True), ("（-10.7 个百分点）", False)],
        [("day16 起差距持续扩大，", False), ("day20-22 三天 Acc. 完全归零", True),
         ("——右图逐日对比", False)],
        [("怀疑上游原因：中间步骤独立判官持续给超长 thinking 打正分，被反复强化", False)],
    ],
)

days = list(range(1, 23))
acc_cur = [45.0,53.0,62.5,55.0,71.8,40.7,15.8,39.8,29.7,38.0,36.7,28.2,34.0,23.9,13.3,35.0,17.9,20.8,18.2,0.0,0.0,0.0]
acc_d6  = [40.0,53.0,71.2,53.3,83.3,63.2,21.2,37.9,49.4,46.4,40.0,31.5,37.5,25.8,22.3,46.7,27.0,36.1,35.3,35.6,36.7,29.5]

# 只保留 Acc. 两条线，对比更清楚（Compl. 不入图）
chart_data = CategoryChartData()
chart_data.categories = [f"D{d:02d}" for d in days]
chart_data.add_series("OPD 反馈修复", acc_cur)
chart_data.add_series("只训练到 day 6", acc_d6)

gframe = slide2.shapes.add_chart(
    XL_CHART_TYPE.LINE_MARKERS, Inches(COL2_X), Inches(3.75), Inches(COL2_W), Inches(2.55), chart_data,
)
chart = gframe.chart
chart.has_title = True
chart.chart_title.text_frame.text = "day01-22 逐日 Acc.（%）：OPD 反馈修复 vs 只训练到 day 6"
chart.chart_title.text_frame.paragraphs[0].runs[0].font.size = Pt(11)
chart.chart_title.text_frame.paragraphs[0].runs[0].font.bold = True

series_style = [
    (RED, 2.5),
    (BLUE, 2.5),
]
plot = chart.plots[0]
for s, (color, width) in zip(plot.series, series_style):
    s.format.line.color.rgb = color
    s.format.line.width = Pt(width)
    s.smooth = False
    mk = s.marker
    mk.style = 2  # circle
    mk.format.fill.solid()
    mk.format.fill.fore_color.rgb = color
    mk.format.line.color.rgb = color
    mk.size = 5

chart.has_legend = True
chart.legend.position = XL_LEGEND_POSITION.BOTTOM
chart.legend.include_in_layout = False
chart.legend.font.size = Pt(9)
style_chart_axes(chart)
chart.value_axis.minimum_scale = 0
chart.value_axis.maximum_scale = 100
chart.category_axis.tick_labels.font.size = Pt(7.5)
set_chart_white_bg(chart)

# --- Section ③ 下一步计划 (y = 6.45 - 7.3, full width) ---
box3 = slide2.shapes.add_textbox(Inches(MARGIN_L), Inches(6.45), Inches(12.35), Inches(0.9))
tf3 = box3.text_frame
tf3.word_wrap = True
add_bullet(
    tf3,
    "③ 下一步计划",
    [
        [("中间步骤打分消融实验：保留中间步骤（样本量不变），奖励来源从判官分改为本轮最终确定性结果，验证上述假设", False)],
        [("终极目标：完整训满 30 天，用", False), ("训练前后全程对比", True), ("取代目前只训练到 day 6 这个权宜结果", False)],
    ],
)

prs.save(OUT)
print("slide 1 + slide 2 built")
