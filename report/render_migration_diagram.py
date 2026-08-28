# -*- coding: utf-8 -*-
"""在 OpenClaw-RL 复现细节图（第5页那张）基础上叠加标注，生成
MetaClaw 迁移对照图 metaclaw_migration_diagram.png。

四种状态，一眼可辨：
  - 灰化 + "不适用"：三角色场景专属机制，MetaClaw 没有对应物，直接不需要
  - 橙框 + "已替换"：原地保留位置，但内容/来源换了
  - 绿框 + "新增"：MetaClaw 独有，原图没有对应物，画在右侧扩展区
  - 不作任何标记：完全不变，照原样保留（含原图像素本身）

所有坐标均由 _detect_boxes.py 对原图做连通域检测 + 人工核对得到，
不是估的。

用法：python render_migration_diagram.py
"""

from PIL import Image, ImageDraw, ImageFont

SRC = "personal_agent_dialogue_vs_training_loop.png"
OUT = "metaclaw_migration_diagram.png"

F_BOLD = "C:/Windows/Fonts/msyhbd.ttc"
F_REG = "C:/Windows/Fonts/msyh.ttc"

GRAY_TAG = (120, 120, 120)
ORANGE = (204, 106, 0)
ORANGE_FILL = (255, 238, 219)
GREEN_NEW = (21, 128, 61)
GREEN_FILL = (223, 246, 231)
WHITE = (255, 255, 255)
INK = (34, 38, 46)

# ---------------------------------------------------------------- 工具函数

def font(size, bold=False):
    return ImageFont.truetype(F_BOLD if bold else F_REG, size)


def centered(d, cx, cy, text, f, fill):
    b = d.textbbox((0, 0), text, font=f)
    d.text((cx - (b[2]-b[0])/2 - b[0], cy - (b[3]-b[1])/2 - b[1]), text, font=f, fill=fill)


def wrap(d, text, f, width):
    lines, cur = [], ""
    for ch in text:
        trial = cur + ch
        if d.textlength(trial, font=f) <= width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines


def fade_region(im, box, pad=10):
    """把区域整体去饱和并提亮，制造"已不适用"的灰化效果。"""
    x0, y0, x1, y1 = box[0]-pad, box[1]-pad, box[2]+pad, box[3]+pad
    region = im.crop((x0, y0, x1, y1)).convert("L").convert("RGB")
    faded = Image.blend(region, Image.new("RGB", region.size, WHITE), 0.55)
    im.paste(faded, (x0, y0))


def tag(d, x, y, text, color, fill, align="left"):
    f = font(19, bold=True)
    b = d.textbbox((0, 0), text, font=f)
    tw, th = b[2]-b[0]+18, b[3]-b[1]+10
    x0 = x if align == "left" else x - tw
    d.rounded_rectangle([x0, y, x0+tw, y+th], radius=6, fill=fill, outline=color, width=2)
    d.text((x0+9-b[0], y+5-b[1]), text, font=f, fill=color)
    return x0, y, x0+tw, y+th


def strike(d, box, color=GRAY_TAG, width=3):
    x0, y0, x1, y1 = box
    d.line([x0, y0, x1, y1], fill=color, width=width)
    d.line([x0, y1, x1, y0], fill=color, width=width)


def replace_box(im, d, box, new_lines, title_size=24, body_size=19):
    """原地重绘一个框：先用白底盖掉原内容，再画橙色边框和新文字。"""
    x0, y0, x1, y1 = box
    d.rounded_rectangle([x0, y0, x1, y1], radius=14, fill=ORANGE_FILL,
                        outline=ORANGE, width=4)
    ty = y0 + 16
    for i, (text, is_title) in enumerate(new_lines):
        f = font(title_size if is_title else body_size, bold=is_title)
        for line in wrap(d, text, f, x1-x0-28):
            centered(d, (x0+x1)/2, ty + (title_size if is_title else body_size)*0.62, line, f, ORANGE)
            ty += (title_size if is_title else body_size) + 8
        ty += 4


def new_box(d, box, lines, title_size=24, body_size=18):
    x0, y0, x1, y1 = box
    d.rounded_rectangle([x0, y0, x1, y1], radius=14, fill=GREEN_FILL,
                        outline=GREEN_NEW, width=4)
    ty = y0 + 16
    for text, is_title in lines:
        f = font(title_size if is_title else body_size, bold=is_title)
        for line in wrap(d, text, f, x1-x0-28):
            centered(d, (x0+x1)/2, ty + (title_size if is_title else body_size)*0.62,
                    line, f, GREEN_NEW if is_title else INK)
            ty += (title_size if is_title else body_size) + 8
        ty += 4


# ---------------------------------------------------------------- 坐标（原图 1840×2600，由 _detect_boxes.py 检测）

SEPARATE = (461, 77, 899, 195)
JOINT = (941, 77, 1379, 195)
HOMEWORK_ROW = (438, 268, 1360, 404)          # 含"作业文件在角色间传递"标签
ROLES_ROW = (441, 484, 1359, 595)             # 含"Simulator 扮演三种角色"标签
SIMULATOR = (701, 689, 1099, 799)

HINT_JUDGE = (978, 1410, 1319, 1540)
EVAL_JUDGE = (1319, 1410, 1660, 1540)
DASHED_LABEL_Y = 1359                          # "同一个 PRM 服务端点…" 标签文字的垂直中心
DASHED_BORDER_Y = 1379                         # 虚线分组框的上边框，实测所得

FILTER_HINT = (978, 1606, 1319, 1720)          # 筛出采纳的 hint
MAJORITY_VOTE = (1319, 1606, 1660, 1720)       # 多数投票

BOTTOM_CAPTION = (442, 2522, 1383, 2547)       # "收敛指标：…" 那一行，实测所得


def main():
    im = Image.open(SRC).convert("RGB")
    W, H = im.size

    # 画布右侧扩展 560px，放两块"新增"内容，避免跟原图挤在一起
    MARGIN = 560
    canvas = Image.new("RGB", (W + MARGIN, H), WHITE)
    canvas.paste(im, (0, 0))
    d = ImageDraw.Draw(canvas)

    # === ① 灰化：三角色场景专属机制，MetaClaw 不需要 ===
    fade_region(canvas, (min(SEPARATE[0], JOINT[0])-20, SEPARATE[1]-46,
                         max(SEPARATE[2], JOINT[2])+20, SEPARATE[3]+8))
    fade_region(canvas, (HOMEWORK_ROW[0]-20, HOMEWORK_ROW[1]-30,
                         HOMEWORK_ROW[2]+20, HOMEWORK_ROW[3]+8))
    fade_region(canvas, (ROLES_ROW[0]-20, ROLES_ROW[1]-30,
                         ROLES_ROW[2]+20, ROLES_ROW[3]+8))

    for box in (SEPARATE, JOINT):
        strike(d, box)
    for box in (HOMEWORK_ROW[:2]+(0,0), ):
        pass  # 占位，避免误删逻辑（homework 三个子框整体划一条线即可）
    strike(d, (438, 292, 1360, 402))
    strike(d, (441, 509, 1359, 595))

    tag(d, JOINT[2]+16, SEPARATE[1]+10, "不适用", GRAY_TAG, WHITE)
    tag(d, 1360+16, 292, "不适用", GRAY_TAG, WHITE)
    tag(d, 1359+16, 509, "不适用", GRAY_TAG, WHITE)

    # === ② 橙框替换：Simulator → MetaClaw-Bench 真实环境 ===
    replace_box(canvas, d, SIMULATOR, [
        ("MetaClaw-Bench", True),
        ("CLI agent 直接执行真实任务，", False),
        ("不再是 Simulator 对话", False),
    ], title_size=25, body_size=18)
    tag(d, SIMULATOR[2]-118, SIMULATOR[1]-34, "已替换", ORANGE, ORANGE_FILL)

    # === ③ 橙框替换：Hint / Eval 判官 → Bench 自身反馈 / 确定性 checker ===
    replace_box(canvas, d, HINT_JUDGE, [
        ("OPD Hint", True),
        ("来自 Bench 反馈", False),
        ("(stdout / 选项说明)", False),
    ])
    replace_box(canvas, d, EVAL_JUDGE, [
        ("Eval 奖励", True),
        ("确定性 checker", False),
        ("(exit code / 精确匹配)", False),
    ])
    tag(d, HINT_JUDGE[0]+8, HINT_JUDGE[1]-34, "已替换", ORANGE, ORANGE_FILL)
    tag(d, EVAL_JUDGE[2]-118, EVAL_JUDGE[1]-34, "已替换", ORANGE, ORANGE_FILL)

    # 原 "同一个 PRM 服务端点…" 分组标签已不成立（判官不再是 PRM）：
    # 划掉标签文字本身（y≈1350-1369），在虚线框上边框与两个判官框之间的
    # 窄条（y≈1379-1410）里放一行短注释，避免跟原文字重叠
    d.line([HINT_JUDGE[0], DASHED_LABEL_Y, EVAL_JUDGE[2], DASHED_LABEL_Y],
           fill=GRAY_TAG, width=3)
    fnote = font(16)
    note = "→ 不再是 PRM 判官"
    centered(d, (HINT_JUDGE[0]+EVAL_JUDGE[2])/2, (DASHED_BORDER_Y+HINT_JUDGE[1])/2, note, fnote, ORANGE)

    # === ④ 灰化：筛出 hint / 多数投票——MetaClaw 是单一确定性结果，无需筛选/投票 ===
    fade_region(canvas, (FILTER_HINT[0]-6, FILTER_HINT[1]-6, MAJORITY_VOTE[2]+6, MAJORITY_VOTE[3]+6))
    strike(d, FILTER_HINT)
    strike(d, MAJORITY_VOTE)
    tag(d, FILTER_HINT[0]+8, FILTER_HINT[1]-34, "不适用：单一确定性结果", GRAY_TAG, WHITE)

    # === ⑤ 灰化：底部收敛指标（Table 3 口径，MetaClaw 不适用）===
    fade_region(canvas, (BOTTOM_CAPTION[0]-10, BOTTOM_CAPTION[1]-6,
                         BOTTOM_CAPTION[2]+10, BOTTOM_CAPTION[3]+6))
    strike(d, BOTTOM_CAPTION, width=2)
    fnote2 = font(18)
    centered(d, (BOTTOM_CAPTION[0]+BOTTOM_CAPTION[2])/2, BOTTOM_CAPTION[3]+26,
            "MetaClaw：改用 Table 1 Acc. / Compl.，同一趟训练全程实时聚合", fnote2, ORANGE)

    # === ⑥ 新增：右侧扩展区两块绿框 ===
    NX0 = W + 30
    NX1 = W + MARGIN - 30
    new_box(d, (NX0, 1240, NX1, 1420), [
        ("中间轮次步骤判官", True),
        ("round 内多轮 tool-call，", False),
        ("任务无关判官独立打分，", False),
        ("不聚合进 round reward，不进 OPD", False),
    ], title_size=25, body_size=18)
    tag(d, NX0, 1206, "新增", GREEN_NEW, GREEN_FILL)

    new_box(d, (NX0, 1470, NX1, 1650), [
        ("串行 rollout driver", True),
        ("day01 → day30 严格顺序、", False),
        ("concurrency=1，保证后面天数", False),
        ("用得上前面天数训完的权重", False),
    ], title_size=25, body_size=18)
    tag(d, NX0, 1436, "新增", GREEN_NEW, GREEN_FILL)

    # 两块新增框跟主图之间画一条弱连接线，提示它们挂在训练循环里
    d.line([EVAL_JUDGE[2], (HINT_JUDGE[1]+HINT_JUDGE[3])//2,
           NX0, (1240+1420)//2], fill=GREEN_NEW, width=2)

    # === ⑦ 图例（画在右侧扩展区顶部）===
    # 整张图最终会被缩到 slide 上很小的尺寸（跟第5页那张同一量级），
    # 图例字号要跟原图里 box 标题的视觉分量匹配，不能比正文还小
    f_leg_title = font(38, bold=True)
    f_leg = font(24)
    lx, ly = NX0, 30
    d.text((lx, ly), "图例", font=f_leg_title, fill=INK)
    ly += 60
    legend = [
        (GRAY_TAG, WHITE, "灰化 + 划线", "不适用，MetaClaw 不需要"),
        (ORANGE, ORANGE_FILL, "橙框", "位置不变，内容/来源已替换"),
        (GREEN_NEW, GREEN_FILL, "绿框", "MetaClaw 独有，原图没有"),
        (None, None, "无标记", "训练主干完全不变"),
    ]
    for color, fill, label, desc in legend:
        if color:
            d.rounded_rectangle([lx, ly, lx+36, ly+36], radius=6, fill=fill, outline=color, width=4)
        else:
            d.rectangle([lx, ly, lx+36, ly+36], outline=INK, width=3)
        d.text((lx+48, ly+3), label, font=font(26, bold=True), fill=INK)
        for i, line in enumerate(wrap(d, desc, f_leg, NX1-lx-48)):
            d.text((lx+48, ly+40+i*30), line, font=f_leg, fill=GRAY_TAG)
        ly += 40 + 30*len(wrap(d, desc, f_leg, NX1-lx-48)) + 22

    canvas.save(OUT)
    print(f"已生成 {OUT}  {canvas.size}")


if __name__ == "__main__":
    main()
