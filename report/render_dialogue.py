# -*- coding: utf-8 -*-
"""渲染第6页用的对话对照图 dialogue_evidence.png。

素材取自 Separate-Student 训练日志（20260811_170852），四块按时间顺序：
  ① 第 1 个 session   —— Turn 1 违规，需用户要求才达标
  ② 第 13 个 session  —— 中期，仍需用户要求
  ③ 第 17 个 session  —— 转折前一个，仍需用户要求
  ④ 第 18~20 个 session —— Turn 1 直接达标，连续 3 次 → 收敛于第 20 个

用法：python render_dialogue.py
"""

from PIL import Image, ImageDraw, ImageFont

OUT = "dialogue_evidence.png"

W, PAD = 1720, 34
BG = (255, 255, 255)
INK = (34, 38, 46)
MUTED = (110, 118, 130)
RED = (200, 42, 42)
GREEN = (22, 130, 86)
BLUE = (52, 88, 168)
RULE = (222, 226, 232)
BAND = (244, 246, 249)
OKBG = (238, 248, 242)

F = "C:/Windows/Fonts/msyh.ttc"
FB = "C:/Windows/Fonts/msyhbd.ttc"
f_sec = ImageFont.truetype(FB, 27)     # 分块标题
f_tag = ImageFont.truetype(FB, 22)     # 角色标签
f_txt = ImageFont.truetype(F, 22)      # 正文
f_note = ImageFont.truetype(F, 20)     # 注释

# (分块标题, 副标题, [(角色, 文本, 状态)])，状态：bad / ok / none / accept
# 文本内 \n 保留为换行（列表须逐行呈现才看得出违规），**…** 之间标红
BLOCKS = [
    ("第 1 个 session", "Turn 1 不达标 —— 用户提出要求后才改好", [
        ("模型 Turn 1", "…they need 100 − 64 = 36 more signatures.\n"
                        "**Answer:** 36 signatures", "bad"),
        ("学生", "Can you rewrite that in a more casual way? Like, drop the bold and the "
                 "equation look, just say it naturally.", "none"),
        ("模型 Turn 2", "So far, Carol has 20 signatures and Jennifer has 44—that's 64 total. "
                        "They want to hit 100…", "ok"),
    ]),
    ("第 13 个 session", "仍然如此 —— 中期没有改善", [
        ("模型 Turn 1", "The handbag cost $220. Here's the step-by-step solution:\n"
                        "**1.** Shoes cost $80.\n"
                        "**2.** 3 times the shoes' cost: $80 × 3 = $240\n"
                        "**3.** Subtract $20: $240 − $20 = $220\n"
                        "So Jillian's handbag cost **$220**.", "bad"),
        ("学生", "That actually looks pretty good, but it's too neat and like a robot wrote it. "
                 "Can you make it sound more…", "none"),
        ("模型 Turn 2", "Jillian's shoes cost $80. If you triple that amount, it's $240. Then, "
                        "since the handbag is $20 less than that, you subtract $20 to get $220.", "ok"),
    ]),
    ("第 17 个 session", "转折前最后一次 —— 用户仍需提要求", [
        ("模型 Turn 1", "The daisies have 15 more petals than the orchids. Here's the step-by-step:\n"
                        "**1. Orchids total petals**: 5 orchids × 5 petals = 25 petals\n"
                        "**2. Daisies total petals**: 4 daisies × 10 petals = 40 petals", "bad"),
        ("学生", "Can you rewrite that so it sounds more natural? Like, just a normal explanation "
                 "without the bold stuff…", "none"),
        ("模型 Turn 2", "Joelle has 5 orchids with 5 petals each, so that's 25 petals total. "
                        "She also has 4 African daisies…", "ok"),
    ]),
    ("第 18 – 20 个 session", "Turn 1 直接达标 —— 用户不再提出任何格式要求", [
        ("第18个 Turn 1", "Jen spends $93 on food for her 3 fish in the month of May. Here's the "
                          "calculation: 3 fish × $1/day × 31 days (May has 31 days) = $93.", "ok"),
        ("第19个 Turn 1", "The total amount they pay is $38. Here's the calculation:\n"
                          "- 2 adult tickets: 2 × $12 = $24\n"
                          "- 1 child ticket: 1 × $8 = $8\n"
                          "- 2 popcorns: 2 × $3 = $6\n"
                          "Total: $24 + $8 + $6 = $38", "ok"),
        ("第20个 Turn 1", "Kelly will have 90 cents left after buying the can of pop.", "ok"),
    ]),
]

HEADER = ("每个 session 的第一句请求都相同：",
          "“Hey, I have my homework in homework/N.txt … Show me the answer first.”")

# 注意反斜杠须写成 \\，否则 \b 会被解析成退格符，渲染出方框
STU_RULE = "学生偏好规则：不含 bold、编号列表、\\boxed{}"
LEGEND = "红色 = 本次日志实际出现的违反（\\boxed{} 未出现过）"
FOOTER = "连续 3 个 session 的 Turn 1 均达标 → 本次复现收敛于第 20 个 session"

TAG_W = 178          # 角色标签列宽
TXT_X = PAD + TAG_W + 16
TXT_W = W - TXT_X - PAD - 40


def wrap(draw, text, font, width):
    """按宽度折行；文本内的 \\n 视为强制换行（列表须逐行呈现）。"""
    lines = []
    for para in text.split("\n"):
        cur = ""
        for w in para.split(" "):
            trial = f"{cur} {w}".strip()
            # 计算宽度时忽略 ** 标记，它不参与渲染
            if draw.textlength(trial.replace("**", ""), font=font) <= width:
                cur = trial
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    return lines


def measure():
    """先量高度，再按实际高度建图。"""
    img = Image.new("RGB", (W, 10), BG)
    d = ImageDraw.Draw(img)
    h = PAD + 74                                   # 顶部说明
    for _, _, rows in BLOCKS:
        h += 46                                    # 分块标题
        for _, text, _ in rows:
            h += 30 * len(wrap(d, text, f_txt, TXT_W)) + 14
        h += 18
    return h + 70                                  # 底部结论


def draw_marked(d, x, y, text, font, base):
    """把 **…** 之间的内容标红，其余用 base 色。"""
    for i, seg in enumerate(text.split("**")):
        if not seg:
            continue
        color = RED if i % 2 else base
        d.text((x, y), seg, font=font, fill=color)
        x += d.textlength(seg, font=font)


def main():
    H = measure()
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # 顶部：统一的开场请求
    d.rectangle([PAD, PAD, W - PAD, PAD + 58], fill=BAND)
    d.text((PAD + 16, PAD + 6), HEADER[0], font=f_note, fill=MUTED)
    d.text((PAD + 16, PAD + 30), HEADER[1], font=f_note, fill=BLUE)
    rw = d.textlength(STU_RULE, font=f_note)
    d.text((W - PAD - 16 - rw, PAD + 6), STU_RULE, font=f_note, fill=INK)
    lw = d.textlength(LEGEND, font=f_note)
    d.text((W - PAD - 16 - lw, PAD + 30), LEGEND, font=f_note, fill=RED)
    y = PAD + 78

    for title, sub, rows in BLOCKS:
        d.text((PAD, y), title, font=f_sec, fill=INK)
        tw = d.textlength(title, font=f_sec)
        d.text((PAD + tw + 14, y + 6), sub, font=f_note, fill=MUTED)
        y += 40
        d.line([PAD, y, W - PAD, y], fill=RULE, width=2)
        y += 8

        for tag, text, state in rows:
            lines = wrap(d, text, f_txt, TXT_W)
            block_h = 30 * len(lines)
            if state in ("ok", "accept"):
                d.rectangle([PAD, y - 4, W - PAD, y + block_h + 6], fill=OKBG)

            tag_color = {"bad": RED, "ok": GREEN, "accept": GREEN}.get(state, MUTED)
            d.text((PAD + 8, y), tag, font=f_tag, fill=tag_color)

            mark = {"bad": "×", "ok": "√", "accept": "√"}.get(state, "")
            if mark:
                d.text((W - PAD - 30, y), mark, font=f_tag, fill=tag_color)

            for i, line in enumerate(lines):
                draw_marked(d, TXT_X, y + 30 * i, line, f_txt,
                            MUTED if state == "none" else INK)
            y += block_h + 14
        y += 18

    d.line([PAD, y, W - PAD, y], fill=GREEN, width=3)
    d.text((PAD, y + 14), FOOTER, font=f_sec, fill=GREEN)

    img.save(OUT)
    print(f"已生成 {OUT}  {img.size}")


if __name__ == "__main__":
    main()
