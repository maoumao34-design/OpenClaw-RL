# -*- coding: utf-8 -*-
from PIL import Image
p = Image.open(r"D:\MAO\Claude\openclaw-rl\report\_mc7_page.png")
for name, box in [
    ("a", (180, 200, 1540, 590)),
    ("b", (200, 215, 1520, 575)),
    ("c", (160, 190, 1550, 600)),
]:
    crop = p.crop(box)
    crop.save(fr"D:\MAO\Claude\openclaw-rl\report\_t1_{name}.png")
    print(name, crop.size)
