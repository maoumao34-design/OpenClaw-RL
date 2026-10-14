# -*- coding: utf-8 -*-
from PIL import Image
p = Image.open(r"D:\MAO\Claude\openclaw-rl\report\_mc7_page.png")
# crop candidate full Table 1
crop = p.crop((160, 145, 1540, 545))
crop.save(r"D:\MAO\Claude\openclaw-rl\report\_t1_full_crop.png")
print(crop.size)
