# -*- coding: utf-8 -*-
"""Inspect MetaClaw PDF page 7 Table 1 bounds."""
import glob
import subprocess
from pathlib import Path
from PIL import Image

PDF = Path(r"D:/MAO/Paper/MetaClaw/MetaClaw_2603.17187.pdf")
TMP = Path(r"D:/MAO/Claude/openclaw-rl/report/_mc7")
HERE = Path(r"D:/MAO/Claude/openclaw-rl/report")

subprocess.run([
    "pdftoppm.exe", "-f", "7", "-l", "7", "-r", "200", "-png",
    str(PDF), str(TMP)
], check=True)
hits = glob.glob(str(HERE / "_mc7-*.png"))
print("hits", hits)
im = Image.open(hits[0])
print("page size", im.size)
im.save(HERE / "_mc7_page.png")
print("saved _mc7_page.png")
