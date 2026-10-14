import win32com.client
import os
import sys
import time

src = os.path.abspath(sys.argv[1])
out_dir = os.path.abspath(sys.argv[2])
os.makedirs(out_dir, exist_ok=True)

app = win32com.client.Dispatch("PowerPoint.Application")
app.Visible = True
pres = app.Presentations.Open(src, WithWindow=False)
for i, slide in enumerate(pres.Slides, start=1):
    path = os.path.join(out_dir, f"slide{i}.png")
    slide.Export(path, "PNG", 1920, 1080)
    print("exported", path)
pres.Close()
app.Quit()
