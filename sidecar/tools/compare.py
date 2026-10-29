"""Montage of preview PNGs: compare.py OUT label1=path1 label2=path2 ... (each tile scaled to the same height)."""
import sys
from PIL import Image, ImageDraw
out = sys.argv[1]
items = [a.split("=", 1) for a in sys.argv[2:]]
H = 560
tiles = []
for lab, p in items:
    im = Image.open(p).convert("RGB")
    tiles.append((lab, im.resize((int(im.width * H / im.height), H))))
W = sum(t.width for _, t in tiles) + 6 * (len(tiles) - 1)
sheet = Image.new("RGB", (W, H + 26), (18, 18, 20))
d = ImageDraw.Draw(sheet)
x = 0
for lab, t in tiles:
    sheet.paste(t, (x, 26))
    d.text((x + 4, 7), lab, fill=(235, 235, 235))
    x += t.width + 6
sheet.save(out)
print(out, sheet.size)
