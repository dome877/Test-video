"""Textures that live inside the 3D scene:
  * strip.png      - the 2x6 photo strip, built from the 4 booth renders (cells), styled like
                     the real Gran Vía strips (logo + couple's names + date in the footer)
  * screen/*.png   - one image per frame for the booth's touchscreen (start button + tap
                     ripple, printing progress with the 4 thumbnails, then a gallery)

    python post/make_textures.py --cells cells/ --out tex/
"""
import argparse
import math
import os
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "blender"))
from anim import RENDER_FRAMES, S2, S3, clamp, in_out, out_cubic  # noqa: E402

FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "fonts")


def F(name, size):
    return ImageFont.truetype(os.path.join(FONT_DIR, name + ".ttf"), size)


def spaced(draw, xy, text, font, fill, tracking=0.0, anchor="mm"):
    """Draw text with letter-spacing (tracking in em)."""
    widths = [draw.textlength(c, font=font) for c in text]
    extra = tracking * font.size
    total = sum(widths) + extra * (len(text) - 1)
    x, y = xy
    if anchor[0] == "m":
        x -= total / 2
    elif anchor[0] == "r":
        x -= total
    for c, w in zip(text, widths):
        draw.text((x, y), c, font=font, fill=fill, anchor="l" + anchor[1])
        x += w + extra
    return total


def logo(draw, cx, cy, scale, color):
    """GRAN VÍA | STUDIO lock-up."""
    f1 = F("PlayfairDisplay-Bold", int(46 * scale))
    f2 = F("Montserrat-Medium", int(19 * scale))
    w1 = draw.textlength("GRAN VÍA", font=f1)
    w2 = sum(draw.textlength(c, font=f2) for c in "STUDIO") + 0.32 * f2.size * 5
    gap = 16 * scale
    total = w1 + gap * 2 + w2
    x0 = cx - total / 2
    draw.text((x0, cy), "GRAN VÍA", font=f1, fill=color, anchor="lm")
    bx = x0 + w1 + gap
    draw.line([(bx, cy - 17 * scale), (bx, cy + 17 * scale)], fill=color, width=max(1, int(2 * scale)))
    spaced(draw, (bx + gap, cy + 1 * scale), "STUDIO", f2, color, tracking=0.32, anchor="lm")


def make_strip(cells_dir, out):
    W, H = 600, 1800
    img = Image.new("RGB", (W, H), (252, 250, 246))
    m, gap = 24, 16
    cw = W - 2 * m
    ch = int(cw * 2 / 3)
    for i in range(4):
        c = Image.open(os.path.join(cells_dir, f"cell_{i + 1}.png")).convert("RGB").resize((cw, ch), Image.LANCZOS)
        img.paste(c, (m, m + i * (ch + gap)))
    d = ImageDraw.Draw(img)
    fy = m + 4 * ch + 3 * gap
    footer_c = (fy + H) / 2
    logo(d, W / 2, footer_c - 58, 1.0, (34, 30, 28))
    d.text((W / 2, footer_c + 8), "Ana i Marko", font=F("Caveat-Medium", 54), fill=(52, 46, 44), anchor="mm")
    d.text((W / 2, footer_c + 64), "12.6.2027.", font=F("Montserrat-SemiBold", 26), fill=(52, 46, 44), anchor="mm")
    img.save(out)
    return [Image.open(os.path.join(cells_dir, f"cell_{i + 1}.png")).convert("RGB") for i in range(4)]


SW, SH = 660, 440
TAP_F = 6
BTN = (330, 257)          # where the bride's mitten lands on the screen


def screen_bg():
    bg = Image.new("RGB", (SW, SH), (28, 25, 24))
    d = ImageDraw.Draw(bg)
    for y in range(SH):
        t = y / SH
        c = (int(36 - 10 * t), int(31 - 8 * t), int(30 - 8 * t))
        d.line([(0, y), (SW, y)], fill=c)
    return bg


def screen_frame(f, cells):
    img = screen_bg()
    d = ImageDraw.Draw(img, "RGBA")
    cream = (243, 236, 226)
    taupe = (196, 172, 156)
    if f < 15:
        logo(d, SW / 2, 52, 0.62, cream)
        press = 0.0
        if TAP_F <= f < TAP_F + 4:
            press = math.sin(math.pi * (f - TAP_F) / 4)
        r = 78 * (1 - 0.08 * press)
        cx, cy = BTN
        d.ellipse([cx - r - 10, cy - r - 10, cx + r + 10, cy + r + 10], outline=(*taupe, 160), width=2)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(int(243 - 30 * press), int(236 - 30 * press), int(226 - 30 * press)))
        # camera glyph
        d.rounded_rectangle([cx - 30, cy - 18, cx + 30, cy + 22], radius=7, fill=(40, 35, 33))
        d.rectangle([cx - 12, cy - 26, cx + 8, cy - 16], fill=(40, 35, 33))
        d.ellipse([cx - 13, cy - 11, cx + 13, cy + 15], fill=cream)
        d.ellipse([cx - 7, cy - 5, cx + 7, cy + 9], fill=(40, 35, 33))
        spaced(d, (SW / 2, 392), "DODIRNI ZA POČETAK", F("Montserrat-SemiBold", 18), cream, tracking=0.28)
        if f >= TAP_F:
            t = (f - TAP_F) / 8
            for k in range(2):
                tt = t - k * 0.25
                if 0 < tt < 1:
                    rr = 30 + 190 * out_cubic(tt)
                    a = int(200 * (1 - tt))
                    d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], outline=(255, 250, 240, a), width=4)
        if f >= 10:
            a = int(255 * clamp((f - 10) / 4))
            ov = Image.new("RGBA", (SW, SH), (28, 25, 24, a))
            img.paste(ov, (0, 0), ov)
            d = ImageDraw.Draw(img, "RGBA")
            d.text((SW / 2, SH / 2), "Pozirajte!", font=F("PlayfairDisplay-Italic", 64), fill=(*cream, a), anchor="mm")
        return img
    if f < S2[0]:
        n = "3" if f < 30 else "2" if f < 45 else "1" if f < 60 else ""
        if n:
            d.text((SW / 2, SH / 2), n, font=F("PlayfairDisplay-Regular", 200), fill=cream, anchor="mm")
        else:
            idx = min(3, (f - 60) // 15)
            th = cells[idx].resize((SW, int(SW * 2 / 3)))
            img.paste(th, (0, (SH - th.size[1]) // 2))
        return img
    if f < S3[0]:
        # printing: the 4 shots as a 2x2 grid + progress bar
        tw, tht = 250, 166
        for i, c in enumerate(cells):
            x = SW / 2 - tw - 8 + (i % 2) * (tw + 16)
            y = 48 + (i // 2) * (tht + 14)
            img.paste(c.resize((tw, tht)), (int(x), int(y)))
        prog = clamp((f - 122) / (168 - 122))
        bx0, bx1, by = 120, SW - 120, 418
        d.rounded_rectangle([bx0, by - 6, bx1, by + 6], radius=6, fill=(70, 62, 58))
        d.rounded_rectangle([bx0, by - 6, bx0 + (bx1 - bx0) * max(prog, 0.02), by + 6], radius=6, fill=cream)
        label = "ISPIS U TIJEKU" if prog < 1 else "GOTOVO ✓"
        fnt = F("Montserrat-SemiBold", 16)
        if prog >= 1:
            spaced(d, (SW / 2, 395), "GOTOVO", fnt, cream, tracking=0.28)
        else:
            spaced(d, (SW / 2, 395), label, fnt, cream, tracking=0.28)
        return img
    # gallery: slow cross-fade through the 4 shots
    period = 36
    i = (f // period) % 4
    j = (i + 1) % 4
    t = clamp(((f % period) - (period - 8)) / 8)
    a = cells[i].resize((SW, int(SW * 2 / 3)))
    b = cells[j].resize((SW, int(SW * 2 / 3)))
    mix = Image.blend(a, b, in_out(t))
    img.paste(mix, (0, (SH - mix.size[1]) // 2))
    logo(ImageDraw.Draw(img), SW / 2, SH - 18, 0.42, cream)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--placeholders", action="store_true", help="flat stand-ins so the scene can be built first")
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "screen"), exist_ok=True)
    if a.placeholders:
        Image.new("RGB", (600, 1800), (240, 235, 228)).save(os.path.join(a.out, "strip.png"))
        for f in range(RENDER_FRAMES):
            screen_bg().save(os.path.join(a.out, "screen", f"screen_{f + 1:04d}.png"))
        print("placeholders written to", a.out)
        return
    cells = make_strip(a.cells, os.path.join(a.out, "strip.png"))
    small = [c.resize((330, 220), Image.LANCZOS) for c in cells]
    for f in range(RENDER_FRAMES):
        screen_frame(f, small).save(os.path.join(a.out, "screen", f"screen_{f + 1:04d}.png"))
    print("textures written to", a.out)


if __name__ == "__main__":
    main()
