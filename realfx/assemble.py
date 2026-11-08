"""Assemble the real-people reel's 435 picture frames (before brand overlays / grade / end card).

Timeline (same beat grid as the animated reel, 30 fps):
  0-14     real booth, the screen gets tapped (ripple drawn on the real screen)
  15-60    real guests (groom's crew) + Blender gold 3D "3-2-1" standing on the real floor; flash 1
  61-75    real couple in orange with props; flash 2
  76-90    real couple under the neon arch; flash 3
  91-111   real kids' group at dusk; flash 4 (held under the white flash)
  112-209  Blender: the strip (made of those 4 real frames) prints out of the real booth panel
  210-299  Blender: the strip lands in the real guestbook; "Živjeli mladenci!" is written onto the page
  300-321  Blender: vintage audio-guestbook phone in the real booth room, handset picked up
  322-389  real bride & groom laughing + voice-message recording overlay
  390-434  real "Better Together" neon

    python realfx/assemble.py --work /tmp/claude-0/real --out /tmp/claude-0/real/frames
"""
import argparse
import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, "..", "assets", "fonts")
W, H = 1080, 1920


def F(name, size):
    return ImageFont.truetype(os.path.join(FONT_DIR, name + ".ttf"), size)


def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def ease(t):
    return t * t * (3 - 2 * t)


def out_cubic(t):
    return 1 - (1 - t) ** 3


def push(im, s, cx, cy):
    """2D push-in: scale about (cx, cy)."""
    if s == 1.0:
        return im
    w, h = W / s, H / s
    x0 = min(max(cx - w * cx / W, 0), W - w)
    y0 = min(max(cy - h * cy / H, 0), H - h)
    return im.transform((W, H), Image.EXTENT, (x0, y0, x0 + w, y0 + h), Image.BICUBIC)


# ---------------------------------------------------------------------------------------
SCREEN = [(388, 618), (732, 618), (732, 868), (395, 872)]
TAP = (520, 745)


def booth_tap(im, f):
    """Ripple + glow on the real touchscreen when it's tapped (frame 5)."""
    if f < 5:
        return im
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    t = (f - 5) / 9
    for k in range(3):
        tt = t - k * 0.18
        if 0 < tt < 1:
            r = 18 + 230 * out_cubic(tt)
            a = int(210 * (1 - tt))
            d.ellipse([TAP[0] - r, TAP[1] - r * 0.92, TAP[0] + r, TAP[1] + r * 0.92], outline=(120, 140, 170, a), width=5)
    if f < 9:
        g = int(90 * (1 - (f - 5) / 4))
        d.ellipse([TAP[0] - 34, TAP[1] - 32, TAP[0] + 34, TAP[1] + 32], fill=(150, 170, 200, g))
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).polygon(SCREEN, fill=255)
    layer.putalpha(Image.composite(layer.getchannel("A"), Image.new("L", (W, H), 0), mask))
    out = im.convert("RGBA")
    out.alpha_composite(layer)
    return out.convert("RGB")


# ---------------------------------------------------------------------------------------
# handwriting on the real guestbook page (page space -> image via the page's own axes)
EX = np.array([0.912, -0.411])                  # along the spiral spine, as seen in the plate
EY = np.array([0.258, 0.966]) * 0.86            # "down" across the page, foreshortened
INK = (36, 46, 96)
HAND = F("Caveat-Medium", 66)
LINES = [("Živjeli", (560, 905), (230, 252)), ("mladenci!", (592, 988), (256, 279))]
HEART_AT = (800, 948)
HEART_T = (282, 293)


def page_layer(text):
    pad = 20
    w = int(HAND.getlength(text)) + 2 * pad
    asc, desc = HAND.getmetrics()
    img = Image.new("L", (w, asc + desc + 2 * pad), 0)
    ImageDraw.Draw(img).text((pad, pad), text, font=HAND, fill=255)
    return img, pad, asc


def warp_onto(base, layer_l, origin, pad, asc, alpha=0.9):
    """Map page-space layer so that (pad, pad+asc) — the baseline start — lands at `origin` in the image."""
    ox, oy = origin
    # inverse affine for PIL: image (x, y) -> page (u, v)
    M = np.array([[EX[0], EY[0]], [EX[1], EY[1]]])
    Mi = np.linalg.inv(M)
    bx, by = pad, pad + asc
    a, b = Mi[0]
    c, d = Mi[1]
    coeffs = (a, b, bx - a * ox - b * oy, c, d, by - c * ox - d * oy)
    warped = layer_l.transform((W, H), Image.AFFINE, coeffs, Image.BICUBIC)
    warped = warped.filter(ImageFilter.GaussianBlur(0.6)).point(lambda v: int(v * alpha))
    ink = Image.new("RGB", (W, H), INK)
    return Image.composite(ink, base, warped)


def handwriting(im, f):
    for text, origin, (a, b) in LINES:
        if f < a:
            continue
        lay, pad, asc = page_layer(text)
        t = clamp((f - a) / (b - a))
        t = clamp(t + 0.03 * math.sin(t * math.pi * 7) * (1 - t) * t * 4)
        reveal = pad - 4 + (lay.width - 2 * pad + 8) * t
        m = Image.new("L", lay.size, 0)
        ImageDraw.Draw(m).rectangle([0, 0, reveal, lay.height], fill=255)
        m = m.filter(ImageFilter.GaussianBlur(3))
        lay = Image.composite(lay, Image.new("L", lay.size, 0), m)
        im = warp_onto(im, lay, origin, pad, asc)
    if f >= HEART_T[0]:
        t = clamp((f - HEART_T[0]) / (HEART_T[1] - HEART_T[0]))
        pts = []
        n = int(80 * t) + 1
        for k in range(n):
            u = 2 * math.pi * k / 79
            x = 16 * math.sin(u) ** 3
            y = -(13 * math.cos(u) - 5 * math.cos(2 * u) - 2 * math.cos(3 * u) - math.cos(4 * u))
            pts.append((40 + x * 1.5, 40 + y * 1.5))
        lay = Image.new("L", (80, 80), 0)
        if len(pts) > 1:
            ImageDraw.Draw(lay).line(pts, fill=255, width=4, joint="curve")
        big = Image.new("L", (W, H), 0)
        big.paste(lay, (HEART_AT[0] - 40, HEART_AT[1] - 40))
        big = big.filter(ImageFilter.GaussianBlur(0.6))
        im = Image.composite(Image.new("RGB", (W, H), (170, 52, 74)), im, big.point(lambda v: int(v * 0.9)))
    return im


# ---------------------------------------------------------------------------------------
def rec_overlay(im, f, start=322):
    """Voice-message recording card over the real couple: red dot, timer, live waveform."""
    t = f - start
    if t < 0:
        return im
    appear = out_cubic(clamp(t / 7))
    card = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(card)
    cw, ch = 560, 118
    cx, cy = W / 2, 1460 + (1 - appear) * 30     # lower third: clear of the headline and their faces
    x0, y0 = cx - cw / 2, cy - ch / 2
    d.rounded_rectangle([x0, y0, x0 + cw, y0 + ch], radius=59, fill=(18, 15, 14, int(150 * appear)))
    pulse = 0.6 + 0.4 * math.sin(t * 0.5)
    d.ellipse([x0 + 34, cy - 13, x0 + 60, cy + 13], fill=(232, 72, 72, int(255 * appear * pulse)))
    secs = t / 30
    fnt = F("Montserrat-SemiBold", 25)
    d.text((x0 + 78, cy - 15), "SNIMANJE", font=fnt, fill=(255, 255, 255, int(235 * appear)), anchor="lm")
    d.text((x0 + 78, cy + 18), f"0:0{int(secs) + 1}", font=F("Montserrat-Regular", 22),
           fill=(236, 214, 200, int(220 * appear)), anchor="lm")
    rng = np.random.default_rng(f)
    bars = 22
    for k in range(bars):
        env = 0.35 + 0.65 * abs(math.sin(t * 0.21 + k * 0.55)) * (0.6 + 0.4 * rng.random())
        if 358 <= f < 380:
            env = min(1.0, env * 1.4)         # the laugh
        bh = 8 + 62 * env
        bx = x0 + 226 + k * 14
        d.rounded_rectangle([bx, cy - bh / 2, bx + 7, cy + bh / 2], radius=3, fill=(255, 246, 238, int(230 * appear)))
    out = im.convert("RGBA")
    out.alpha_composite(card)
    return out.convert("RGB")


# ---------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    C, R = os.path.join(a.work, "clips"), a.work
    os.makedirs(a.out, exist_ok=True)

    def clip(name, i):
        return Image.open(os.path.join(C, name, f"{i:04d}.png")).convert("RGB")

    def rend(name, i, rgba=False):
        im = Image.open(os.path.join(R, "r_" + name, f"frame_{i:04d}.png"))
        im = im.convert("RGBA" if rgba else "RGB")
        return im if im.size == (W, H) else im.resize((W, H), Image.LANCZOS)

    for f in range(435):
        if f < 15:
            im = booth_tap(clip("booth_white", f), f)
            im = push(im, 1.0 + 0.07 * ease(f / 14), 560, 745)
        elif f <= 60:
            im = clip("group", min(f - 15, 45)).convert("RGBA")
            if f < 60:
                im.alpha_composite(rend("numerals", f - 15, rgba=True))
            im = push(im.convert("RGB"), 1.0 + 0.05 * ease((f - 15) / 45), 540, 900)
        elif f <= 75:
            im = push(clip("orange", f - 61), 1.0 + 0.03 * ((f - 61) / 14), 540, 700)
        elif f <= 90:
            im = push(clip("hero", f - 76), 1.0 + 0.03 * ((f - 76) / 14), 540, 800)
        elif f < 112:
            im = push(clip("kids", min(f - 91, 14)), 1.0 + 0.03 * clamp((f - 91) / 14), 540, 1000)
        elif f < 210:
            im = rend("print", f - 112)
        elif f < 300:
            im = handwriting(rend("guestbook", f - 210), f)
            im = push(im, 1.0 + 0.08 * ease((f - 210) / 89), 760, 1250)
        elif f < 322:
            im = rend("phone", f - 300)
        elif f < 390:
            im = rec_overlay(push(clip("couple", f - 322), 1.0 + 0.04 * ((f - 322) / 67), 540, 760), f)
        else:
            im = clip("neon", f - 390)
        im.save(os.path.join(a.out, f"frame_{f:04d}.png"))
        if f % 50 == 0:
            print("assembled", f, flush=True)


if __name__ == "__main__":
    main()
