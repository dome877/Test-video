"""Composite the final reel: grade the Blender frames, add brand typography, the booth
viewfinder/countdown, flash + whip transitions and the end card, then encode with audio.

    python post/composite.py --frames renders/ --audio mix.wav --out reel.mp4 [--alt-audio sfx.wav --alt-out reel_sfx.mp4]
"""
import argparse
import os
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "blender"))
from anim import (COUNTDOWN, END, FLASHES, FPS, RENDER_FRAMES, S1B, S2, S3, S4, S5, TOTAL_FRAMES,  # noqa: E402
                  clamp, in_out, out_back, out_cubic)

W, H = 1080, 1920
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "fonts")
WHITE = (255, 255, 255)
BLUSH = (236, 214, 200)      # the taupe/blush accent from the brand reels, lifted for dark backgrounds
ITALIC = (255, 246, 238)     # over photos the brand reels set the italic line in (warm) white
CREAM = (246, 240, 232)


def F(name, size):
    return ImageFont.truetype(os.path.join(FONT_DIR, name + ".ttf"), size)


def text_layer(text, font, fill, tracking=0.0, shadow=True, pad=40):
    """Render one line of text (with optional soft shadow) to a tight RGBA image."""
    tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    widths = [tmp.textlength(c, font=font) for c in text]
    extra = tracking * font.size
    tw = int(sum(widths) + extra * max(0, len(text) - 1)) + 2
    asc, desc = font.getmetrics()
    th = asc + desc
    img = Image.new("RGBA", (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))

    def draw_into(im, color):
        d = ImageDraw.Draw(im)
        x = pad
        for c, w in zip(text, widths):
            d.text((x, pad), c, font=font, fill=color)
            x += w + extra

    if shadow:
        sh = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw_into(sh, (0, 0, 0, 150))
        sh = sh.filter(ImageFilter.GaussianBlur(9))
        img = Image.alpha_composite(img, sh)
    draw_into(img, (*fill, 255))
    return img


class Line:
    def __init__(self, img, cy, delay=0):
        self.img, self.cy, self.delay = img, cy, delay


def block(start, end, lines, rise=26, fade_in=8, fade_out=6):
    return dict(start=start, end=end, lines=lines, rise=rise, fi=fade_in, fo=fade_out)


def headline(eyebrow, l1, l2, top=300, start=0, end=60, delay=3):
    lines = []
    y = top
    if eyebrow:
        lines.append(Line(text_layer(eyebrow, F("Montserrat-Medium", 27), CREAM, tracking=0.32), y, 0))
        y += 78
    lines.append(Line(text_layer(l1, F("PlayfairDisplay-Regular", 78), WHITE), y, delay))
    lines.append(Line(text_layer(l2, F("PlayfairDisplay-Italic", 78), ITALIC), y + 92, delay * 2))
    return block(start, end, lines)


BLOCKS = [
    # hook: on screen from the very first frame
    headline("GRAN VÍA PHOTOBOOTH", "Što se dogodi kad", "stisneš gumb?", top=250, start=-8, end=58, delay=0),
    headline("01 · ISPIS NA LICU MJESTA", "Fotografija u ruci", "za 12 sekundi", start=S2[0] + 8, end=203),
    headline("02 · KNJIGA GOSTIJU", "Fotka i poruka", "za mladence", start=S3[0] + 4, end=S3[1] - 3),
    headline("03 · AUDIO KNJIGA GOSTIJU", "Glas koji ostaje", "zauvijek", start=S4[0] + 4, end=S4[1] - 3),
    headline(None, "Trenuci koje", "vrijedi čuvati", top=280, start=S5[0] + 15, end=END[0] + 2),
]


def end_card_lines():
    s = END[0]
    L = []
    L.append(Line(text_layer("GRAN VÍA", F("PlayfairDisplay-Bold", 132), WHITE), 640, s + 4))
    L.append(Line(text_layer("STUDIO", F("Montserrat-Regular", 40), WHITE, tracking=0.55), 742, s + 7))
    L.append(Line(text_layer("Rezervirajte", F("PlayfairDisplay-Regular", 96), WHITE), 935, s + 12))
    L.append(Line(text_layer("svoj datum", F("PlayfairDisplay-Italic", 96), WHITE), 1040, s + 15))
    # white pill with the handle, as in the brand's end cards
    pill = Image.new("RGBA", (560, 104), (255, 255, 255, 255))
    t = text_layer("@GRANVIASTUDIO", F("Montserrat-Medium", 34), (20, 18, 17), tracking=0.22, shadow=False, pad=0)
    pill.alpha_composite(t, ((560 - t.width) // 2, (104 - t.height) // 2 + 2))
    L.append(Line(pill, 1215, s + 20))
    L.append(Line(text_layer("Slobodni termini za sezonu 2027.", F("Montserrat-Regular", 29), CREAM, tracking=0.04),
                  1330, s + 24))
    L.append(Line(text_layer("granviastudio.com.hr", F("Montserrat-Medium", 29), BLUSH, tracking=0.08), 1378, s + 26))
    return L


END_LINES = end_card_lines()
COUNT_FONT = F("PlayfairDisplay-Regular", 190)
SMALL = F("Montserrat-SemiBold", 26)


def paste_center(canvas, img, cx, cy, alpha=1.0, scale=1.0):
    if alpha <= 0.002:
        return
    if scale != 1.0:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
    if alpha < 1.0:
        a = img.getchannel("A").point(lambda v: int(v * alpha))
        img = img.copy()
        img.putalpha(a)
    canvas.alpha_composite(img, (int(cx - img.width / 2), int(cy - img.height / 2)))


SCRIM_H = 900
_scrim = np.clip(1 - np.arange(SCRIM_H) / SCRIM_H, 0, 1) ** 1.25 * 0.7
SCRIM = np.zeros((SCRIM_H, W, 4), dtype=np.uint8)
SCRIM[..., 3] = (_scrim[:, None] * 255).astype(np.uint8)
SCRIM = Image.fromarray(SCRIM, "RGBA")


def block_alpha(b, f):
    if not (b["start"] <= f < b["end"] + b["fo"]):
        return 0.0
    return clamp((f - b["start"]) / b["fi"]) * (1 - clamp((f - b["end"]) / b["fo"]))


def draw_scrim(canvas, f):
    """Soft dark gradient behind the headlines so they read on any background (as in the brand reels)."""
    a = max(block_alpha(b, f) for b in BLOCKS)
    if a > 0.01:
        sc = SCRIM.copy()
        sc.putalpha(sc.getchannel("A").point(lambda v: int(v * a)))
        canvas.alpha_composite(sc, (0, 0))


def draw_blocks(canvas, f):
    for b in BLOCKS:
        if not (b["start"] <= f < b["end"] + b["fo"]):
            continue
        out = 1 - clamp((f - b["end"]) / b["fo"])
        for ln in b["lines"]:
            t = clamp((f - b["start"] - ln.delay) / b["fi"])
            a = out_cubic(t) * out
            dy = (1 - out_cubic(t)) * b["rise"]
            paste_center(canvas, ln.img, W / 2, ln.cy + dy, a)


def viewfinder(canvas, f):
    """Booth camera UI over the POV shot: corner marks, 3-2-1 with a progress ring, shot counter."""
    if not (S1B[0] <= f < S1B[1]):
        return
    d = ImageDraw.Draw(canvas, "RGBA")
    a = int(190 * clamp((f - S1B[0]) / 4))
    m, L = 64, 70
    top, bot = 200, H - 330
    for (x, y, sx, sy) in [(m, top, 1, 1), (W - m, top, -1, 1), (m, bot, 1, -1), (W - m, bot, -1, -1)]:
        d.line([(x, y), (x + sx * L, y)], fill=(255, 255, 255, a), width=4)
        d.line([(x, y), (x, y + sy * L)], fill=(255, 255, 255, a), width=4)
    # countdown
    cx, cy, r = W / 2, 1390, 112
    for start, label in COUNTDOWN:
        if start <= f < start + 15:
            t = (f - start) / 15
            pop = out_back(clamp((f - start) / 5), 2.2)
            ring = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            rd = ImageDraw.Draw(ring)
            rd.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(20, 16, 15, 70), outline=(255, 255, 255, 90), width=3)
            rd.arc([cx - r, cy - r, cx + r, cy + r], start=-90, end=-90 + 360 * t, fill=(255, 255, 255, 235), width=7)
            canvas.alpha_composite(ring)
            lay = text_layer(label, COUNT_FONT, WHITE)
            paste_center(canvas, lay, cx, cy - 18, 1.0, 0.7 + 0.3 * pop)
    # shot counter + flash dots
    shots = sum(1 for fl in FLASHES if f >= fl)
    if f >= FLASHES[0]:
        yy = 1420
        for k in range(4):
            x = W / 2 + (k - 1.5) * 34
            filled = k < shots
            d.ellipse([x - 9, yy - 9, x + 9, yy + 9], outline=(255, 255, 255, 220), width=3,
                      fill=(255, 255, 255, 235) if filled else None)
        lab = text_layer(f"FOTO {max(shots, 1)}/4", SMALL, WHITE, tracking=0.3)
        paste_center(canvas, lab, W / 2, yy - 52, 0.95)


def end_card(canvas, f):
    if f < END[0]:
        return
    for ln in END_LINES:
        t = clamp((f - ln.delay) / 9)
        paste_center(canvas, ln.img, W / 2, ln.cy + (1 - out_cubic(t)) * 22, out_cubic(t))


# ---------------------------------------------------------------------------------
# image grading
# ---------------------------------------------------------------------------------
yy_, xx_ = np.mgrid[0:H, 0:W].astype(np.float32)
VIGNETTE = 1 - 0.28 * np.clip(((xx_ - W / 2) / (W * 0.62)) ** 2 + ((yy_ - H / 2) / (H * 0.62)) ** 2, 0, 1) ** 1.4
VIGNETTE = VIGNETTE[..., None]
rng = np.random.default_rng(7)


def load_frame(frames_dir, f):
    im = Image.open(os.path.join(frames_dir, f"frame_{f:04d}.png")).convert("RGB")
    if im.size != (W, H):
        im = im.resize((W, H), Image.LANCZOS)
    return im


def bloom(im, strength=0.36):
    small = im.resize((W // 4, H // 4), Image.BILINEAR)
    a = np.asarray(small).astype(np.float32) / 255
    lum = a.max(axis=2, keepdims=True)
    hi = a * np.clip((lum - 0.82) / 0.18, 0, 1)
    hi_img = Image.fromarray((hi * 255).astype(np.uint8))
    b1 = np.asarray(hi_img.filter(ImageFilter.GaussianBlur(6))).astype(np.float32)
    b2 = np.asarray(hi_img.filter(ImageFilter.GaussianBlur(20))).astype(np.float32)
    glow = Image.fromarray(np.clip(b1 * 0.6 + b2 * 0.8, 0, 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    return np.asarray(glow).astype(np.float32) / 255 * strength


def directional_blur(arr, amount):
    if amount < 1:
        return arr
    n = 14
    acc = np.zeros_like(arr)
    for k in range(n):
        sh = int(round((k / (n - 1) - 0.5) * amount))
        acc += np.roll(arr, sh, axis=0)
    return acc / n


WHIP = {203: 12, 204: 26, 205: 45, 206: 70, 207: 100, 208: 135, 209: 170, 210: 170, 211: 110, 212: 55, 213: 20}


def flash_alpha(f):
    a = 0.0
    for fl in FLASHES[:-1]:
        d = f - fl
        if 0 <= d < 5:
            a = max(a, [0.9, 0.55, 0.28, 0.12, 0.04][d])
    d = f - FLASHES[-1]
    if 0 <= d < 7:
        a = max(a, 0.93)
    if S2[0] <= f < S2[0] + 10:
        a = max(a, 0.93 * (1 - in_out((f - S2[0]) / 10)))
    return a


def graded(frames_dir, f, end_bg):
    if f < RENDER_FRAMES:
        im = load_frame(frames_dir, f)
        arr = np.asarray(im).astype(np.float32) / 255
        arr = arr + bloom(im)
    else:
        t = (f - END[0]) / (END[1] - END[0])
        z = 1.0 + 0.06 * t
        bw, bh = int(W / z), int(H / z)
        crop = end_bg.crop(((W - bw) // 2, (H - bh) // 2, (W - bw) // 2 + bw, (H - bh) // 2 + bh)).resize((W, H))
        arr = np.asarray(crop).astype(np.float32) / 255
    if f in WHIP:
        arr = directional_blur(arr, WHIP[f])
    # warm, filmic touch
    arr = arr * np.array([1.02, 1.0, 0.97], dtype=np.float32)
    arr = arr * VIGNETTE
    fa = flash_alpha(f)
    if fa > 0:
        arr = arr * (1 - fa) + np.array([1.0, 0.99, 0.97], dtype=np.float32) * fa
    grain = rng.standard_normal((H // 2, W // 2)).astype(np.float32)
    grain = np.asarray(Image.fromarray(((grain * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)).resize((W, H),
                                                                                                       Image.BILINEAR))
    arr = arr + ((grain.astype(np.float32) / 255 - 0.5) * 0.045)[..., None]
    return np.clip(arr, 0, 1)


def make_end_bg(frames_dir):
    im = load_frame(frames_dir, RENDER_FRAMES - 1)
    im = im.filter(ImageFilter.GaussianBlur(16))
    arr = np.asarray(im).astype(np.float32) * 0.42
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def render_frame(frames_dir, f, end_bg):
    arr = graded(frames_dir, f, end_bg)
    if END[0] <= f < END[0] + 7:
        # dissolve: last 3D frame -> blurred end-card background
        t = in_out((f - END[0]) / 7)
        last = np.asarray(load_frame(frames_dir, RENDER_FRAMES - 1)).astype(np.float32) / 255
        arr = arr * t + last * VIGNETTE * (1 - t)
    canvas = Image.fromarray((arr * 255).astype(np.uint8)).convert("RGBA")
    draw_scrim(canvas, f)
    viewfinder(canvas, f)
    draw_blocks(canvas, f)
    end_card(canvas, f)
    return canvas.convert("RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alt-audio", default="")
    ap.add_argument("--alt-out", default="")
    ap.add_argument("--stills", default="", help="optional: also save these frames as PNG (comma list)")
    ap.add_argument("--stills-dir", default="")
    ap.add_argument("--limit", type=int, default=TOTAL_FRAMES)
    a = ap.parse_args()
    end_bg = make_end_bg(a.frames)
    stills = {int(x) for x in a.stills.split(",") if x.strip()}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
           "-i", "-", "-i", a.audio, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "slow", "-crf", "19",
           "-profile:v", "high", "-level", "4.2", "-pix_fmt", "yuv420p", "-colorspace", "bt709", "-color_primaries",
           "bt709", "-color_trc", "bt709", "-c:a", "aac", "-b:a", "256k", "-ar", "48000", "-shortest",
           "-movflags", "+faststart", a.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(min(a.limit, TOTAL_FRAMES)):
        im = render_frame(a.frames, f, end_bg)
        if f in stills and a.stills_dir:
            os.makedirs(a.stills_dir, exist_ok=True)
            im.save(os.path.join(a.stills_dir, f"still_{f:04d}.png"))
        proc.stdin.write(im.tobytes())
        if f % 50 == 0:
            print("composited", f, flush=True)
    proc.stdin.close()
    proc.wait()
    if a.alt_audio and a.alt_out:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", a.out, "-i", a.alt_audio, "-map", "0:v", "-map", "1:a",
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart",
                        a.alt_out], check=True)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
