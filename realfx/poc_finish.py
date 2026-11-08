"""Finish the real-people test shot: grade + brand headlines + flash, a 5 s soundtrack, encode.

    python realfx/poc_finish.py --frames frames/ --out test.mp4
"""
import argparse
import os
import subprocess
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("post", "audio", "blender"):
    sys.path.insert(0, os.path.join(HERE, "..", sub))
import composite as C  # noqa: E402
import soundtrack as S  # noqa: E402

N = 150
FLASH = 60
FPS = 30

C.BLOCKS = [
    C.headline("GRAN VÍA PHOTOBOOTH", "Što se dogodi kad", "stisneš gumb?", top=200, start=-8, end=56, delay=0),
    C.headline("01 · ISPIS NA LICU MJESTA", "Fotografija u ruci", "za 12 sekundi", top=250, start=90, end=170),
]
rng = np.random.default_rng(3)


def flash_alpha(f):
    d = f - FLASH
    if d < 0:
        return 0.0
    return float(np.clip(0.96 * np.exp(-max(0, d - 2) / 3.2), 0, 0.96))


def frame(frames_dir, f):
    im = Image.open(os.path.join(frames_dir, f"frame_{f:04d}.png")).convert("RGB").resize((C.W, C.H), Image.LANCZOS)
    arr = np.asarray(im).astype(np.float32) / 255
    arr = arr + C.bloom(im, 0.4)
    arr = arr * np.array([1.02, 1.0, 0.97], np.float32) * C.VIGNETTE
    fa = flash_alpha(f)
    arr = arr * (1 - fa) + fa
    g = rng.standard_normal((C.H // 2, C.W // 2)).astype(np.float32)
    g = np.asarray(Image.fromarray(((g * 0.5 + 0.5).clip(0, 1) * 255).astype(np.uint8)).resize((C.W, C.H)))
    arr = arr + ((g.astype(np.float32) / 255 - 0.5) * 0.04)[..., None]
    canvas = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).convert("RGBA")
    C.draw_scrim(canvas, f)
    C.draw_blocks(canvas, f)
    return canvas.convert("RGB")


def soundtrack(path):
    S.DUR = N / FPS
    S.N = int(S.DUR * S.SR)
    T = lambda fr: fr / FPS  # noqa: E731
    music, sfx = S.Bus(), S.Bus()
    music.add(0.0, S.pad([S.note(n) for n in ["G3", "B3", "D4", "F#4", "A4"]], 2.1), 0.45)
    for b in range(4):
        music.add(b * 0.5, S.shaker(0.8), 0.6, pan=0.3)
    music.add(0.95, S.sfx_whoosh(1.05, up=True, peak_at=0.97), 0.35)
    t = 2.0
    while t < S.DUR - 0.05:
        bi = int(round((t - 2.0) / 0.5))
        music.add(t, S.lp(S.kick(), 2500), 0.62)
        if bi % 2 == 1:
            music.add(t, S.clap(), 0.5)
            music.add(t + 0.004, S.snap(), 0.35, pan=0.15)
        for k in range(4):
            music.add(t + k * 0.125, S.shaker(1.0 if k == 2 else 0.55), 0.7, pan=0.35)
        t += 0.5
    for t0, root, tones in [(2.0, "G2", ["G3", "B3", "D4", "F#4"]), (4.0, "E2", ["E3", "G3", "B3", "D4"])]:
        for off, ln, mul in [(0.0, 0.42, 1), (0.75, 0.2, 1), (1.0, 0.42, 2), (1.5, 0.3, 1)]:
            if t0 + off < S.DUR:
                music.add(t0 + off, S.bass(S.note(root) * mul, ln), 1.0)
        for off in np.arange(0.25, 2.0, 0.5):
            for k, n in enumerate(tones):
                music.add(t0 + off, S.epiano(S.note(n), 0.32, 0.9), 0.5, pan=-0.25 + 0.15 * k)
    for b, n, ln in S.MELODY[:8]:
        music.add(b * 0.5, S.bell(S.note(n), min(1.4, ln * 0.5 + 0.6)), 0.85, pan=0.1)
    for fr in (15, 30, 45):
        sfx.add(T(fr), S.sfx_beep(988), 0.8)
    sfx.add(T(FLASH) - 0.01, S.sfx_shutter(), 1.1)
    sfx.add(T(FLASH + 6), S.sfx_whoosh(0.6, up=False, peak_at=0.6), 0.5)
    sfx.add(T(FLASH + 12), S.sfx_flutter(0.9), 0.9)
    sfx.add(T(FLASH + 52), S.sfx_land(), 1.0)
    mix = music.stereo() * 0.62 + sfx.stereo()
    fade = np.clip((S.DUR - np.arange(S.N) / S.SR) / 0.35, 0, 1)
    S.write_wav(path, S.master(mix * fade[:, None]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    wav = os.path.splitext(a.out)[0] + ".wav"
    soundtrack(wav)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{C.W}x{C.H}", "-r", str(FPS),
           "-i", "-", "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "slow", "-crf", "19",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", a.out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(N):
        im = frame(a.frames, f)
        if f in (0, 40, 56, 66, 100, 149):
            im.resize((540, 960)).save(os.path.splitext(a.out)[0] + f"_still_{f:03d}.jpg", quality=90)
        p.stdin.write(im.tobytes())
    p.stdin.close()
    p.wait()
    os.remove(wav)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
