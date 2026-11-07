"""Tiny animation toolkit: easing curves, tweens and the reel's beat grid.

All timing in this project is expressed in 0-based frame indices `f` at 30 fps.
Blender scene frame = f + 1.  Music runs at 120 BPM, so one beat = 15 frames.
"""
import math

FPS = 30
BEAT = 15  # frames per beat at 120 BPM

# ---- shot table (0-based frames, end exclusive) ---------------------------
S1A = (0, 15)      # finger taps the booth screen
S1B = (15, 112)    # booth POV: 3-2-1, four flashes
S2 = (112, 210)    # strip prints out of the booth
S3 = (210, 300)    # strip lands in the guestbook, message is written
S4 = (300, 390)    # audio guestbook phone
S5 = (390, 435)    # pull-back reveal of the whole corner
END = (435, 510)   # 2D end card (composited, not rendered)
RENDER_FRAMES = S5[1]
TOTAL_FRAMES = END[1]

FLASHES = [60, 75, 90, 105]
COUNTDOWN = [(15, "3"), (30, "2"), (45, "1")]


def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def lerp(a, b, t):
    if isinstance(a, (tuple, list)):
        return tuple(lerp(x, y, t) for x, y in zip(a, b))
    return a + (b - a) * t


def linear(t):
    return t


def smooth(t):
    return t * t * (3 - 2 * t)


def in_out(t):  # cubic
    return 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def in_out_expo(t):
    if t <= 0:
        return 0.0
    if t >= 1:
        return 1.0
    return 2 ** (20 * t - 10) / 2 if t < 0.5 else (2 - 2 ** (-20 * t + 10)) / 2


def out_cubic(t):
    return 1 - (1 - t) ** 3


def in_cubic(t):
    return t ** 3


def in_quad(t):
    return t * t


def out_back(t, s=1.9):
    t -= 1
    return t * t * ((s + 1) * t + s) + 1


def out_back_soft(t):
    return out_back(t, 1.2)


def out_elastic(t):
    if t <= 0:
        return 0.0
    if t >= 1:
        return 1.0
    return 2 ** (-10 * t) * math.sin((t * 10 - 0.75) * (2 * math.pi) / 3.2) + 1


def out_bounce(t):
    n1, d1 = 7.5625, 2.75
    if t < 1 / d1:
        return n1 * t * t
    if t < 2 / d1:
        t -= 1.5 / d1
        return n1 * t * t + 0.75
    if t < 2.5 / d1:
        t -= 2.25 / d1
        return n1 * t * t + 0.9375
    t -= 2.625 / d1
    return n1 * t * t + 0.984375


def tween(f, keys, default_ease=in_out):
    """keys: [(frame, value[, ease])]. Ease of key i shapes the move INTO key i."""
    if f <= keys[0][0]:
        return keys[0][1]
    for i in range(1, len(keys)):
        k0, k1 = keys[i - 1], keys[i]
        if f <= k1[0]:
            ease = k1[2] if len(k1) > 2 else default_ease
            span = k1[0] - k0[0]
            t = 1.0 if span <= 0 else (f - k0[0]) / span
            return lerp(k0[1], k1[1], ease(clamp(t)))
    return keys[-1][1]


def wobble(f, seed=0.0, amp=1.0, speed=1.0):
    """Smooth pseudo-random drift built from incommensurate sines (handheld feel)."""
    t = f / FPS * speed
    return amp * (0.55 * math.sin(t * 1.13 + seed * 3.1)
                  + 0.3 * math.sin(t * 2.71 + seed * 1.7)
                  + 0.15 * math.sin(t * 5.37 + seed * 0.3))


def pulse(f, at, attack=0, decay=4):
    """Flash-style envelope: 1 at `at`, exponential falloff."""
    d = f - at
    if d < -attack:
        return 0.0
    if d < 0:
        return 1 + d / max(attack, 1)
    return math.exp(-d / decay)


def in_range(f, rng):
    return rng[0] <= f < rng[1]
