"""Original soundtrack for the reel, synthesized from scratch with numpy (no samples).

120 BPM so every beat is exactly 15 video frames. The countdown beeps sit on beats,
the music drops on the first camera flash, and every sound effect is placed on the
frame where it happens in the animation.

    python audio/soundtrack.py --out mix.wav [--sfx-only sfx.wav]
"""
import argparse
import os
import sys
import wave

import numpy as np
from scipy.signal import butter, lfilter, sosfilt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "blender"))
from anim import FLASHES, FPS, TOTAL_FRAMES  # noqa: E402

SR = 48000
DUR = TOTAL_FRAMES / FPS          # 17.0 s
N = int(DUR * SR)
BEAT = 0.5
rng = np.random.default_rng(877)


def T(frame):
    return frame / FPS


def note(name):
    names = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7,
             "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}
    p, o = name[:-1], int(name[-1])
    midi = 12 * (o + 1) + names[p]
    return 440.0 * 2 ** ((midi - 69) / 12)


class Bus:
    def __init__(self):
        self.l = np.zeros(N)
        self.r = np.zeros(N)

    def add(self, t, sig, gain=1.0, pan=0.0):
        i = int(t * SR)
        if i >= N or i + len(sig) <= 0:
            return
        if i < 0:
            sig = sig[-i:]
            i = 0
        sig = sig[: N - i]
        gl = gain * np.sqrt(0.5 * (1 - pan))
        gr = gain * np.sqrt(0.5 * (1 + pan))
        self.l[i:i + len(sig)] += sig * gl
        self.r[i:i + len(sig)] += sig * gr

    def stereo(self):
        return np.stack([self.l, self.r], axis=1)


def tt(dur):
    return np.arange(int(dur * SR)) / SR


def env(dur, a=0.005, d=0.1, s=0.0, r=0.05, hold=0.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    e = np.ones(n)
    e = np.where(t < a, t / max(a, 1e-6), e)
    dec = (t >= a) & (t < a + d)
    e = np.where(dec, 1 - (1 - s) * (t - a) / max(d, 1e-6), e)
    e = np.where(t >= a + d, s, e)
    if s > 0:
        rel_start = dur - r
        e = np.where(t > rel_start, e * np.clip((dur - t) / max(r, 1e-6), 0, 1), e)
    return e


def expdec(dur, tau):
    return np.exp(-tt(dur) / tau)


def bp(x, lo, hi, order=2):
    sos = butter(order, [lo / (SR / 2), min(hi / (SR / 2), 0.99)], btype="band", output="sos")
    return sosfilt(sos, x)


def lp(x, fc, order=2):
    sos = butter(order, min(fc / (SR / 2), 0.99), btype="low", output="sos")
    return sosfilt(sos, x)


def hp(x, fc, order=2):
    sos = butter(order, fc / (SR / 2), btype="high", output="sos")
    return sosfilt(sos, x)


def noise(dur):
    return rng.standard_normal(int(dur * SR))


# ---------------------------------------------------------------------------------
# instruments
# ---------------------------------------------------------------------------------
def kick():
    d = 0.42
    t = tt(d)
    f = 46 + 120 * np.exp(-t / 0.03)
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) * np.exp(-t / 0.16)
    click = hp(noise(0.006), 2500) * 0.25
    s[: len(click)] += click
    return np.tanh(1.6 * s) * 0.9


def clap():
    d = 0.25
    s = np.zeros(int(d * SR))
    for k, off in enumerate([0.0, 0.009, 0.018]):
        i = int(off * SR)
        L = len(s) - i
        n = bp(rng.standard_normal(L), 900, 3200) * np.exp(-np.arange(L) / SR / (0.012 if k < 2 else 0.07))
        s[i:] += n
    return s * 0.5


def snap():
    d = 0.12
    s = bp(noise(d), 1800, 6000) * np.exp(-tt(d) / 0.018)
    tone = np.sin(2 * np.pi * 1450 * tt(d)) * np.exp(-tt(d) / 0.01) * 0.4
    return (s + tone) * 0.55


def shaker(accent=1.0):
    d = 0.07
    e = np.minimum(tt(d) / 0.01, 1) * np.exp(-tt(d) / 0.025)
    return hp(noise(d), 6500) * e * 0.18 * accent


def epiano(freq, dur, vel=1.0):
    t = tt(dur)
    idx = 1.8 * np.exp(-t / 0.25) + 0.25
    mod = np.sin(2 * np.pi * freq * t) * idx
    car = np.sin(2 * np.pi * freq * t + mod)
    bell = np.sin(2 * np.pi * freq * 4.0 * t) * np.exp(-t / 0.05) * 0.15
    e = np.minimum(t / 0.004, 1) * np.exp(-t / 0.9)
    e *= np.clip((dur - t) / 0.06, 0, 1)
    return (car + bell) * e * 0.22 * vel


def bell(freq, dur=1.2, vel=1.0):
    t = tt(dur)
    mod = np.sin(2 * np.pi * freq * 3.5 * t) * (2.2 * np.exp(-t / 0.15))
    car = np.sin(2 * np.pi * freq * t + mod)
    e = np.minimum(t / 0.002, 1) * np.exp(-t / 0.45)
    return car * e * 0.2 * vel


def bass(freq, dur):
    t = tt(dur)
    s = np.sin(2 * np.pi * freq * t) + 0.35 * np.sin(2 * np.pi * 2 * freq * t) + 0.1 * np.sin(2 * np.pi * 3 * freq * t)
    e = np.minimum(t / 0.006, 1) * np.exp(-t / 0.5) * np.clip((dur - t) / 0.03, 0, 1)
    return np.tanh(1.3 * s * e) * 0.42


def pad(freqs, dur):
    t = tt(dur)
    s = np.zeros_like(t)
    for f in freqs:
        for det in (-0.12, 0.0, 0.13):
            ff = f * 2 ** (det / 12)
            s += 2 * ((ff * t + rng.random()) % 1.0) - 1
    s = lp(s / (3 * len(freqs)), 1400)
    e = np.minimum(t / 0.35, 1) * np.clip((dur - t) / 0.4, 0, 1)
    return s * e * 0.5


# ---------------------------------------------------------------------------------
# sound effects
# ---------------------------------------------------------------------------------
def sfx_tap():
    tock = (np.sin(2 * np.pi * 1900 * tt(0.03)) + 0.6 * np.sin(2 * np.pi * 2700 * tt(0.03))) * expdec(0.03, 0.006)
    thump = np.sin(2 * np.pi * 170 * tt(0.06)) * expdec(0.06, 0.015)
    t = tt(0.09)
    f = 1250 + 600 * t / 0.09
    blip = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.minimum(t / 0.004, 1) * np.exp(-t / 0.035)
    out = np.zeros(int(0.12 * SR))
    out[: len(tock)] += tock * 0.5
    out[: len(thump)] += thump * 0.6
    out[int(0.02 * SR): int(0.02 * SR) + len(blip)] += blip * 0.45
    return out


def sfx_beep(freq=988, dur=0.12):
    t = tt(dur)
    s = np.sin(2 * np.pi * freq * t) + 0.2 * np.sin(2 * np.pi * 2 * freq * t)
    return s * np.minimum(t / 0.004, 1) * np.clip((dur - t) / 0.03, 0, 1) * 0.42


def sfx_shutter():
    d = 0.32
    out = np.zeros(int(d * SR))
    for off, g, lo, hi in [(0.0, 1.0, 1500, 7000), (0.045, 0.7, 900, 4500)]:
        c = bp(noise(0.03), lo, hi) * expdec(0.03, 0.005)
        i = int(off * SR)
        out[i:i + len(c)] += c * g
    t = tt(0.25)
    pop = np.sin(2 * np.pi * (90 + 60 * np.exp(-t / 0.02)) * t) * np.exp(-t / 0.05) * 0.5
    out[: len(pop)] += pop
    shimmer = hp(noise(0.25), 7000) * np.exp(-tt(0.25) / 0.08) * 0.25
    out[: len(shimmer)] += shimmer
    return out * 0.75


def sfx_pop(f0=500, f1=1100, dur=0.09):
    t = tt(dur)
    f = f0 + (f1 - f0) * (t / dur) ** 0.5
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.minimum(t / 0.003, 1) * np.exp(-t / 0.03) * 0.4


def sfx_patter():
    d = 0.06
    return lp(noise(d), 900) * np.exp(-tt(d) / 0.012) * 0.6


def sfx_printer(dur):
    t = tt(dur)
    steps = 6
    am = 0.55 + 0.45 * (0.5 - 0.5 * np.cos(2 * np.pi * steps * t / dur))
    hum = (np.sin(2 * np.pi * 96 * t) + 0.5 * np.sin(2 * np.pi * 192 * t + 1) + 0.3 * np.sin(2 * np.pi * 288 * t)
           + 0.2 * np.sign(np.sin(2 * np.pi * 48 * t)))
    paper = bp(noise(dur), 2200, 6500) * 0.5
    e = np.minimum(t / 0.05, 1) * np.clip((dur - t) / 0.05, 0, 1)
    return (lp(hum, 900) * 0.45 + paper) * am * e * 0.35


def sfx_cut():
    c = bp(noise(0.04), 1200, 5000) * expdec(0.04, 0.006)
    k = np.sin(2 * np.pi * 220 * tt(0.05)) * expdec(0.05, 0.012) * 0.6
    out = np.zeros(int(0.06 * SR))
    out[: len(c)] += c
    out[: len(k)] += k
    return out * 0.6


def sfx_flutter(dur):
    am = np.abs(np.sin(2 * np.pi * 23 * tt(dur))) ** 3
    return bp(noise(dur), 1200, 5000) * am * np.minimum(tt(dur) / 0.05, 1) * np.clip((dur - tt(dur)) / 0.08, 0, 1) * 0.2


def sfx_whoosh(dur, up=True, peak_at=0.8):
    t = tt(dur)
    x = noise(dur)
    out = np.zeros_like(x)
    seg = int(0.02 * SR)
    for i in range(0, len(x), seg):
        u = i / len(x)
        fc = 300 * (12 ** u) if up else 3600 / (12 ** u)
        out[i:i + seg] = bp(x[max(0, i - 2000):i + seg], fc * 0.6, fc * 1.6)[-len(x[i:i + seg]):]
    e = np.where(t / dur < peak_at, (t / dur / peak_at) ** 2, np.clip((1 - t / dur) / (1 - peak_at), 0, 1))
    return out * e * 0.5


def sfx_land():
    th = lp(noise(0.15), 400) * expdec(0.15, 0.03)
    slap = bp(noise(0.03), 1500, 6000) * expdec(0.03, 0.006) * 0.5
    out = np.zeros(int(0.15 * SR))
    out += th * 0.9
    out[: len(slap)] += slap
    return out * 0.7


def sfx_scribble(dur, seed):
    r = np.random.default_rng(seed)
    t = tt(dur)
    strokes = np.abs(np.sin(2 * np.pi * (8.5 + 1.5 * np.sin(2 * np.pi * 0.7 * t)) * t + r.random() * 6)) ** 1.5
    jitter = lp(r.standard_normal(len(t)), 30)
    jitter = 0.6 + 0.4 * jitter / (np.abs(jitter).max() + 1e-9)
    s = bp(r.standard_normal(len(t)), 2500, 8000) * strokes * jitter
    e = np.minimum(t / 0.02, 1) * np.clip((dur - t) / 0.03, 0, 1)
    return s * e * 0.28


def sfx_pickup():
    out = np.zeros(int(0.35 * SR))
    for off, g in [(0.0, 1.0), (0.06, 0.6), (0.11, 0.35)]:
        c = bp(noise(0.03), 600, 3500) * expdec(0.03, 0.007)
        k = np.sin(2 * np.pi * 320 * tt(0.04)) * expdec(0.04, 0.01)
        i = int(off * SR)
        out[i:i + len(c)] += c * g
        out[i:i + len(k)] += k * g * 0.5
    ting = np.sin(2 * np.pi * 2350 * tt(0.3)) * expdec(0.3, 0.07) * 0.08
    out[: len(ting)] += ting
    return out * 0.7


VOWELS = {"a": (820, 1250, 2700), "e": (520, 1950, 2700), "i": (330, 2350, 3000),
          "o": (520, 900, 2600), "u": (360, 820, 2500)}


def voice_syllable(vowel, dur, f0_start, f0_end, consonant=True, breathy=0.05):
    t = tt(dur)
    f0 = f0_start + (f0_end - f0_start) * t / dur
    ph = np.cumsum(f0) / SR
    src = (ph % 1.0)
    src = (1 - src) ** 3 - 0.25                    # glottal-ish pulse
    src += breathy * rng.standard_normal(len(t))
    out = np.zeros_like(t)
    for k, (fm, g) in enumerate(zip(VOWELS[vowel], (1.0, 0.6, 0.25))):
        bw = 80 + 40 * k
        out += bp(src, max(60, fm - bw), fm + bw) * g
    e = np.minimum(t / 0.012, 1) * np.clip((dur - t) / 0.025, 0, 1)
    out *= e
    if consonant:
        c = hp(noise(0.018), 3000) * expdec(0.018, 0.006) * 0.4
        out[: len(c)] += c
    return out


def sfx_babble(start, end):
    """Cute, wordless 'voice message' (Animal-Crossing-style syllables) through a phone filter."""
    r = np.random.default_rng(31)
    pieces = []
    t = 0.0
    total = end - start
    while t < total - 0.1:
        dur = r.uniform(0.07, 0.115)
        v = r.choice(list("aeiouaeao"))
        base = 300 + 40 * np.sin(2 * np.pi * t / 1.3) + r.uniform(-20, 30)
        pieces.append((t, voice_syllable(v, dur, base * 1.03, base * 0.97, consonant=r.random() < 0.7)))
        t += dur + (r.uniform(0.09, 0.16) if r.random() < 0.15 else r.uniform(0.0, 0.025))
    buf = np.zeros(int((total + 0.2) * SR))
    for t0, s in pieces:
        i = int(t0 * SR)
        buf[i:i + len(s)] += s
    return buf


def sfx_laugh():
    buf = np.zeros(int(0.6 * SR))
    for k in range(4):
        s = voice_syllable("a", 0.085, 420 - 25 * k, 380 - 25 * k, consonant=True, breathy=0.25)
        i = int(k * 0.12 * SR)
        buf[i:i + len(s)] += s * (1 - 0.15 * k)
    return buf


def phone_filter(x):
    return np.tanh(2.0 * bp(x, 350, 3400, order=3)) * 0.6


# ---------------------------------------------------------------------------------
# arrangement
# ---------------------------------------------------------------------------------
CHORDS = [  # (start_s, root for bass, chord tones for keys/pad)
    (0.0, "G2", ["G3", "B3", "D4", "F#4", "A4"]),
    (2.0, "G2", ["G3", "B3", "D4", "F#4"]),
    (4.0, "E2", ["E3", "G3", "B3", "D4"]),
    (6.0, "C2", ["C3", "E3", "G3", "B3"]),
    (8.0, "D2", ["D3", "F#3", "A3", "C4"]),
    (10.0, "G2", ["G3", "B3", "D4", "F#4"]),
    (12.0, "E2", ["E3", "G3", "B3", "D4"]),
    (14.0, "C2", ["C3", "E3", "G3", "B3"]),
    (14.5, "D2", ["D3", "F#3", "A3", "C4"]),
    (15.5, "G2", ["G3", "B3", "D4", "F#4", "A4"]),
]
MELODY = [  # (beat, note, length_beats) — a small hook that answers the flashes
    (4.0, "D5", 0.5), (4.5, "B4", 0.5), (5.0, "D5", 0.5), (5.5, "G5", 1.0), (6.5, "F#5", 0.5), (7.0, "D5", 1.0),
    (8.0, "E5", 0.5), (8.5, "B4", 0.5), (9.0, "E5", 0.5), (9.5, "G5", 1.0), (10.5, "F#5", 0.5), (11.0, "E5", 1.0),
    (12.0, "E5", 0.5), (12.5, "C5", 0.5), (13.0, "E5", 0.5), (13.5, "G5", 0.5), (14.0, "B5", 1.0), (15.0, "A5", 1.0),
    (16.0, "A5", 0.5), (16.5, "F#5", 0.5), (17.0, "A5", 0.5), (17.5, "D6", 1.0), (18.5, "C6", 0.5), (19.0, "A5", 1.0),
    (20.0, "D5", 0.5), (20.5, "B4", 0.5), (21.0, "D5", 0.5), (21.5, "G5", 1.0), (22.5, "A5", 0.5), (23.0, "B5", 1.0),
    (24.0, "B5", 0.5), (24.5, "G5", 0.5), (25.0, "E5", 0.5), (25.5, "G5", 1.0), (26.5, "F#5", 0.5), (27.0, "E5", 1.0),
    (29.0, "G5", 0.5), (29.5, "B5", 0.5), (30.0, "D6", 0.5), (31.0, "G6", 3.0),
]


def chord_at(t):
    cur = CHORDS[0]
    for c in CHORDS:
        if c[0] <= t + 1e-6:
            cur = c
    return cur


def build_music():
    drums, keys, bassb, melb, padb = Bus(), Bus(), Bus(), Bus(), Bus()
    end_music = 16.6
    # intro bar: soft pad + ticking, riser into the drop on the first flash
    padb.add(0.0, pad([note(n) for n in CHORDS[0][2]], 2.1), 0.5)
    for b in range(4):
        drums.add(b * BEAT, shaker(0.8), 0.6, pan=0.3)
    rs = sfx_whoosh(1.05, up=True, peak_at=0.97)
    drums.add(0.95, rs, 0.35)
    # groove from the drop (2.0 s) until the end card resolves
    beat_t = 2.0
    drums_end = 14.5                      # drums step aside for the end card
    while beat_t < drums_end - 0.01:
        bi = int(round((beat_t - 2.0) / BEAT))
        drums.add(beat_t, lp(kick(), 2500), 0.62)
        if bi % 2 == 1:
            drums.add(beat_t, clap(), 0.5, pan=-0.05)
            drums.add(beat_t + 0.004, snap(), 0.35, pan=0.15)
        for s16 in range(4):
            drums.add(beat_t + s16 * BEAT / 4, shaker(1.0 if s16 == 2 else 0.55), 0.7, pan=0.35)
        beat_t += BEAT
    # bass + keys follow the chords
    for i, (t0, root, tones) in enumerate(CHORDS):
        t1 = CHORDS[i + 1][0] if i + 1 < len(CHORDS) else end_music
        if t0 >= 2.0 or t0 >= 15.5:
            rf = note(root)
            for off, ln, mul in [(0.0, 0.42, 1), (0.75, 0.2, 1), (1.0, 0.42, 2), (1.5, 0.3, 1)]:
                if t0 + off < t1 - 0.05:
                    bassb.add(t0 + off, bass(rf * mul, min(ln, t1 - t0 - off)), 1.0)
            for off in np.arange(0.25, t1 - t0, 0.5):        # off-beat stabs
                for k, n in enumerate(tones):
                    keys.add(t0 + off, epiano(note(n), 0.32, 0.9), 0.55, pan=-0.25 + 0.15 * k)
        padb.add(t0, pad([note(n) for n in tones], t1 - t0 + 0.3), 0.35)
    for b, n, ln in MELODY:
        t0 = b * BEAT
        if t0 < end_music:
            melb.add(t0, bell(note(n), min(1.4, ln * BEAT + 0.6)), 0.85, pan=0.1)
    # final chord rings out
    for k, n in enumerate(["G3", "B3", "D4", "F#4", "A4"]):
        keys.add(15.5 + k * 0.02, epiano(note(n), 1.6, 1.0), 0.6, pan=-0.2 + 0.1 * k)
    # sidechain-ish ducking against the kick
    duck = np.ones(N)
    beat_t = 2.0
    while beat_t < drums_end:
        i = int(beat_t * SR)
        L = int(0.22 * SR)
        duck[i:i + L] = np.minimum(duck[i:i + L], 0.55 + 0.45 * (np.arange(min(L, N - i)) / L) ** 0.7)
        beat_t += BEAT
    mix = (drums.stereo() + bassb.stereo() * 0.9 + (keys.stereo() * 0.8 + padb.stereo() * 0.7) * duck[:, None]
           + melb.stereo() * 0.75)
    fade = np.clip((DUR - np.arange(N) / SR) / 0.6, 0, 1)
    return mix * fade[:, None]


def build_sfx(real=False):
    """real=True: the real-footage edit (no cartoon footsteps/pops/babble; laugh on the bride's real laugh)."""
    s = Bus()
    s.add(T(5.5), sfx_tap(), 0.9)
    for f, _ in [(15, "3"), (30, "2"), (45, "1")]:
        s.add(T(f), sfx_beep(988), 0.8)
    for k in range(6):
        if not real:
            s.add(T(17 + 18 * k / 5) + 0.01, sfx_patter(), 0.55, pan=0.2)
    for f in FLASHES:
        s.add(T(f) - 0.01, sfx_shutter(), 1.0)
    if not real:
        s.add(T(64), sfx_pop(420, 1000), 0.6, pan=0.3)       # mustache + sign pop in
        s.add(T(66), sfx_pop(520, 1250), 0.5, pan=-0.3)
        s.add(T(80), sfx_pop(650, 1500, 0.12), 0.65)          # heart
        s.add(T(112), sfx_land(), 0.5)
    else:
        for fr in (15, 30, 45):                                # gold numbers popping up on the floor
            s.add(T(fr), sfx_pop(380, 900, 0.1), 0.45, pan=0.1)
    s.add(T(122), sfx_printer(T(168) - T(122)), 1.0)
    s.add(T(168), sfx_cut(), 0.9)
    s.add(T(181), sfx_flutter(0.45), 1.0)
    s.add(T(193), sfx_whoosh(T(212) - T(193), up=False, peak_at=0.85), 0.8)
    s.add(T(222) - 0.01, sfx_land(), 1.0)
    for i, (a, b) in enumerate([(230, 252), (256, 279), (282, 293)]):
        s.add(T(a), sfx_scribble(T(b) - T(a), 10 + i), 1.0, pan=0.25)
    s.add(T(306), sfx_pickup(), 0.9, pan=0.15)
    s.add(T(318), sfx_beep(1000, 0.34), 0.55)
    if not real:
        voice = sfx_babble(T(322), T(360))
        s.add(T(322), phone_filter(voice), 0.85, pan=0.1)
        s.add(T(362), phone_filter(sfx_laugh()), 0.9, pan=0.1)
        s.add(T(377), phone_filter(sfx_babble(T(377), T(388))), 0.7, pan=0.1)
    else:
        s.add(T(358), phone_filter(sfx_laugh()), 0.8, pan=0.1)
        s.add(T(368), phone_filter(sfx_laugh()), 0.55, pan=-0.1)
    s.add(T(388), sfx_whoosh(T(434) - T(388), up=True, peak_at=0.5), 0.55)
    for k, n in enumerate(["G5", "B5", "D6", "G6"]):
        s.add(T(392) + 0.07 * k, bell(note(n), 0.9), 0.45, pan=-0.3 + 0.2 * k)
    for k, n in enumerate(["D6", "G6", "B6"]):
        s.add(T(435) + 0.03 * k, bell(note(n), 1.6), 0.4, pan=-0.2 + 0.2 * k)
    return s.stereo()


def master(x, peak_db=-1.0):
    x = x - x.mean(axis=0)
    x = np.tanh(x * 1.1) / 1.1
    pk = np.abs(x).max()
    return x / pk * 10 ** (peak_db / 20)


def write_wav(path, x):
    y = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(y.tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sfx-only", default="")
    ap.add_argument("--real", action="store_true", help="sound design for the real-footage edit")
    a = ap.parse_args()
    music = build_music()
    sfx = build_sfx(real=a.real)
    write_wav(a.out, master(music * 0.62 + sfx * 1.0))
    if a.sfx_only:
        write_wav(a.sfx_only, master(sfx))
    print("wrote", a.out, f"{DUR:.2f}s")


if __name__ == "__main__":
    main()
