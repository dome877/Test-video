"""Brand overlays + grade + end card + audio for the real-people reel, reusing post/composite.py.

The 2D countdown ring is switched off (the countdown is the Blender gold 3D numbers standing on the
real floor); everything else — hook, section headlines, viewfinder corners, flashes, whip, end card —
is the same as the animated reel, so both versions feel like one brand.

    python realfx/finish_real.py --frames frames/ --audio mix.wav --sfx sfx.wav --out reel.mp4 --out-sfx reel_sfx.mp4
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "post"))
sys.path.insert(0, os.path.join(HERE, "..", "blender"))
import composite as C  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--frames", required=True)
ap.add_argument("--audio", required=True)
ap.add_argument("--sfx", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--out-sfx", required=True)
ap.add_argument("--stills-dir", default="")
a = ap.parse_args()

C.COUNTDOWN = []
argv = ["composite.py", "--frames", a.frames, "--audio", a.audio, "--out", a.out,
        "--alt-audio", a.sfx, "--alt-out", a.out_sfx]
if a.stills_dir:
    argv += ["--stills", "0,40,62,80,95,160,190,250,295,310,350,420,470,509", "--stills-dir", a.stills_dir]
sys.argv = argv
C.main()
