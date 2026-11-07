#!/usr/bin/env bash
# Builds the Gran Vía photobooth reel end to end.
#
#   PY=/path/to/python-with-bpy ./make_reel.sh
#
# PY must be a Python that can `import bpy` (pip install bpy==5.2.3) plus numpy, pillow, scipy.
# Set PERCENT=100 for a full-resolution render (slower); default renders at 80% and upscales.
set -euo pipefail

REPO="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python3}"
WORK="${WORK:-/tmp/granvia}"
PERCENT="${PERCENT:-80}"
OUT="${OUT:-$REPO/output}"

mkdir -p "$WORK"/{tex,cells,frames,audio} "$OUT"

echo "== 1/6 placeholder textures (the scene references them)"
"$PY" "$REPO/post/make_textures.py" --placeholders --out "$WORK/tex"

echo "== 2/6 build the Blender scene"
"$PY" "$REPO/blender/build_scene.py" -- --tex "$WORK/tex" --out "$WORK/granvia_photobooth.blend"

echo "== 3/6 render the four booth photos, then the strip + touchscreen textures"
"$PY" "$REPO/blender/render.py" -- --blend "$WORK/granvia_photobooth.blend" --out "$WORK/cells" --cells --samples 32
"$PY" "$REPO/post/make_textures.py" --cells "$WORK/cells" --out "$WORK/tex"

echo "== 4/6 render the animation (435 frames)"
"$PY" "$REPO/blender/render.py" -- --blend "$WORK/granvia_photobooth.blend" --out "$WORK/frames" \
    --frames 0-434 --percent "$PERCENT" --skip-existing

echo "== 5/6 soundtrack"
"$PY" "$REPO/audio/soundtrack.py" --out "$WORK/audio/mix.wav" --sfx-only "$WORK/audio/sfx.wav"

echo "== 6/6 composite + encode"
"$PY" "$REPO/post/composite.py" --frames "$WORK/frames" --audio "$WORK/audio/mix.wav" \
    --out "$OUT/granvia-photobooth-reel.mp4" \
    --alt-audio "$WORK/audio/sfx.wav" --alt-out "$OUT/granvia-photobooth-reel-sfx-only.mp4"

echo "done -> $OUT"
