"""Prepare a real photo for 2.5D: person matte (U^2-Net human seg), depth (Depth Anything V2),
and a clean background plate with the people removed. Nothing is generated: the models only
*measure* the real photo (who is a person, how far away things are).

    python realfx/prepare.py --image hero.png --out prep/ --models models/
"""
import argparse
import os

import cv2
import numpy as np
import onnxruntime as ort

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def person_matte(img, model):
    s = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    h, w = img.shape[:2]
    x = cv2.resize(img, (320, 320), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    x = (x - MEAN) / STD
    out = s.run(None, {s.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0][0, 0]
    out = (out - out.min()) / (out.max() - out.min() + 1e-8)
    m = cv2.resize(out, (w, h), interpolation=cv2.INTER_CUBIC)
    # edge-aware refinement: guided by the photo so hair / sleeves follow real edges
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    m = guided_filter(g, m, 8, 1e-3)
    return np.clip((m - 0.35) / 0.3, 0, 1)


def guided_filter(I, p, r, eps):
    k = (2 * r + 1, 2 * r + 1)
    mI = cv2.blur(I, k)
    mp = cv2.blur(p, k)
    cIp = cv2.blur(I * p, k) - mI * mp
    vI = cv2.blur(I * I, k) - mI * mI
    a = cIp / (vI + eps)
    b = mp - a * mI
    return cv2.blur(a, k) * I + cv2.blur(b, k)


def depth(img, model):
    s = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    h, w = img.shape[:2]
    x = cv2.resize(img, (518, 518), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255
    x = (x - MEAN) / STD
    d = s.run(None, {s.get_inputs()[0].name: x.transpose(2, 0, 1)[None]})[0][0]
    d = cv2.resize(d, (w, h), interpolation=cv2.INTER_CUBIC)
    return (d - d.min()) / (d.max() - d.min() + 1e-8)       # relative disparity, 1 = near


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--exclude", action="append", default=[],
                    help="x0,y0,x1,y1 box (px) to drop from the person matte, e.g. a drape the model mistook for a person")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    img = cv2.cvtColor(cv2.imread(a.image), cv2.COLOR_BGR2RGB)
    m = person_matte(img, os.path.join(a.models, "u2net_human_seg.onnx"))
    for box in a.exclude:
        x0, y0, x1, y1 = (int(v) for v in box.split(","))
        m[y0:y1, x0:x1] = 0
    d = depth(img, os.path.join(a.models, "depth_anything_v2_vits.onnx"))
    # clean plate: remove the people (generous margin) and fill from the surroundings
    hole = (cv2.dilate((m > 0.05).astype(np.uint8), np.ones((41, 41), np.uint8)) > 0).astype(np.uint8)
    near = (cv2.dilate((m > 0.05).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0).astype(np.float32)
    small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    hs = cv2.resize(hole, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    fill = cv2.inpaint(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), hs * 255, 9, cv2.INPAINT_TELEA)
    fill = cv2.cvtColor(cv2.resize(fill, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
    soft = cv2.GaussianBlur(near, (7, 7), 0)[..., None]   # keep every original pixel that isn't behind a person
    plate = (img * (1 - soft) + fill * soft).astype(np.uint8)
    # background depth: same idea, so the wall behind the couple continues smoothly
    dbg = cv2.inpaint((d * 65535).astype(np.uint16)[::4, ::4].astype(np.float32).astype(np.uint16) // 256 * 0 +
                      (d[::4, ::4] * 255).astype(np.uint8), hole[::4, ::4] * 255, 15, cv2.INPAINT_TELEA)
    dbg = cv2.resize(dbg.astype(np.float32) / 255, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    dbg = np.where(hole > 0, dbg, d)
    cv2.imwrite(os.path.join(a.out, "matte.png"), (m * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(a.out, "plate.png"), cv2.cvtColor(plate, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(a.out, "depth.png"), (np.clip(d, 0, 1) * 65535).astype(np.uint16))
    cv2.imwrite(os.path.join(a.out, "depth_bg.png"), (np.clip(dbg, 0, 1) * 65535).astype(np.uint16))
    rgba = np.dstack([img, (m * 255).astype(np.uint8)])
    cv2.imwrite(os.path.join(a.out, "people.png"), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    ys, xs = np.where(m > 0.5)
    print("people bbox (px):", xs.min(), ys.min(), xs.max(), ys.max())
    print("people median disparity:", float(np.median(d[m > 0.5])), " bg range:", float(d.min()), float(d.max()))


if __name__ == "__main__":
    main()
