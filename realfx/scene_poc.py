"""Real-people test shot (5 s): a real Gran Vía photo becomes a 3D space.

  f0-59    the photo is rebuilt as geometry at its measured depth; the camera pushes in with real
           parallax while neon-style 3D countdown numbers sit *behind* the couple.
  f60      flash
  f60-149  that moment is now a physical 4x6 print that drops onto a pile of real booth prints.

    python realfx/scene_poc.py -- --prep prep/ --prints prints/ --print-tex hero_print.png --out poc.blend
"""
import argparse
import math
import os
import random
import sys

import bpy  # noqa: I001
import bmesh
import numpy as np
from mathutils import Euler, Quaternion, Vector

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "blender"))
from anim import clamp, in_cubic, in_out, lerp, out_back, out_cubic, smooth, wobble  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_DIR = os.path.join(REPO, "assets", "fonts")
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--prep", required=True)
ap.add_argument("--prints", required=True)
ap.add_argument("--print-tex", default="")
ap.add_argument("--out", required=True)
A = ap.parse_args(argv)

N_FRAMES = 150
FLASH = 60
VFOV = math.radians(66.0)          # phone main camera, portrait
W, H = 1080, 1920
TAN_Y = math.tan(VFOV / 2)
TAN_X = TAN_Y * W / H
INV_FAR, INV_NEAR = 1 / 14.0, 0.533   # disparity -> metres (couple ends up ~3 m away)
random.seed(4)

bpy.ops.wm.read_factory_settings(use_empty=True)
SC = bpy.context.scene
COLL = SC.collection


def srgb(h, a=1.0):
    h = h.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c) + (a,)


def link(ob, parent=None):
    COLL.objects.link(ob)
    if parent:
        ob.parent = parent
    return ob


def load_png16(path):
    img = bpy.data.images.load(path)
    img.colorspace_settings.name = "Non-Color"
    w, h = img.size
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    return px.reshape(h, w, 4)[::-1, :, 0]   # top row first


def mat_photo(name, path, alpha=False):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(path)
    tex.interpolation = "Cubic"
    tex.extension = "EXTEND"
    em = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(tex.outputs["Color"], em.inputs["Color"])
    if alpha:
        tr = nt.nodes.new("ShaderNodeBsdfTransparent")
        mix = nt.nodes.new("ShaderNodeMixShader")
        nt.links.new(tex.outputs["Alpha"], mix.inputs[0])
        nt.links.new(tr.outputs[0], mix.inputs[1])
        nt.links.new(em.outputs[0], mix.inputs[2])
        nt.links.new(mix.outputs[0], out.inputs[0])
    else:
        nt.links.new(em.outputs[0], out.inputs[0])
    m.cycles.emission_sampling = "NONE"
    return m


def depth_mesh(name, disp, mat, step=4, keep=None, cut_ratio=1.22, zfunc=None):
    """Grid over the photo; each vertex pushed along its camera ray to the measured depth."""
    h, w = disp.shape
    ys = np.arange(0, h, step)
    xs = np.arange(0, w, step)
    if ys[-1] != h - 1:
        ys = np.append(ys, h - 1)
    if xs[-1] != w - 1:
        xs = np.append(xs, w - 1)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    d = disp[gy, gx]
    z = zfunc(d) if zfunc else 1.0 / (INV_FAR + d * (INV_NEAR - INV_FAR))
    ndx = (gx + 0.5) / w * 2 - 1
    ndy = 1 - (gy + 0.5) / h * 2
    verts = np.stack([ndx * TAN_X * z, z, ndy * TAN_Y * z], -1).reshape(-1, 3)
    uvs = np.stack([(gx + 0.5) / w, 1 - (gy + 0.5) / h], -1).reshape(-1, 2)
    R, C = gy.shape
    idx = np.arange(R * C).reshape(R, C)
    a, b, c_, d_ = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
    quads = np.stack([a, d_, c_, b], -1).reshape(-1, 4)
    zz = z.reshape(-1)
    zq = zz[quads]
    ok = zq.max(1) / zq.min(1) < cut_ratio          # tear at depth edges instead of stretching
    if keep is not None:
        kk = keep[gy, gx].reshape(-1)
        ok &= kk[quads].max(1) > 0.02
    quads = quads[ok]
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(verts))
    me.vertices.foreach_set("co", verts.astype(np.float32).ravel())
    me.loops.add(len(quads) * 4)
    me.polygons.add(len(quads))
    me.loops.foreach_set("vertex_index", quads.astype(np.int32).ravel())
    me.polygons.foreach_set("loop_start", np.arange(0, len(quads) * 4, 4, dtype=np.int32))
    me.polygons.foreach_set("loop_total", np.full(len(quads), 4, np.int32))
    uv = me.uv_layers.new(name="UVMap")
    uv.data.foreach_set("uv", uvs[quads.ravel()].astype(np.float32).ravel())
    me.update()
    me.materials.append(mat)
    return link(bpy.data.objects.new(name, me))


def font(name):
    return bpy.data.fonts.load(os.path.join(FONT_DIR, name + ".ttf"), check_existing=True)


def px_to_world(px, py, z):
    ndx = (px + 0.5) / W * 2 - 1
    ndy = 1 - (py + 0.5) / H * 2
    return Vector((ndx * TAN_X * z, z, ndy * TAN_Y * z))


def key(idb, path, fn, n=3, frames=range(N_FRAMES), interp=1):
    if idb.animation_data is None:
        idb.animation_data_create()
    if idb.animation_data.action is None:
        idb.animation_data.action = bpy.data.actions.new(idb.name + "_anim")
    vals = [fn(f) for f in frames]
    if n == 1:
        vals = [(v,) for v in vals]
    for i in range(n):
        fc = idb.animation_data.action.fcurve_ensure_for_datablock(idb, path, index=i)
        kp = fc.keyframe_points
        kp.add(len(vals))
        co = []
        for f, v in zip(frames, vals):
            co += [f + 1, float(v[i])]
        kp.foreach_set("co", co)
        kp.foreach_set("interpolation", [interp] * len(vals))
        fc.update()


def look_euler(src, dst, prev=None):
    e = (Vector(dst) - Vector(src)).to_track_quat("-Z", "Y").to_euler()
    if prev is not None:
        e.make_compatible(prev)
    return e


# ------------------------------------------------------------------------------------------
# 1) the real photo as 3D: clean background plate + the people at their own depth
# ------------------------------------------------------------------------------------------
disp = load_png16(os.path.join(A.prep, "depth.png"))
disp_bg = load_png16(os.path.join(A.prep, "depth_bg.png"))
matte = load_png16(os.path.join(A.prep, "matte.png"))

# people depth extended a little past their edges so silhouettes don't smear backwards
import cv2  # noqa: E402
hard = (matte > 0.5).astype(np.uint8)
d8 = (disp * 255).astype(np.uint8)
ext = cv2.inpaint(d8[::2, ::2], (1 - hard[::2, ::2]) * 255, 5, cv2.INPAINT_TELEA).astype(np.float32) / 255
ext = cv2.resize(ext, (W, H), interpolation=cv2.INTER_LINEAR)
disp_people = np.where(hard > 0, disp, ext)
disp_people = cv2.GaussianBlur(disp_people, (0, 0), 6)

# the night sky is pure black: pin it to the depth of whatever it touches (arch, hedge) so edges
# don't stretch into it when the camera moves
_plate = cv2.cvtColor(cv2.imread(os.path.join(A.prep, "plate.png")), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
sky = ((disp_bg < 0.32) & (cv2.GaussianBlur(_plate, (0, 0), 4) < 0.07)).astype(np.uint8)
q = cv2.inpaint((disp_bg[::4, ::4] * 255).astype(np.uint8), sky[::4, ::4] * 255, 9, cv2.INPAINT_TELEA)
q = cv2.resize(q.astype(np.float32) / 255, (W, H), interpolation=cv2.INTER_LINEAR)
disp_bg = np.where(sky > 0, q, disp_bg)
disp_bg = cv2.GaussianBlur(disp_bg, (0, 0), 2.5)
_p = float(np.median(disp[matte > 0.5]))
_zp = 1.0 / (INV_FAR + _p * (INV_NEAR - INV_FAR))


def bg_z(d):
    """Floor in front keeps its measured depth; everything behind the couple sits in a tight band
    (0.35-1.3 m behind them) so it separates from the couple but holds together as one backdrop."""
    front = 1.0 / (INV_FAR + d * (INV_NEAR - INV_FAR))
    behind = _zp + 0.35 + (np.clip(_p - d, 0, None) / _p) * 0.95
    w = np.clip((d - (_p - 0.04)) / 0.08, 0, 1)
    return front * w + behind * (1 - w)


bg = depth_mesh("photo_background", disp_bg, mat_photo("plate", os.path.join(A.prep, "plate.png")), step=3,
                cut_ratio=100.0, zfunc=bg_z)
people = depth_mesh("photo_people", disp_people, mat_photo("people", os.path.join(A.prep, "people.png"), alpha=True),
                    step=3, keep=matte, cut_ratio=10.0)
people.location.y -= 0.01

# ------------------------------------------------------------------------------------------
# 2) neon 3D countdown that lives between the couple and the arch
# ------------------------------------------------------------------------------------------
neon = bpy.data.materials.new("neon_numerals")
neon.use_nodes = True
nt = neon.node_tree
nt.nodes.clear()
o = nt.nodes.new("ShaderNodeOutputMaterial")
em = nt.nodes.new("ShaderNodeEmission")
em.inputs["Color"].default_value = srgb("#FFD9A6")
em.inputs["Strength"].default_value = 1.35
nt.links.new(em.outputs[0], o.inputs[0])
neon.cycles.emission_sampling = "NONE"

_pd = float(np.median(disp[matte > 0.5]))
PEOPLE_Z = 1.0 / (INV_FAR + _pd * (INV_NEAR - INV_FAR))
NUM_Z = PEOPLE_Z + 0.24                   # just behind the couple, in front of the arch
NUM_POS = px_to_world(845, 740, NUM_Z)
numerals = []
for i, label in enumerate(["3", "2", "1"]):
    cu = bpy.data.curves.new(f"num_{label}", "FONT")
    cu.body = label
    cu.font = font("Caveat-Medium")         # same hand as the real "Better Together" neon
    cu.size = 1.65
    cu.align_x = "CENTER"
    cu.align_y = "CENTER"
    cu.extrude = 0.01
    cu.bevel_depth = 0.008
    cu.offset = 0.004
    cu.materials.append(neon)
    ob = link(bpy.data.objects.new(f"num_{label}", cu))
    ob.location = NUM_POS
    ob.rotation_euler = Euler((math.radians(90), 0, 0))      # faces the camera
    numerals.append(ob)


def numeral_scale(i):
    start = 15 * (i + 1) - 15 + 15        # 3 at f15, 2 at f30, 1 at f45
    def fn(f):
        if f < start or f >= start + 15:
            return (0.0, 0.0, 0.0)
        t = f - start
        s = out_back(clamp(t / 6), 2.0) if t < 11 else 1 - in_cubic(clamp((t - 11) / 4))
        return (s, s, s)
    return fn


for i, ob in enumerate(numerals):
    key(ob, "scale", numeral_scale(i))
    key(ob, "rotation_euler", lambda f, i=i: (math.radians(90), 0, math.radians(8 * math.sin(f * 0.12 + i))))


# ------------------------------------------------------------------------------------------
# 3) the print scene (lives far away, has its own camera)
# ------------------------------------------------------------------------------------------
PRINT_O = Vector((0, 200, 0))
PW, PH = 1.0, 1.5


def principled(name, color, rough=0.5, coat=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = srgb(color)
    p.inputs["Roughness"].default_value = rough
    p.inputs["Coat Weight"].default_value = coat
    return m


def print_material(name, photo_path):
    """4x6 glossy print: white border, the photo, GRAN VÍA footer — composed as one texture."""
    from PIL import Image, ImageDraw, ImageFont
    pw, ph = 1000, 1500
    paper = Image.new("RGB", (pw, ph), (250, 249, 246))
    photo = Image.open(photo_path).convert("RGB")
    tw, th = 880, 1100
    r = photo.width / photo.height
    if r > tw / th:
        nw = int(photo.height * tw / th)
        photo = photo.crop(((photo.width - nw) // 2, 0, (photo.width - nw) // 2 + nw, photo.height))
    else:
        nh = int(photo.width * th / tw)
        photo = photo.crop((0, 0, photo.width, nh))
    paper.paste(photo.resize((tw, th), Image.LANCZOS), (60, 60))
    d = ImageDraw.Draw(paper)
    f1 = ImageFont.truetype(os.path.join(FONT_DIR, "PlayfairDisplay-Bold.ttf"), 64)
    f2 = ImageFont.truetype(os.path.join(FONT_DIR, "Montserrat-Medium.ttf"), 26)
    w1 = d.textlength("GRAN VÍA", font=f1)
    x0 = pw / 2 - (w1 + 40 + 150) / 2
    d.text((x0, 1300), "GRAN VÍA", font=f1, fill=(34, 30, 28), anchor="lm")
    d.line([(x0 + w1 + 20, 1272), (x0 + w1 + 20, 1328)], fill=(34, 30, 28), width=3)
    xs = x0 + w1 + 40
    for ch in "STUDIO":
        d.text((xs, 1302), ch, font=f2, fill=(34, 30, 28), anchor="lm")
        xs += d.textlength(ch, font=f2) + 9
    d.text((pw / 2, 1400), "12.6.2027.", font=f2, fill=(90, 82, 78), anchor="mm")
    path = os.path.join(os.path.dirname(os.path.abspath(A.out)), name + ".png")
    paper.save(path)
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    tex = m.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(path)
    tex.interpolation = "Cubic"
    m.node_tree.links.new(tex.outputs["Color"], p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.22
    p.inputs["Coat Weight"].default_value = 0.6
    p.inputs["Coat Roughness"].default_value = 0.08
    return m


def make_print(name, mat):
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=12, y_segments=18, size=0.5)
    bmesh.ops.scale(bm, vec=(PW, PH, 1), verts=bm.verts)
    uvl = bm.loops.layers.uv.new("UVMap")
    for fc in bm.faces:
        for lp in fc.loops:
            co = lp.vert.co
            lp[uvl].uv = (co.x / PW + 0.5, co.y / PH + 0.5)
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    me.materials.append(mat)
    ob = link(bpy.data.objects.new(name, me))
    so = ob.modifiers.new("solid", "SOLIDIFY")
    so.thickness = 0.006
    return ob


table = bpy.data.objects.new("table", bpy.data.meshes.new("table"))
bm = bmesh.new()
bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=6)
bm.to_mesh(table.data)
bm.free()
wood = bpy.data.materials.new("dark_walnut")
wood.use_nodes = True
wnt = wood.node_tree
wp = wnt.nodes["Principled BSDF"]
wave = wnt.nodes.new("ShaderNodeTexWave")
wave.inputs["Scale"].default_value = 1.6
wave.inputs["Distortion"].default_value = 7
wave.inputs["Detail"].default_value = 3
ramp = wnt.nodes.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].color = srgb("#1B1512")
ramp.color_ramp.elements[1].color = srgb("#3A2A20")
wnt.links.new(wave.outputs["Fac"], ramp.inputs[0])
wnt.links.new(ramp.outputs[0], wp.inputs["Base Color"])
wp.inputs["Roughness"].default_value = 0.35
wp.inputs["Coat Weight"].default_value = 0.3
table.data.materials.append(wood)
link(table)
table.location = PRINT_O

pile = []
for k, fn in enumerate(sorted(os.listdir(A.prints))):
    pm = print_material(f"print_{k}", os.path.join(A.prints, fn))
    ob = make_print(f"pile_{k}", pm)
    a = random.uniform(-0.9, 0.9)
    ob.location = PRINT_O + Vector((math.cos(k * 1.9) * 0.75, math.sin(k * 1.7) * 0.6, 0.007 * (k + 1)))
    ob.rotation_euler = (0, 0, a)
    pile.append(ob)

hero_tex = A.print_tex or os.path.join(A.prep, "plate.png")
hero = make_print("hero_print", print_material("print_hero", hero_tex))
HERO_REST = PRINT_O + Vector((0.05, -0.05, 0.08))


def hero_xf(f):
    if f < FLASH:
        return PRINT_O + Vector((0, 0, 40)), (0, 0, 0)
    t = clamp((f - (FLASH + 8)) / 44)
    e = in_out(t)
    loc = HERO_REST + Vector((0, 0, 1.5 * (1 - e) ** 1.4))
    rot = (math.radians(14) * (1 - e) * math.sin(f * 0.15), math.radians(-10) * (1 - e), lerp(0.0, -0.12, e))
    if f > FLASH + 52:
        bounce = math.sin((f - FLASH - 52) * 0.9) * 0.012 * math.exp(-(f - FLASH - 52) / 5)
        loc = loc + Vector((0, 0, abs(bounce)))
    return loc, rot


key(hero, "location", lambda f: tuple(hero_xf(f)[0]))
key(hero, "rotation_euler", lambda f: hero_xf(f)[1])

for name, loc, power, size in [("print_key", PRINT_O + Vector((-2.0, -1.6, 3.4)), 600, 2.5),
                               ("print_rim", PRINT_O + Vector((2.4, 2.2, 2.0)), 260, 2.0)]:
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy = power
    ld.size = size
    ld.color = srgb("#FFDDB5")[:3]
    lo = link(bpy.data.objects.new(name, ld))
    lo.location = loc
    lo.rotation_euler = look_euler(loc, PRINT_O)

# ------------------------------------------------------------------------------------------
# cameras
# ------------------------------------------------------------------------------------------
def make_cam(name):
    cd = bpy.data.cameras.new(name)
    cd.sensor_fit = "VERTICAL"
    cd.angle_y = VFOV
    cd.clip_start = 0.05
    cd.clip_end = 500
    return link(bpy.data.objects.new(name, cd))


cam_photo = make_cam("cam_photo")
TARGET = px_to_world(600, 960, 3.0)


def photo_cam(f):
    t = smooth(clamp(f / FLASH))
    pos = Vector((lerp(0.0, 0.16, t), lerp(0.0, 0.36, t), lerp(0.0, 0.03, t)))
    pos += Vector((wobble(f, 1, 0.006), 0, wobble(f, 2, 0.005)))
    return pos


prev = [None]


def photo_rot(f):
    t = smooth(clamp(f / FLASH))
    pos = photo_cam(f)
    tgt = Vector((0, 3.0, 0)).lerp(TARGET, t * 0.85)
    tgt.z = pos.z + 0.02                   # stay level: never look past the photo's bottom edge
    e = look_euler(pos, tgt, prev[0])
    prev[0] = e
    return tuple(e)


key(cam_photo, "location", lambda f: tuple(photo_cam(f)))
key(cam_photo, "rotation_euler", photo_rot)
key(cam_photo.data, "angle_y", lambda f: VFOV * lerp(1.0, 0.84, smooth(clamp(f / FLASH))), n=1)

cam_print = make_cam("cam_print")
cam_print.data.angle_y = math.radians(50)


def print_cam(f):
    # starts tight on the falling print (the photo fills the frame) and pulls up to the pile
    t = in_out(clamp((f - FLASH) / 70))
    hp, _ = hero_xf(max(f, FLASH))
    start = hp + Vector((0, -0.05, 1.45))
    end = PRINT_O + Vector((0.05, -1.25, 3.2))
    pos = start.lerp(end, t) + Vector((wobble(f, 3, 0.01), wobble(f, 4, 0.01), 0))
    tgt = hp.lerp(PRINT_O + Vector((0.05, 0.15, 0)), t)
    return pos, tgt


prev2 = [None]


def print_rot(f):
    pos, tgt = print_cam(f)
    e = (tgt - pos).to_track_quat("-Z", "Y")
    e = e.to_euler()
    if prev2[0] is not None:
        e.make_compatible(prev2[0])
    prev2[0] = e
    return tuple(e)


key(cam_print, "location", lambda f: tuple(print_cam(f)[0]))
key(cam_print, "rotation_euler", print_rot)
cam_print.data.dof.use_dof = True
cam_print.data.dof.aperture_fstop = 2.2
key(cam_print.data, "dof.focus_distance", lambda f: (print_cam(f)[0] - hero_xf(max(f, FLASH))[0]).length, n=1)

for nm, fr, cam in (("photo", 0, cam_photo), ("print", FLASH, cam_print)):
    mk = SC.timeline_markers.new(nm, frame=fr + 1)
    mk.camera = cam
SC.camera = cam_photo

# ------------------------------------------------------------------------------------------
SC.world = bpy.data.worlds.new("w")
SC.world.use_nodes = True
SC.world.node_tree.nodes["Background"].inputs["Color"].default_value = srgb("#050506")
SC.frame_start, SC.frame_end = 1, N_FRAMES
SC.render.fps = 30
SC.render.resolution_x, SC.render.resolution_y = W, H
SC.render.resolution_percentage = 80
SC.render.engine = "CYCLES"
SC.cycles.device = "CPU"
SC.cycles.samples = 24
SC.cycles.use_denoising = True
SC.cycles.max_bounces = 4
SC.cycles.use_light_tree = False
SC.render.use_motion_blur = True
SC.render.motion_blur_shutter = 0.4
SC.render.use_persistent_data = True
SC.view_settings.view_transform = "Standard"     # photo pixels pass through untouched
SC.render.image_settings.color_mode = "RGB"
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(A.out))
print("SAVED", A.out)
