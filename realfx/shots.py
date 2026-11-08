"""Blender shots for the real-people reel. Each real photo/plate is rebuilt as geometry at its
measured depth, then 3D objects are placed on the real surfaces (fitted from the depth):

  numerals   gold 3D "3-2-1" standing on the real floor in front of the guests (RGBA + shadow)
  print      the photo strip slides out of the real booth's front panel
  guestbook  the strip lands on the real guestbook page; "Živjeli mladenci!" gets written in ink
  phone      a vintage audio-guestbook phone in the real booth room; the handset is picked up

    python realfx/shots.py -- --shot print --plate plate.png --depth depth.png --strip strip.png --out shot.blend
"""
import argparse
import math
import os
import sys

import bpy  # noqa: I001
import bmesh
import numpy as np
from mathutils import Euler, Matrix, Quaternion, Vector

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "blender"))
from anim import clamp, in_cubic, in_out, in_quad, lerp, out_back, out_cubic, smooth, wobble  # noqa: E402

FONT_DIR = os.path.join(HERE, "..", "assets", "fonts")
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--shot", required=True, choices=["numerals", "print", "guestbook", "phone"])
ap.add_argument("--plate", required=True)
ap.add_argument("--depth", required=True)
ap.add_argument("--strip", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--ink-only", action="store_true", help="guestbook: render only the handwriting, transparent, "
                "so it can be laid over the page in compositing (the real page must never hide the ink)")
A = ap.parse_args(argv)

W, H = 1080, 1920
VFOV = math.radians(66.0)
TAN_Y = math.tan(VFOV / 2)
TAN_X = TAN_Y * W / H
NEAR_FAR = {"numerals": (1.6, 7.0), "print": (1.2, 7.0), "guestbook": (0.42, 3.0), "phone": (1.2, 7.0)}
NEAR, FAR = NEAR_FAR[A.shot]

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


def font(name):
    return bpy.data.fonts.load(os.path.join(FONT_DIR, name + ".ttf"), check_existing=True)


def key(idb, path, fn, n, frames, interp=1):
    if idb.animation_data is None:
        idb.animation_data_create()
    if idb.animation_data.action is None:
        idb.animation_data.action = bpy.data.actions.new(idb.name + "_anim")
    vals = [fn(f) for f in frames]
    if n == 1:
        vals = [(v,) for v in vals]
    if path == "rotation_euler":
        prev, fixed = None, []
        for v in vals:
            e = Euler(v)
            if prev is not None:
                e.make_compatible(prev)
            fixed.append(tuple(e))
            prev = e
        vals = fixed
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


# ------------------------------------------------------------------------------------------
# plate -> geometry
# ------------------------------------------------------------------------------------------
def load_gray(path):
    img = bpy.data.images.load(path)
    img.colorspace_settings.name = "Non-Color"
    w, h = img.size
    px = np.empty(w * h * 4, np.float32)
    img.pixels.foreach_get(px)
    return px.reshape(h, w, 4)[::-1, :, 0]


DISP = load_gray(A.depth)


def zmap(d):
    return 1.0 / (1 / FAR + d * (1 / NEAR - 1 / FAR))


def ray(px, py):
    return Vector((((px + 0.5) / W * 2 - 1) * TAN_X, 1.0, (1 - (py + 0.5) / H * 2) * TAN_Y))


def px3d(px, py):
    d = float(DISP[int(np.clip(py, 0, H - 1)), int(np.clip(px, 0, W - 1))])
    return ray(px, py) * zmap(d)


def fit_plane(boxes, stride=12, keep=None):
    pts = []
    for (x0, y0, x1, y1) in boxes:
        for y in range(y0, y1, stride):
            for x in range(x0, x1, stride):
                if keep is None or keep(x, y):
                    pts.append(px3d(x, y))
    P = np.array([tuple(p) for p in pts])
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c)
    n = Vector(vt[2])
    if n.dot(Vector(-c)) < 0:
        n = -n
    return Vector(c), n.normalized()


def surf(px, py, lift=0.02):
    """Point on the reconstructed surface at a pixel, nudged toward the camera so it always sits on top."""
    import cv2
    d = float(cv2.GaussianBlur(DISP, (0, 0), 2.0)[int(py), int(px)])
    return ray(px, py) * zmap(d) * (1 - lift)


def hit(px, py, plane):
    c, n = plane
    r = ray(px, py)
    t = c.dot(n) / r.dot(n)
    return r * t


def plate_mesh(path, step=3):
    gy, gx = np.meshgrid(np.r_[np.arange(0, H, step), H - 1], np.r_[np.arange(0, W, step), W - 1], indexing="ij")
    import cv2
    d = cv2.GaussianBlur(DISP, (0, 0), 2.0)[gy, gx]
    z = zmap(d)
    ndx = (gx + 0.5) / W * 2 - 1
    ndy = 1 - (gy + 0.5) / H * 2
    verts = np.stack([ndx * TAN_X * z, z, ndy * TAN_Y * z], -1).reshape(-1, 3)
    uvs = np.stack([(gx + 0.5) / W, 1 - (gy + 0.5) / H], -1).reshape(-1, 2)
    R, C = gy.shape
    idx = np.arange(R * C).reshape(R, C)
    quads = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
    me = bpy.data.meshes.new("plate")
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
    m = bpy.data.materials.new("plate")
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
    # camera sees the photo; everything else (shadows, reflections on CG objects) sees soft room light
    lp = nt.nodes.new("ShaderNodeLightPath")
    mix = nt.nodes.new("ShaderNodeMixShader")
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    nt.links.new(lp.outputs["Is Shadow Ray"], mix.inputs[0])
    nt.links.new(em.outputs[0], mix.inputs[1])
    nt.links.new(tr.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs[0])
    m.cycles.emission_sampling = "NONE"
    me.materials.append(m)
    ob = link(bpy.data.objects.new("plate", me))
    ob.visible_shadow = False
    return ob


# ------------------------------------------------------------------------------------------
# materials + objects
# ------------------------------------------------------------------------------------------
def principled(name, color, rough=0.5, metal=0.0, coat=0.0, emit=None, emit_str=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = srgb(color) if isinstance(color, str) else color
    p.inputs["Roughness"].default_value = rough
    p.inputs["Metallic"].default_value = metal
    p.inputs["Coat Weight"].default_value = coat
    if emit:
        p.inputs["Emission Color"].default_value = srgb(emit)
        p.inputs["Emission Strength"].default_value = emit_str
    return m


def photo_strip_material(path):
    m = principled("strip", "#FFFFFF", rough=0.25, coat=0.6)
    nt = m.node_tree
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(path)
    tex.interpolation = "Cubic"
    nt.links.new(tex.outputs["Color"], nt.nodes["Principled BSDF"].inputs["Base Color"])
    return m


def quad(name, corners, uvs, mat, thickness=0.0):
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(c) for c in corners], [], [(0, 1, 2, 3)])
    uvl = me.uv_layers.new(name="UVMap")
    for li, vi in enumerate(me.polygons[0].vertices):
        uvl.data[li].uv = uvs[vi]
    me.materials.append(mat)
    ob = link(bpy.data.objects.new(name, me))
    if thickness:
        s = ob.modifiers.new("solid", "SOLIDIFY")
        s.thickness = thickness
    return ob


def shadow_decal(name, mat_strength=0.45, soft=0.35):
    """A soft contact shadow: black with a rounded-rectangle falloff in alpha. Scale/opacity are keyed."""
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    nt.links.new(tc.outputs["UV"], sep.inputs[0])
    fall = []
    for ax in ("X", "Y"):
        sub = nt.nodes.new("ShaderNodeMath")
        sub.operation = "SUBTRACT"
        sub.inputs[1].default_value = 0.5
        nt.links.new(sep.outputs[ax], sub.inputs[0])
        ab = nt.nodes.new("ShaderNodeMath")
        ab.operation = "ABSOLUTE"
        nt.links.new(sub.outputs[0], ab.inputs[0])
        mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["From Min"].default_value = 0.5 - soft
        mr.inputs["From Max"].default_value = 0.5
        mr.inputs["To Min"].default_value = 1.0
        mr.inputs["To Max"].default_value = 0.0
        nt.links.new(ab.outputs[0], mr.inputs["Value"])
        fall.append(mr)
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    nt.links.new(fall[0].outputs[0], mul.inputs[0])
    nt.links.new(fall[1].outputs[0], mul.inputs[1])
    oi = nt.nodes.new("ShaderNodeObjectInfo")
    mul2 = nt.nodes.new("ShaderNodeMath")
    mul2.operation = "MULTIPLY"
    nt.links.new(mul.outputs[0], mul2.inputs[0])
    nt.links.new(oi.outputs["Alpha"], mul2.inputs[1])
    mul3 = nt.nodes.new("ShaderNodeMath")
    mul3.operation = "MULTIPLY"
    mul3.inputs[1].default_value = mat_strength
    nt.links.new(mul2.outputs[0], mul3.inputs[0])
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (0, 0, 0, 1)
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(mul3.outputs[0], mix.inputs[0])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(em.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs[0])
    return m


def ink_reveal(name, color):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")      # ink sits *in* the photo: inherit its exposure, not CG lighting
    em.inputs["Color"].default_value = srgb(color)
    em.inputs["Strength"].default_value = 1.0
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    tc = nt.nodes.new("ShaderNodeTexCoord")
    sep = nt.nodes.new("ShaderNodeSeparateXYZ")
    oi = nt.nodes.new("ShaderNodeObjectInfo")
    sepc = nt.nodes.new("ShaderNodeSeparateColor")
    sub = nt.nodes.new("ShaderNodeMath")
    sub.operation = "SUBTRACT"
    mr = nt.nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = -0.002
    mr.inputs["From Max"].default_value = 0.002
    mul = nt.nodes.new("ShaderNodeMath")
    mul.operation = "MULTIPLY"
    mul.inputs[1].default_value = 0.88
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    nt.links.new(oi.outputs["Color"], sepc.inputs[0])
    nt.links.new(sepc.outputs[0], sub.inputs[0])
    nt.links.new(sep.outputs[0], sub.inputs[1])
    nt.links.new(sub.outputs[0], mr.inputs["Value"])
    nt.links.new(mr.outputs[0], mul.inputs[0])
    nt.links.new(mul.outputs[0], mix.inputs[0])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(em.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs[0])
    return m


def area(name, loc, target, power, color, size=1.0):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy = power
    ld.color = srgb(color)[:3]
    ld.size = size
    ob = link(bpy.data.objects.new(name, ld))
    ob.location = loc
    ob.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
    return ob


def make_cam():
    cd = bpy.data.cameras.new("cam")
    cd.sensor_fit = "VERTICAL"
    cd.sensor_height = 24.0
    cd.lens = 12.0 / math.tan(VFOV / 2)
    cd.clip_start = 0.01
    cd.clip_end = 200
    cam = link(bpy.data.objects.new("cam", cd))
    cam.rotation_euler = (math.radians(90), 0, 0)
    SC.camera = cam
    return cam


def basis(n, u_hint):
    """Orthonormal frame on a plane: u along an image direction, v = n x u (plane-up), n = normal."""
    u = (u_hint - n * u_hint.dot(n)).normalized()
    v = n.cross(u).normalized()
    return u, v, n


def frame_matrix(origin, x, y, z):
    m = Matrix((x, y, z)).transposed().to_4x4()
    m.translation = origin
    return m


def settings(nframes, samples=32, transparent=False):
    SC.frame_start, SC.frame_end = 1, nframes
    SC.render.fps = 30
    SC.render.resolution_x, SC.render.resolution_y = W, H
    SC.render.resolution_percentage = 100
    SC.render.engine = "CYCLES"
    SC.cycles.device = "CPU"
    SC.cycles.samples = samples
    SC.cycles.use_denoising = True
    SC.cycles.max_bounces = 4
    SC.cycles.use_light_tree = False
    SC.render.use_motion_blur = True
    SC.render.motion_blur_shutter = 0.45
    SC.render.use_persistent_data = True
    SC.view_settings.view_transform = "Standard"
    SC.render.film_transparent = transparent
    SC.render.image_settings.color_mode = "RGBA" if transparent else "RGB"
    SC.world = bpy.data.worlds.new("w")
    SC.world.use_nodes = True
    bgn = SC.world.node_tree.nodes["Background"]
    bgn.inputs["Color"].default_value = srgb("#6E625A")     # soft warm room bounce for CG objects
    bgn.inputs["Strength"].default_value = 0.6


# ------------------------------------------------------------------------------------------
# strip helpers (shared by print + guestbook)
# ------------------------------------------------------------------------------------------
STRIP_ASPECT = 3.0


def strip_object(name, width, origin_at_trailing_edge, flip_u=False):
    L = width * STRIP_ASPECT
    hw = width / 2
    if origin_at_trailing_edge:      # extends along local +Y; image top at the trailing edge
        corners = [(-hw, 0, 0), (hw, 0, 0), (hw, L, 0), (-hw, L, 0)]
        uvs = [(0, 1), (1, 1), (1, 0), (0, 0)]
        if flip_u:
            uvs = [(1 - a, b) for a, b in uvs]
    else:                            # centred; image top toward local +Y
        corners = [(-hw, -L / 2, 0), (hw, -L / 2, 0), (hw, L / 2, 0), (-hw, L / 2, 0)]
        uvs = [(0, 0), (1, 0), (1, 1), (0, 1)]
    return quad(name, [Vector(c) for c in corners], uvs, photo_strip_material(A.strip), thickness=width * 0.006)


# ==========================================================================================
if A.shot == "numerals":
    # gold 3D numbers standing on the real floor in front of the guests (rendered with alpha + shadow)
    settings(45, samples=48, transparent=True)
    cam = make_cam()
    floor = fit_plane([(80, 1560, 1000, 1900)])
    u, v, n = basis(floor[1], Vector((1, 0, 0)))
    base = hit(540, 1590, floor)
    up = n
    catcher = bpy.data.objects.new("catcher", bpy.data.meshes.new("catcher"))
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.55)     # only around the numbers: cheap
    bm.to_mesh(catcher.data)
    bm.free()
    link(catcher)
    catcher.matrix_world = frame_matrix(base, u, n.cross(u), n)
    catcher.is_shadow_catcher = True
    gold = principled("gold", "#D9B26A", rough=0.22, metal=1.0, coat=0.3)
    face = principled("gold_face", "#E8C27E", rough=0.16, metal=1.0, coat=0.5)
    nums = []
    for i, label in enumerate(["3", "2", "1"]):
        cu = bpy.data.curves.new("num" + label, "FONT")
        cu.body = label
        cu.font = font("PlayfairDisplay-Bold")
        cu.size = 1.25
        cu.align_x = "CENTER"
        cu.align_y = "BOTTOM_BASELINE"
        cu.extrude = 0.09
        cu.bevel_depth = 0.02
        cu.bevel_resolution = 3
        cu.materials.append(face)
        cu.materials.append(gold)
        ob = link(bpy.data.objects.new("num" + label, cu))
        nums.append(ob)
    # text faces the camera, stands upright on the floor
    fwd = (Vector((0, 0, 0)) - base)
    fwd = (fwd - up * fwd.dot(up)).normalized()           # toward camera, along the floor
    side = up.cross(fwd).normalized()
    M0 = frame_matrix(base, side, up, fwd)
    for i, ob in enumerate(nums):
        start = 15 * i

        def xf(f, start=start):
            t = f - start
            if t < 0 or t >= 15:
                return (0.01, 0.01, 0.01), 0.0      # hidden via hide_render; never a zero scale
            s = max(0.01, out_back(clamp(t / 6), 2.2) if t < 12 else 1 - in_cubic(clamp((t - 12) / 3)))
            return (s, s, s), math.radians(14) * (1 - out_cubic(clamp(t / 8))) * (1 if i % 2 else -1)

        def loc(f, start=start):
            return tuple(M0.translation)

        def rot(f, xf=xf):
            sc, spin = xf(f)
            q = M0.to_quaternion() @ Quaternion((0, 1, 0), spin)
            return tuple(q.to_euler())

        ob.rotation_mode = "XYZ"
        key(ob, "location", loc, 3, range(45))
        key(ob, "rotation_euler", rot, 3, range(45))
        key(ob, "scale", lambda f, xf=xf: xf(f)[0], 3, range(45))
        key(ob, "hide_render", lambda f, start=start: 0.0 if start <= f < start + 15 else 1.0, 1, range(45), interp=0)
        # each numeral also has its own little pop of light
    area("key", base + up * 1.6 + fwd * 0.9 + side * 0.7, base + up * 0.3, 160, "#FFF1DE", size=1.2)
    area("fill", base + up * 0.8 + fwd * 1.2 - side * 1.0, base + up * 0.3, 60, "#FFE7CC", size=1.5)
    cam.data.dof.use_dof = False

# ==========================================================================================
elif A.shot == "print":
    N = 98                                  # global frames 112..209
    settings(N, samples=32)
    cam = make_cam()
    plate_mesh(A.plate)
    panel = fit_plane([(440, 560, 780, 1010)], keep=lambda x, y: not (470 < x < 760 and 600 < y < 860))
    u, v, n = basis(panel[1], px3d(760, 960) - px3d(440, 960))
    if v.z < 0:
        v = -v
    panel_w = (hit(430, 953, panel) - hit(765, 953, panel)).length
    sw = panel_w * 0.16
    slot = hit(600, 952, panel)
    # slot: a dark slit with a brass lip, sitting flush on the real panel
    slit = quad("slit", [slot - u * sw * 0.7 - v * sw * 0.035, slot + u * sw * 0.7 - v * sw * 0.035,
                         slot + u * sw * 0.7 + v * sw * 0.035, slot - u * sw * 0.7 + v * sw * 0.035],
                [(0, 0), (1, 0), (1, 1), (0, 1)], principled("slit", "#0A0908", rough=0.6))
    slit.location += n * 0.0015
    lipm = principled("brass", "#C9A46B", rough=0.3, metal=1.0)
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    lip = bpy.context.active_object
    lip.data.materials.append(lipm)
    lip.matrix_world = frame_matrix(slot - v * sw * 0.07 + n * sw * 0.06, u, v, n) @ Matrix.Diagonal(
        (sw * 1.55, sw * 0.03, sw * 0.12, 1))
    # local Y = out of the panel, local Z = up (image faces up), so local X = -u; mirror UVs to compensate
    strip = strip_object("strip", sw, True, flip_u=True)
    R0 = Matrix((-u, n, (-u).cross(n))).transposed()
    L = sw * STRIP_ASPECT

    def strip_xf(lf):
        e_t = clamp((lf - 10) / 46)
        ext = L * (e_t - 0.8 * math.sin(2 * math.pi * 6 * e_t) / (2 * math.pi * 6))
        droop = -0.06 * e_t
        pos = slot + n * (ext - L) + v * sw * 0.0
        if lf >= 65:
            t = clamp((lf - 65) / 14)
            droop = lerp(-0.06, -1.3, in_quad(t))
            pos = slot + n * 0.002
            if lf >= 76:
                pos = pos + n * 0.03 * sw * (lf - 76) - v * 0.0024 * (lf - 76) ** 2 * (sw / 0.08)
        q = R0.to_quaternion() @ Quaternion((1, 0, 0), droop)
        return pos, q.to_euler()

    key(strip, "location", lambda f: tuple(strip_xf(f)[0]), 3, range(N))
    key(strip, "rotation_euler", lambda f: tuple(strip_xf(f)[1]), 3, range(N))
    key(strip, "hide_render", lambda f: 1.0 if f < 10 else 0.0, 1, range(N), interp=0)
    # soft shadow of the strip on the panel just under the slot
    sh = quad("strip_shadow", [Vector((-0.5, -0.5, 0)), Vector((0.5, -0.5, 0)), Vector((0.5, 0.5, 0)), Vector((-0.5, 0.5, 0))],
              [(0, 0), (1, 0), (1, 1), (0, 1)], shadow_decal("shadow", 0.55, 0.4))

    def sh_xf(f):
        e = clamp((f - 10) / 46) if f < 65 else 1 - clamp((f - 65) / 10)
        c = slot - v * sw * (0.25 + 0.55 * e) + n * 0.002
        return c, e

    key(sh, "location", lambda f: tuple(sh_xf(f)[0]), 3, range(N))
    sh.rotation_euler = frame_matrix(Vector(), u, v, n).to_euler()
    sh.scale = (sw * 1.6, sw * 1.3, 1)
    key(sh, "color", lambda f: (1, 1, 1, sh_xf(f)[1]), 4, range(N))
    area("key", slot + n * 1.2 + v * 1.0 - u * 0.6, slot, 40, "#FFE3C2", size=1.0)
    area("ring", slot + n * 0.9 + v * 0.6 + u * 1.0, slot, 25, "#F4F6FF", size=0.8)
    goal = slot + n * L * 0.55

    def cam_xf(f):
        t = in_out(clamp(f / 60))
        pos = Vector((0, 0, 0)).lerp(goal, 0.12 * t) + Vector((wobble(f, 1, 0.002), 0, wobble(f, 2, 0.0015)))
        aim = goal - v * L * 0.25
        tgt = (Vector((0, 1, 0)) * aim.length).lerp(aim, min(1.0, 0.25 + t))
        return pos, (tgt - pos).to_track_quat("-Z", "Y").to_euler()

    key(cam, "location", lambda f: tuple(cam_xf(f)[0]), 3, range(N))
    key(cam, "rotation_euler", lambda f: tuple(cam_xf(f)[1]), 3, range(N))
    key(cam.data, "lens", lambda f: 12.0 / math.tan(VFOV * lerp(0.62, 0.3, in_out(clamp(f / 60))) / 2), 1, range(N))
    cam.data.dof.use_dof = True
    cam.data.dof.aperture_fstop = 2.8
    key(cam.data, "dof.focus_distance", lambda f: (cam_xf(f)[0] - goal).length, 1, range(N))

# ==========================================================================================
elif A.shot == "guestbook":
    N = 90                                  # global frames 210..299
    settings(N, samples=32, transparent=A.ink_only)
    cam = make_cam()
    if not A.ink_only:
        plate_mesh(A.plate)
    page = fit_plane([(470, 1020, 1000, 1700), (520, 700, 900, 1000)],
                     keep=lambda x, y: True)
    # scale from the real 2x6 strip already glued into the book
    tl, tr_, bl = hit(700, 1193, page), hit(813, 1207, page), hit(640, 1487, page)
    real_w, real_l = (tr_ - tl).length, (bl - tl).length
    sw = (real_w + real_l / 3) / 2
    along = (bl - tl).normalized()          # direction of the real strip's long side ("down" the strip)
    u, v, n = basis(page[1], tr_ - tl)
    LIFT = 0.0
    low_page = fit_plane([(700, 1250, 1000, 1600)])
    up_page = fit_plane([(600, 790, 880, 990)])
    n = low_page[1]
    land = surf(845, 1420, 0.025)
    strip = strip_object("strip", sw, False)
    yax = (-(along - n * along.dot(n))).normalized()    # image "up" = opposite the real strip's long direction
    xax = yax.cross(n).normalized()                      # right-handed (X, Y, Z=n): not mirrored
    Rl = Matrix((xax, yax, n)).transposed()
    rest_q = Rl.to_quaternion() @ Quaternion((0, 0, 1), math.radians(-7))

    def strip_xf(f):
        t = clamp(f / 12)
        h = lerp(land.length * 0.62, 0.0, in_quad(t))
        if 12 <= f < 20:
            h = sw * 0.12 * math.sin(math.pi * (f - 12) / 8) * (1 - (f - 12) / 10)
        wob = 1 - clamp(f / 16)
        q = rest_q @ Quaternion((0, 0, 1), 0.6 * wob) @ Quaternion((1, 0, 0), 0.4 * wob * math.sin(f * 0.9))
        pos = land + n * (h + LIFT) + xax * sw * 0.4 * wob
        return pos, q.to_euler()

    key(strip, "location", lambda f: tuple(strip_xf(f)[0]), 3, range(N))
    key(strip, "rotation_euler", lambda f: tuple(strip_xf(f)[1]), 3, range(N))
    strip.hide_render = A.ink_only
    sh = quad("strip_shadow", [Vector((-0.5, -0.5, 0)), Vector((0.5, -0.5, 0)), Vector((0.5, 0.5, 0)), Vector((-0.5, 0.5, 0))],
              [(0, 0), (1, 0), (1, 1), (0, 1)], shadow_decal("shadow", 0.5, 0.25))
    sh.hide_render = A.ink_only
    sh.location = land + n * (LIFT * 0.8) + (u * 0.15 - v * 0.15) * sw
    sh.rotation_euler = rest_q.to_euler()
    sh.scale = (sw * 1.25, sw * STRIP_ASPECT * 1.08, 1)
    key(sh, "color", lambda f: (1, 1, 1, clamp((f - 4) / 10)), 4, range(N))
    # handwriting on the blank part of the upper page, parallel to the spine
    spine = (hit(1000, 1000, page) - hit(560, 1180, page)).normalized()
    hu = (spine - n * spine.dot(n)).normalized()
    hv = n.cross(hu).normalized()
    if hv.dot(hit(700, 800, page) - hit(700, 1000, page)) < 0:
        hu, hv = -hu, -hv
    ink = ink_reveal("ink", "#24305C")
    lines = []
    for i, (txt, px, py) in enumerate([("Živjeli", 650, 925), ("mladenci!", 672, 992)]):
        cu = bpy.data.curves.new("hw" + str(i), "FONT")
        cu.body = txt
        cu.font = font("Caveat-Medium")
        cu.size = sw * 0.5
        cu.align_x = "LEFT"
        cu.align_y = "BOTTOM_BASELINE"
        cu.offset = sw * 0.004
        cu.materials.append(ink)
        ob = link(bpy.data.objects.new("hw" + str(i), cu))
        nu = up_page[1]
        hu2 = (hu - nu * hu.dot(nu)).normalized()
        ob.matrix_world = frame_matrix(surf(px, py, 0.02), hu2, nu.cross(hu2).normalized(), nu)
        lines.append(ob)
    bpy.context.view_layer.update()
    widths = [ob.dimensions.x / max(ob.scale.x, 1e-6) + sw * 0.05 for ob in lines]
    for ob in lines:
        ob.hide_render = not A.ink_only
    for ob, wdt, (a, b) in zip(lines, widths, [(20, 42), (46, 69)]):
        def rv(f, a=a, b=b, wdt=wdt):
            t = clamp((f - a) / (b - a))
            t = t + 0.03 * math.sin(t * math.pi * 7) * (1 - t) * t * 4
            return (-0.01 + (wdt + 0.02) * clamp(t), 0, 0, 1)
        key(ob, "color", rv, 4, range(N))
    hp = []
    for k in range(90):
        t = 2 * math.pi * k / 89
        x = 16 * math.sin(t) ** 3
        y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
        hp.append((x * sw * 0.012, y * sw * 0.012, 0))
    cu = bpy.data.curves.new("heart", "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = sw * 0.012
    cu.bevel_factor_mapping_end = "SPLINE"
    sp = cu.splines.new("POLY")
    sp.points.add(len(hp) - 1)
    for i, p in enumerate(hp):
        sp.points[i].co = (*p, 1)
    hm = bpy.data.materials.new("heart_ink")
    hm.use_nodes = True
    hm.node_tree.nodes.clear()
    o = hm.node_tree.nodes.new("ShaderNodeOutputMaterial")
    e = hm.node_tree.nodes.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = srgb("#A23A4E")
    e.inputs["Strength"].default_value = 0.9
    hm.node_tree.links.new(e.outputs[0], o.inputs[0])
    cu.materials.append(hm)
    heart = link(bpy.data.objects.new("heart", cu))
    nu = up_page[1]
    hu2 = (hu - nu * hu.dot(nu)).normalized()
    heart.matrix_world = frame_matrix(surf(842, 862, 0.02), hu2, nu.cross(hu2).normalized(), nu)
    key(cu, "bevel_factor_end", lambda f: clamp((f - 72) / 11), 1, range(N))
    heart.hide_render = not A.ink_only
    area("key", land + n * 0.9 + v * 0.3 - u * 0.4, land, 18, "#FFE9CF", size=0.8)

    # camera stays exactly on the original viewpoint; the gentle push-in happens in 2D afterwards so the
    # handwriting (drawn in compositing) stays locked to the page
    def cam_xf(f):
        return Vector((0, 0, 0)), Euler((math.radians(90), 0, 0))
    cam.data.dof.use_dof = True
    cam.data.dof.aperture_fstop = 4.0
    key(cam.data, "dof.focus_distance", lambda f: (cam_xf(f)[0] - land).length, 1, range(N))

# ==========================================================================================
elif A.shot == "phone":
    N = 24                                  # global frames 300..323
    settings(N, samples=40)
    cam = make_cam()
    plate_mesh(A.plate)
    # a small linen-draped table right in front of the camera; the real booth room behind it
    fu, fv, fn = Vector((1, 0, 0)), Vector((0, 1, 0)), Vector((0, 0, 1))
    table_h = 0.8
    top_c = Vector((0.02, 1.08, -0.4))
    linen = principled("linen", "#EEE7DD", rough=0.85)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=96, radius1=0.36, radius2=0.31, depth=table_h)
    for vtx in bm.verts:
        a = math.atan2(vtx.co.y, vtx.co.x)
        k = (0.5 - vtx.co.z / table_h)
        vtx.co.x *= 1 + 0.035 * k * math.sin(9 * a)
        vtx.co.y *= 1 + 0.035 * k * math.sin(9 * a)
    me = bpy.data.meshes.new("table")
    bm.to_mesh(me)
    bm.free()
    me.materials.append(linen)
    for p in me.polygons:
        p.use_smooth = True
    table = link(bpy.data.objects.new("table", me))
    table.matrix_world = frame_matrix(top_c - fn * table_h / 2, fu, fv, fn)
    bvt = table.modifiers.new("bevel", "BEVEL")
    bvt.width = 0.02
    bvt.segments = 4
    bvt.limit_method = "ANGLE"
    # the phone: ivory body, brass dial, handset on its cradle
    S = 1.15
    ivory = principled("ivory", "#EFE4D0", rough=0.18, coat=0.9)
    brass = principled("brass", "#C9A46B", rough=0.25, metal=1.0)
    black = principled("black", "#141210", rough=0.3)
    phone = link(bpy.data.objects.new("phone", None))
    phone.matrix_world = frame_matrix(top_c + fv * 0.02, fu, fv, fn)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=4, radius1=0.15 * S, radius2=0.085 * S, depth=0.075 * S)
    bmesh.ops.rotate(bm, verts=bm.verts, cent=(0, 0, 0), matrix=Matrix.Rotation(math.radians(45), 3, "Z"))
    bmesh.ops.translate(bm, verts=bm.verts, vec=(0, 0, 0.0375 * S))
    me = bpy.data.meshes.new("body")
    bm.to_mesh(me)
    bm.free()
    me.materials.append(ivory)
    for p in me.polygons:
        p.use_smooth = True
    body = link(bpy.data.objects.new("body", me), phone)
    bv = body.modifiers.new("bevel", "BEVEL")
    bv.width = 0.018 * S
    bv.segments = 2
    ss = body.modifiers.new("sub", "SUBSURF")
    ss.levels = ss.render_levels = 2
    tilt = math.atan2((0.15 - 0.085) * math.cos(math.radians(45)), 0.075)
    dial = link(bpy.data.objects.new("dial", None), phone)
    dial.location = (0, -((0.15 + 0.085) / 2) * math.cos(math.radians(45)) * S * 0.93, 0.036 * S)
    dial.rotation_euler = (math.pi / 2 - tilt, 0, 0)
    for nm, r, dz, mat in (("dial_plate", 0.043, 0.0, brass), ("dial_face", 0.027, 0.003, ivory)):
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=48, radius1=r * S, radius2=r * S, depth=0.006)
        me = bpy.data.meshes.new(nm)
        bm.to_mesh(me)
        bm.free()
        me.materials.append(mat)
        ob = link(bpy.data.objects.new(nm, me), dial)
        ob.location = (0, 0, dz)
    for k in range(10):
        a = math.radians(-60 + k * 28)
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=12, v_segments=6, radius=0.0065 * S)
        me = bpy.data.meshes.new(f"hole{k}")
        bm.to_mesh(me)
        bm.free()
        me.materials.append(black)
        ob = link(bpy.data.objects.new(f"hole{k}", me), dial)
        ob.location = (0.034 * S * math.cos(a), 0.034 * S * math.sin(a), 0.004)
        ob.scale = (1, 1, 0.4)
    handset = link(bpy.data.objects.new("handset", None))
    pts = [((-0.11 + 0.22 * k / 24) * S, 0, (0.03 * (1 - ((-0.11 + 0.22 * k / 24) / 0.11) ** 2) + 0.008) * S) for k in range(25)]
    cu = bpy.data.curves.new("handle", "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = 0.016 * S
    cu.bevel_resolution = 6
    cu.use_fill_caps = True
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for i, p in enumerate(pts):
        sp.points[i].co = (*p, 1)
    cu.materials.append(ivory)
    link(bpy.data.objects.new("handle", cu), handset)
    for sx in (-1, 1):
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=32, radius1=0.032 * S, radius2=0.026 * S, depth=0.035 * S)
        me = bpy.data.meshes.new(f"cup{sx}")
        bm.to_mesh(me)
        bm.free()
        me.materials.append(ivory)
        for p in me.polygons:
            p.use_smooth = True
        ob = link(bpy.data.objects.new(f"cup{sx}", me), handset)
        ob.location = (sx * 0.11 * S, 0, -0.004 * S)
    PM = phone.matrix_world.copy()
    rest = PM @ Matrix.Translation((0, 0, 0.087 * S))
    camgoal = Vector((0.05, 0.12, -0.02))

    def hs_xf(f):
        t = in_out(clamp((f - 6) / 12))
        loc = rest.translation.lerp(camgoal, t) + fn * 0.12 * math.sin(math.pi * t)
        q = rest.to_quaternion().slerp((rest.to_quaternion() @ Quaternion((1, 0, 0), -1.2) @ Quaternion((0, 0, 1), 0.5)), t)
        return loc, q.to_euler()

    key(handset, "location", lambda f: tuple(hs_xf(f)[0]), 3, range(N))
    key(handset, "rotation_euler", lambda f: tuple(hs_xf(f)[1]), 3, range(N))
    tc = top_c + fv * 0.02
    area("key", tc + fn * 1.4 - fu * 0.6 - fv * 0.4, tc, 28, "#FFE3C4", size=1.4)
    area("rim", tc + fn * 0.9 + fu * 0.8 + fv * 1.2, tc, 30, "#F2F4FF", size=1.0)
    sh = quad("phone_shadow", [Vector((-0.5, -0.5, 0)), Vector((0.5, -0.5, 0)), Vector((0.5, 0.5, 0)), Vector((-0.5, 0.5, 0))],
              [(0, 0), (1, 0), (1, 1), (0, 1)], shadow_decal("shadow", 0.55, 0.45))
    sh.matrix_world = frame_matrix(tc + fn * 0.001, fu, fv, fn) @ Matrix.Diagonal((0.36 * S, 0.34 * S, 1, 1))
    sh.color = (1, 1, 1, 1)

    def cam_xf(f):
        t = smooth(clamp(f / N))
        pos = Vector((0.0, -0.05 + 0.08 * t, -0.03 * t)) + Vector((wobble(f, 5, 0.003), 0, wobble(f, 6, 0.002)))
        return pos, (tc + fn * 0.06 - pos).to_track_quat("-Z", "Y").to_euler()

    key(cam, "location", lambda f: tuple(cam_xf(f)[0]), 3, range(N))
    key(cam, "rotation_euler", lambda f: tuple(cam_xf(f)[1]), 3, range(N))
    cam.data.lens = 12.0 / math.tan(math.radians(40) / 2)
    cam.data.dof.use_dof = True
    cam.data.dof.aperture_fstop = 1.2
    key(cam.data, "dof.focus_distance", lambda f: (cam_xf(f)[0] - tc).length, 1, range(N))

bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(A.out))
print("SAVED", A.out)
