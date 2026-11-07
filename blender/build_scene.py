"""Builds the whole Gran Vía photobooth diorama in Blender, animates it and saves a .blend.

Run with Blender's Python module (bpy) or inside Blender:
    python blender/build_scene.py -- --tex /path/to/textures --out scene.blend

Everything here is modeled procedurally: the characters, the booth, the flower arch
(each rose is a generated spiral of petals), the guestbook, the rotary phone, the
string lights. Animation is written as Python functions of the frame and then baked
to ordinary keyframes, so the saved .blend can be opened and scrubbed in Blender.
"""
import argparse
import math
import os
import random
import sys
from functools import lru_cache

import bpy  # noqa: I001  (bpy must be imported before bmesh)
import bmesh
from mathutils import Euler, Matrix, Quaternion, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from anim import (BEAT, COUNTDOWN, END, FLASHES, RENDER_FRAMES, S1A, S1B, S2, S3, S4, S5,  # noqa: E402
                  clamp, in_cubic, in_out, in_out_expo, in_quad, lerp, linear, out_back,
                  out_back_soft, out_cubic, pulse, smooth, tween, wobble)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT_DIR = os.path.join(REPO, "assets", "fonts")

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
ap = argparse.ArgumentParser()
ap.add_argument("--tex", default="/tmp/granvia/tex")
ap.add_argument("--out", default="/tmp/granvia/granvia_photobooth.blend")
ARGS = ap.parse_args(argv)
TEX = os.path.abspath(ARGS.tex)

random.seed(877)

# --------------------------------------------------------------------------------------
# layout (metres, Z up). The booth looks along +Y at the flower arch.
# --------------------------------------------------------------------------------------
ARCH_Y = 1.78
BOOTH_Y = -1.55
BOOTH_FRONT = BOOTH_Y + 0.21          # front face of the cabinet (faces +Y)
GB = Vector((-2.0, 0.15, 0.0))        # guestbook table
PH = Vector((2.02, 0.12, 0.0))        # phone table
GB_TOP = 0.62
PH_TOP = 0.60
PHONE_S = 1.35                        # phone scale (chibi world)
STRIP_W, STRIP_L = 0.09, 0.27         # 2x6 photo strip, scaled for the toy world

FPS = 30


def sf(f):
    """0-based frame -> Blender scene frame."""
    return f + 1


# --------------------------------------------------------------------------------------
# scene reset
# --------------------------------------------------------------------------------------
bpy.ops.wm.read_factory_settings(use_empty=True)
SCENE = bpy.context.scene
SCENE.name = "GranVia"
COLL = SCENE.collection


def srgb(h, a=1.0):
    h = h.lstrip("#")
    c = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return tuple(x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c) + (a,)


FONTS = {}


def font(name):
    if name not in FONTS:
        FONTS[name] = bpy.data.fonts.load(os.path.join(FONT_DIR, name + ".ttf"), check_existing=True)
    return FONTS[name]


# --------------------------------------------------------------------------------------
# materials
# --------------------------------------------------------------------------------------
def _mat(name):
    m = bpy.data.materials.new(name)
    try:
        m.use_nodes = True
    except Exception:
        pass
    return m


def principled(name, color, rough=0.5, metal=0.0, sheen=0.0, coat=0.0, sss=0.0,
               emit=None, emit_str=0.0, spec=0.5, transmission=0.0):
    m = _mat(name)
    p = m.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = srgb(color) if isinstance(color, str) else color
    p.inputs["Roughness"].default_value = rough
    p.inputs["Metallic"].default_value = metal
    p.inputs["Sheen Weight"].default_value = sheen
    p.inputs["Coat Weight"].default_value = coat
    p.inputs["Specular IOR Level"].default_value = spec
    p.inputs["Transmission Weight"].default_value = transmission
    if sss:
        p.inputs["Subsurface Weight"].default_value = sss
        p.inputs["Subsurface Radius"].default_value = (0.8, 0.35, 0.25)
        p.inputs["Subsurface Scale"].default_value = 0.02
    if emit:
        p.inputs["Emission Color"].default_value = srgb(emit)
        p.inputs["Emission Strength"].default_value = emit_str
    return m


def emission(name, color, strength):
    m = _mat(name)
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = srgb(color)
    em.inputs["Strength"].default_value = strength
    nt.links.new(em.outputs[0], out.inputs[0])
    return m


def fading_emission(name, color, strength):
    """Emission whose opacity comes from the object's color alpha (keyframed per object)."""
    m = _mat(name)
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = srgb(color)
    em.inputs["Strength"].default_value = strength
    tr = nt.nodes.new("ShaderNodeBsdfTransparent")
    oi = nt.nodes.new("ShaderNodeObjectInfo")
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(oi.outputs["Alpha"], mix.inputs[0])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(em.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs[0])
    return m


def ink_reveal(name, color):
    """Ink that is revealed left->right: visible where object-space X < object color R."""
    m = _mat(name)
    nt = m.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = srgb(color)
    bsdf.inputs["Roughness"].default_value = 0.35
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
    mr.inputs["To Min"].default_value = 0.0
    mr.inputs["To Max"].default_value = 1.0
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(tc.outputs["Object"], sep.inputs[0])
    nt.links.new(oi.outputs["Color"], sepc.inputs[0])
    nt.links.new(sepc.outputs[0], sub.inputs[0])       # threshold
    nt.links.new(sep.outputs[0], sub.inputs[1])        # - x
    nt.links.new(sub.outputs[0], mr.inputs["Value"])
    nt.links.new(mr.outputs[0], mix.inputs[0])
    nt.links.new(tr.outputs[0], mix.inputs[1])
    nt.links.new(bsdf.outputs[0], mix.inputs[2])
    nt.links.new(mix.outputs[0], out.inputs[0])
    return m


def image_material(name, path, rough=0.35, emit_str=0.0, coat=0.3, sequence_len=0):
    m = _mat(name)
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]
    tex = nt.nodes.new("ShaderNodeTexImage")
    img = bpy.data.images.load(path, check_existing=True)
    if sequence_len:
        img.source = "SEQUENCE"
        tex.image_user.frame_duration = sequence_len
        tex.image_user.frame_start = 1
        tex.image_user.frame_offset = 0
        tex.image_user.use_auto_refresh = True
    tex.image = img
    tex.interpolation = "Cubic"
    if emit_str:
        p.inputs["Base Color"].default_value = srgb("#050505")
        nt.links.new(tex.outputs["Color"], p.inputs["Emission Color"])
        p.inputs["Emission Strength"].default_value = emit_str
    else:
        nt.links.new(tex.outputs["Color"], p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = rough
    p.inputs["Coat Weight"].default_value = coat
    m.cycles.emission_sampling = "NONE"
    return m


def wood_material(name, light="#DCC29C", dark="#C4A177", scale=7.0):
    m = _mat(name)
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Scale"].default_value = (1.0, 1.0, 9.0)
    wave = nt.nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = "Z"
    wave.inputs["Scale"].default_value = scale
    wave.inputs["Distortion"].default_value = 6.0
    wave.inputs["Detail"].default_value = 2.0
    wave.inputs["Detail Scale"].default_value = 1.5
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].color = srgb(dark)
    ramp.color_ramp.elements[1].color = srgb(light)
    ramp.color_ramp.elements[0].position = 0.25
    nt.links.new(tc.outputs["Object"], mp.inputs[0])
    nt.links.new(mp.outputs[0], wave.inputs[0])
    nt.links.new(wave.outputs["Fac"], ramp.inputs[0])
    nt.links.new(ramp.outputs[0], p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.45
    return m


def stone_material(name):
    """Pale Dalmatian stone pavers, like the terraces in the real event photos."""
    m = _mat(name)
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]
    tc = nt.nodes.new("ShaderNodeTexCoord")
    br = nt.nodes.new("ShaderNodeTexBrick")
    br.inputs["Scale"].default_value = 2.2
    br.inputs["Mortar Size"].default_value = 0.012
    br.inputs["Mortar Smooth"].default_value = 0.4
    br.inputs["Color1"].default_value = srgb("#9E9283")
    br.inputs["Color2"].default_value = srgb("#8A7F72")
    br.inputs["Mortar"].default_value = srgb("#5C544B")
    br.inputs["Brick Width"].default_value = 0.5
    br.inputs["Row Height"].default_value = 0.25
    br.offset = 0.5
    noise = nt.nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 18.0
    mix = nt.nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.inputs["Factor"].default_value = 0.25
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.35
    nt.links.new(tc.outputs["Object"], br.inputs["Vector"])
    nt.links.new(tc.outputs["Object"], noise.inputs["Vector"])
    nt.links.new(br.outputs["Color"], mix.inputs["A"])
    nt.links.new(noise.outputs["Color"], mix.inputs["B"])
    nt.links.new(mix.outputs["Result"], p.inputs["Base Color"])
    nt.links.new(br.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs[0], p.inputs["Normal"])
    p.inputs["Roughness"].default_value = 0.75
    return m


def rose_material(name, palette):
    """One material for all roses: Object Info 'Random' picks the shade per flower."""
    m = _mat(name)
    nt = m.node_tree
    p = nt.nodes["Principled BSDF"]
    oi = nt.nodes.new("ShaderNodeObjectInfo")
    ramp = nt.nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.interpolation = "CONSTANT"
    els = ramp.color_ramp.elements
    els[0].color = srgb(palette[0])
    els[0].position = 0.0
    els[1].color = srgb(palette[1])
    els[1].position = 1.0 / len(palette)
    for i, c in enumerate(palette[2:], start=2):
        e = els.new(i / len(palette))
        e.color = srgb(c)
    nt.links.new(oi.outputs["Random"], ramp.inputs[0])
    nt.links.new(ramp.outputs[0], p.inputs["Base Color"])
    p.inputs["Roughness"].default_value = 0.55
    p.inputs["Sheen Weight"].default_value = 0.2
    return m


M = {}
M["skin_a"] = principled("skin_bride", "#F0C6A9", rough=0.55, spec=0.35)
M["skin_b"] = principled("skin_groom", "#E2B28C", rough=0.55, spec=0.35)
M["skin_c"] = principled("skin_friend", "#EDBF9D", rough=0.55, spec=0.35)
M["eye"] = principled("eye", "#1B1412", rough=0.12, coat=1.0)
M["catch"] = emission("catchlight", "#FFFFFF", 6.0)
M["blush"] = principled("blush", "#EE9E98", rough=0.7)
M["mouth"] = principled("mouth", "#4A2622", rough=0.5)
M["mouth_in"] = principled("mouth_in", "#6E2B30", rough=0.5)
M["hair_bride"] = principled("hair_bride", "#3B261C", rough=0.5, spec=0.35)
M["hair_groom"] = principled("hair_groom", "#241A15", rough=0.5, spec=0.35)
M["hair_friend"] = principled("hair_friend", "#8A4126", rough=0.5, spec=0.35)
M["dress"] = principled("dress_white", "#EDE7DF", rough=0.55, sheen=0.15)
M["veil"] = principled("veil", "#FFFFFF", rough=0.3, transmission=0.55, sheen=0.5)
M["sash"] = principled("sash", "#E3BDB6", rough=0.5, sheen=0.15)
M["suit"] = principled("suit", "#26304A", rough=0.62, sheen=0.1, spec=0.35)
M["shirt"] = principled("shirt", "#F7F5F1", rough=0.5)
M["black"] = principled("gloss_black", "#121212", rough=0.22, coat=0.4)
M["teal"] = principled("dress_teal", "#3C7274", rough=0.58, sheen=0.12, spec=0.4)
M["tan"] = principled("shoe_tan", "#C79C6D", rough=0.35)
M["gold"] = principled("gold", "#D1A862", rough=0.28, metal=1.0)
M["satin"] = principled("satin", "#ECE6DE", rough=0.45, sheen=0.25)
M["linen"] = principled("linen", "#BFA696", rough=0.85, sheen=0.2)
M["cream"] = principled("cream_plaster", "#EAE0D2", rough=0.8)
M["paper"] = principled("paper", "#E4DACB", rough=0.85)
M["cover"] = principled("book_cover", "#C3A79A", rough=0.75, sheen=0.15)
M["ceramic"] = principled("ceramic", "#EFE7DA", rough=0.25, coat=0.5)
M["phone"] = principled("phone_ivory", "#EDE3D1", rough=0.2, coat=0.8)
M["leaf"] = principled("leaf", "#8E9F7E", rough=0.55, sheen=0.3)
M["bush"] = principled("bush", "#22372A", rough=0.7, sheen=0.4)
M["pole"] = principled("pole", "#2B2622", rough=0.5)
M["wire"] = principled("wire", "#1C1A18", rough=0.5)
M["lens_glass"] = principled("lens_glass", "#0B111B", rough=0.04, coat=1.0)
M["ink"] = principled("logo_ink", "#2A2522", rough=0.6)
M["wood"] = wood_material("birch_wood")
M["wood_dark"] = wood_material("table_wood", "#B48E66", "#9C7752", scale=5)
M["stone"] = stone_material("stone_pavers")
M["roses_arch"] = rose_material("roses_arch", ["#FBF6F1", "#F3D5D3", "#F7E6E1", "#E8B6B9", "#FFFFFF", "#F1CFC8"])
M["roses_white"] = rose_material("roses_white", ["#FFFFFF", "#F8F1EA", "#F4E4DE"])
M["roses_blush"] = rose_material("roses_blush", ["#F1C9C6", "#E6AEB3", "#F7E2DC"])
M["neon"] = emission("neon_warm", "#FFE2B2", 14.0)
M["ring"] = emission("ringlight", "#FFF6EA", 5.0)
M["bulb"] = emission("bulb", "#FFB467", 22.0)
M["bokeh"] = emission("bokeh", "#FFC27C", 45.0)
M["bokeh2"] = emission("bokeh2", "#FFE0A8", 35.0)
M["heart"] = principled("heart", "#E8707F", rough=0.3, coat=0.6, emit="#E8707F", emit_str=0.6)
M["glow"] = fading_emission("sound_ring", "#F6D9A8", 4.0)
M["writing"] = ink_reveal("handwriting_ink", "#2E3550")
M["heart_ink"] = principled("heart_ink", "#B94A5C", rough=0.35)
M["card"] = principled("sign_card", "#FBF7F0", rough=0.6)
M["sign_ink"] = principled("sign_ink", "#2A2522", rough=0.5)
# decorative emitters are seen directly but don't need to be sampled as lights (big noise/speed win)
for _k in ("catch", "neon", "ring", "bulb", "bokeh", "bokeh2", "glow"):
    M[_k].cycles.emission_sampling = "NONE"


# --------------------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------------------
def link(ob, parent=None):
    COLL.objects.link(ob)
    if parent is not None:
        ob.parent = parent
    return ob


def empty(name, loc=(0, 0, 0), parent=None, rot=(0, 0, 0)):
    ob = bpy.data.objects.new(name, None)
    ob.empty_display_size = 0.05
    ob.location = loc
    ob.rotation_euler = rot
    return link(ob, parent)


def finish_mesh(name, bm, mat, smooth=True, parent=None, loc=(0, 0, 0), rot=(0, 0, 0),
                scale=(1, 1, 1), subsurf=0, bevel=0.0, bevel_seg=3):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    if smooth:
        me.shade_smooth()
    ob = bpy.data.objects.new(name, me)
    if mat is not None:
        me.materials.append(mat)
    ob.location = loc
    ob.rotation_euler = rot
    ob.scale = scale
    if bevel:
        b = ob.modifiers.new("bevel", "BEVEL")
        b.width = bevel
        b.segments = bevel_seg
        b.limit_method = "ANGLE"
        b.harden_normals = True
    if subsurf:
        s = ob.modifiers.new("subsurf", "SUBSURF")
        s.levels = subsurf
        s.render_levels = subsurf
    return link(ob, parent)


def sphere(name, r, mat, seg=40, rings=20, **kw):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=rings, radius=r)
    return finish_mesh(name, bm, mat, **kw)


def cone(name, r1, r2, depth, mat, seg=48, cap=True, **kw):
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=cap, cap_tris=False, segments=seg, radius1=r1, radius2=r2, depth=depth)
    return finish_mesh(name, bm, mat, **kw)


def box(name, size, mat, bevel=0.0, smooth=True, **kw):
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bmesh.ops.scale(bm, vec=Vector(size), verts=bm.verts)
    return finish_mesh(name, bm, mat, smooth=smooth, bevel=bevel, **kw)


def from_pydata(name, verts, faces, mat, uvs=None, smooth=True, **kw):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    if uvs is not None:
        uvl = me.uv_layers.new(name="UVMap")
        for poly in me.polygons:
            for li in poly.loop_indices:
                uvl.data[li].uv = uvs[me.loops[li].vertex_index]
    me.update()
    bm = bmesh.new()
    bm.from_mesh(me)
    bpy.data.meshes.remove(me)
    return finish_mesh(name, bm, mat, smooth=smooth, **kw)


def torus(name, R, r, mat, seg=48, rseg=16, arc=2 * math.pi, **kw):
    closed = abs(arc - 2 * math.pi) < 1e-6
    n = seg if closed else seg + 1
    verts, faces = [], []
    for i in range(n):
        a = arc * i / seg
        for j in range(rseg):
            b = 2 * math.pi * j / rseg
            verts.append(((R + r * math.cos(b)) * math.cos(a), (R + r * math.cos(b)) * math.sin(a), r * math.sin(b)))
    for i in range(seg):
        i2 = (i + 1) % n
        for j in range(rseg):
            j2 = (j + 1) % rseg
            faces.append((i * rseg + j, i2 * rseg + j, i2 * rseg + j2, i * rseg + j2))
    return from_pydata(name, verts, faces, mat, **kw)


def curve_obj(name, pts, bevel, mat, parent=None, loc=(0, 0, 0), rot=(0, 0, 0), res=6, caps=True):
    cu = bpy.data.curves.new(name, "CURVE")
    cu.dimensions = "3D"
    cu.bevel_depth = bevel
    cu.bevel_resolution = res
    cu.use_fill_caps = caps
    sp = cu.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for i, p in enumerate(pts):
        sp.points[i].co = (p[0], p[1], p[2], 1.0)
    cu.materials.append(mat)
    ob = bpy.data.objects.new(name, cu)
    ob.location = loc
    ob.rotation_euler = rot
    return link(ob, parent)


def text_obj(name, body, fnt, size, mat, align="CENTER", extrude=0.0, parent=None, loc=(0, 0, 0),
             rot=(0, 0, 0), spacing=1.0, valign="CENTER", bevel=0.0):
    cu = bpy.data.curves.new(name, "FONT")
    cu.body = body
    cu.font = font(fnt)
    cu.size = size
    cu.align_x = align
    cu.align_y = valign
    cu.extrude = extrude
    cu.bevel_depth = bevel
    cu.space_character = spacing
    cu.resolution_u = 4
    cu.materials.append(mat)
    ob = bpy.data.objects.new(name, cu)
    ob.location = loc
    ob.rotation_euler = rot
    return link(ob, parent)


def look_rot(src, dst, roll=0.0):
    d = Vector(dst) - Vector(src)
    q = d.to_track_quat("-Z", "Y")
    if roll:
        q = q @ Quaternion((0, 0, 1), roll)
    return q.to_euler()


FACE_PLUS_Y = (math.radians(90), 0, math.radians(180))   # text/plane facing +Y, upright
FACE_MINUS_Y = (math.radians(90), 0, 0)                  # facing -Y, upright


# --------------------------------------------------------------------------------------
# procedural rose: a spiral ribbon of petals that opens outward
# --------------------------------------------------------------------------------------
def rose_mesh(name, turns=3.2, segs=110, flare=0.55, seed=0):
    rnd = random.Random(seed)
    verts, faces = [], []
    rows = 4
    for i in range(segs + 1):
        t = i / segs
        ang = t * turns * 2 * math.pi
        r = 0.12 + 0.88 * t ** 0.75
        h = 0.95 - 0.35 * t
        wav = 1 + 0.06 * math.sin(ang * 2.5 + rnd.random())
        for k in range(rows):
            v = k / (rows - 1)
            rr = (r + flare * t * v ** 1.6) * wav
            z = h * v - 0.15 * t * v * v + 0.02 * math.sin(ang * 5) * v
            verts.append((rr * math.cos(ang), rr * math.sin(ang), z))
    for i in range(segs):
        for k in range(rows - 1):
            a = i * rows + k
            faces.append((a, a + rows, a + rows + 1, a + 1))
    # little green calyx disc underneath
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    me.shade_smooth()
    return me


ROSE_MESHES = [rose_mesh("rose_a", 3.2, seed=1), rose_mesh("rose_b", 2.6, 96, 0.7, seed=2),
               rose_mesh("rose_c", 3.8, 130, 0.45, seed=3)]


def leaf_mesh(name):
    verts, faces = [], []
    n = 10
    for i in range(n + 1):
        t = i / n
        w = math.sin(math.pi * t) ** 0.8 * 0.38
        y = t
        verts += [(-w, y, 0.08 * math.sin(math.pi * t) + 0.12 * w), (0, y, 0.0), (w, y, 0.08 * math.sin(math.pi * t) + 0.12 * w)]
    for i in range(n):
        a = i * 3
        faces += [(a, a + 3, a + 4, a + 1), (a + 1, a + 4, a + 5, a + 2)]
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.update()
    me.shade_smooth()
    return me


LEAF = leaf_mesh("leaf")


def rose(name, loc, normal, size, mat, parent=None, spin=None):
    ob = bpy.data.objects.new(name, random.choice(ROSE_MESHES))
    if not ob.data.materials:
        ob.data.materials.append(mat)
    ob.material_slots[0].link = "OBJECT"
    ob.material_slots[0].material = mat
    q = Vector(normal).normalized().to_track_quat("Z", "Y")
    q = q @ Quaternion((0, 0, 1), spin if spin is not None else random.uniform(0, 6.28))
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = q
    ob.location = loc
    ob.scale = (size, size, size * 0.9)
    return link(ob, parent)


def leaf(name, loc, direction, size, parent=None, mat=None):
    ob = bpy.data.objects.new(name, LEAF)
    if not LEAF.materials:
        LEAF.materials.append(M["leaf"])
    q = Vector(direction).normalized().to_track_quat("Y", "Z")
    q = q @ Quaternion((0, 1, 0), random.uniform(-0.6, 0.6))
    ob.rotation_mode = "QUATERNION"
    ob.rotation_quaternion = q
    ob.location = loc
    ob.scale = (size, size, size)
    return link(ob, parent)


def flower_cluster(prefix, center, n, radius, mat, size=(0.022, 0.04), up=(0, 0, 1), parent=None, leaves=4):
    c = Vector(center)
    upv = Vector(up).normalized()
    for i in range(n):
        off = Vector((random.gauss(0, 1), random.gauss(0, 1), random.gauss(0, 1))).normalized() * radius * random.uniform(0.2, 1.0)
        off -= upv * min(0.0, off.dot(upv)) * 1.5
        nrm = (off.normalized() * 0.7 + upv).normalized()
        rose(f"{prefix}_rose{i}", c + off, nrm, random.uniform(*size), mat, parent=parent)
    for i in range(leaves):
        d = Vector((random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-0.2, 0.4))).normalized()
        leaf(f"{prefix}_leaf{i}", c + d * radius * 0.5, d, random.uniform(0.07, 0.11), parent=parent)


# --------------------------------------------------------------------------------------
# animation registry -> baked keyframes
# --------------------------------------------------------------------------------------
ANIM = []


def animate(idb, path, func, n=3, constant=False):
    ANIM.append((idb, path, n, func, constant))


def _set_static(idb, path, i, n, v):
    if "." in path:
        owner_path, attr = path.rsplit(".", 1)
        owner = idb.path_resolve(owner_path)
    else:
        owner, attr = idb, path
    if n == 1:
        cur = getattr(owner, attr)
        setattr(owner, attr, type(cur)(v) if isinstance(cur, (bool, int)) else v)
    else:
        getattr(owner, attr)[i] = v


def bake():
    for idb, path, n, func, constant in ANIM:
        vals = [func(f) for f in range(RENDER_FRAMES)]
        if path == "rotation_euler":   # keep eulers continuous so motion blur never spins
            prev = None
            fixed = []
            for v in vals:
                e = Euler(v)
                if prev is not None:
                    e.make_compatible(prev)
                fixed.append(tuple(e))
                prev = e
            vals = fixed
        if n == 1:
            vals = [(float(v),) for v in vals]
        for i in range(n):
            series = [float(v[i]) for v in vals]
            if max(series) - min(series) < 1e-7:
                _set_static(idb, path, i, n, series[0])
                continue
            if idb.animation_data is None:
                idb.animation_data_create()
            if idb.animation_data.action is None:
                idb.animation_data.action = bpy.data.actions.new(idb.name + "_anim")
            fc = idb.animation_data.action.fcurve_ensure_for_datablock(idb, path, index=i if n > 1 else 0)
            kp = fc.keyframe_points
            kp.add(len(series))
            co = []
            for f, v in enumerate(series):
                co += [sf(f), v]
            kp.foreach_set("co", co)
            interp = 0 if constant else 1  # CONSTANT / LINEAR
            kp.foreach_set("interpolation", [interp] * len(series))
            fc.update()


def visible_during(ob, *ranges):
    def vis(f):
        return 0.0 if any(a <= f < b for a, b in ranges) else 1.0
    animate(ob, "hide_render", vis, n=1, constant=True)
    for ch in ob.children_recursive:
        animate(ch, "hide_render", vis, n=1, constant=True)


# --------------------------------------------------------------------------------------
# world, platform, environment
# --------------------------------------------------------------------------------------
world = bpy.data.worlds.new("night")
SCENE.world = world
world.use_nodes = True
bg = world.node_tree.nodes["Background"]
# deep night gradient: plum-navy above, a warmer glow toward the horizon
wnt = world.node_tree
_tc = wnt.nodes.new("ShaderNodeTexCoord")
_sep = wnt.nodes.new("ShaderNodeSeparateXYZ")
_ramp = wnt.nodes.new("ShaderNodeValToRGB")
_ramp.color_ramp.elements[0].position = 0.0
_ramp.color_ramp.elements[0].color = srgb("#1A1418")
_ramp.color_ramp.elements[1].position = 0.45
_ramp.color_ramp.elements[1].color = srgb("#0B0C15")
wnt.links.new(_tc.outputs["Generated"], _sep.inputs[0])
_mr = wnt.nodes.new("ShaderNodeMapRange")
_mr.inputs["From Min"].default_value = 0.5
_mr.inputs["From Max"].default_value = 1.0
wnt.links.new(_sep.outputs["Z"], _mr.inputs["Value"])
wnt.links.new(_mr.outputs[0], _ramp.inputs[0])
wnt.links.new(_ramp.outputs[0], bg.inputs["Color"])
bg.inputs["Strength"].default_value = 1.0

# platform: stone top + cream plaster rim
bm = bmesh.new()
bmesh.ops.create_circle(bm, cap_ends=True, segments=128, radius=3.6)
finish_mesh("platform_top", bm, M["stone"], smooth=False)
cone("platform_side", 3.6, 3.6, 0.32, M["cream"], seg=128, cap=False, loc=(0, 0, -0.16))
bm = bmesh.new()
bmesh.ops.create_circle(bm, cap_ends=True, segments=128, radius=3.6)
finish_mesh("platform_bottom", bm, M["cream"], smooth=False, loc=(0, 0, -0.32))
torus("platform_lip", 3.6, 0.025, M["gold"], seg=160, rseg=8, loc=(0, 0, 0.0))

# bushes behind the arch
for i, (ang, s) in enumerate([(32, 0.55), (52, 0.7), (72, 0.62), (90, 0.75), (108, 0.66), (128, 0.72), (148, 0.58),
                               (168, 0.5), (12, 0.48)]):
    a = math.radians(ang)
    r = 3.05 + random.uniform(-0.1, 0.15)
    bm = bmesh.new()
    bmesh.ops.create_icosphere(bm, subdivisions=4, radius=s)
    ob = finish_mesh(f"bush{i}", bm, M["bush"], loc=(r * math.cos(a), r * math.sin(a), s * 0.55), scale=(1, 0.85, 0.9))
    tex = bpy.data.textures.new(f"bushtex{i}", "CLOUDS")
    tex.noise_scale = 0.18
    d = ob.modifiers.new("disp", "DISPLACE")
    d.texture = tex
    d.strength = 0.16
    d.mid_level = 0.5
    # a few white blossoms on the hedge
    for k in range(5):
        th = random.uniform(0, 6.28)
        p = Vector((r * math.cos(a), r * math.sin(a), s * 0.55)) + Vector((math.cos(th), math.sin(th), random.uniform(-0.2, 0.7))).normalized() * s * 0.95
        rose(f"bush{i}_bloom{k}", p, p - Vector((r * math.cos(a), r * math.sin(a), s * 0.4)), random.uniform(0.022, 0.032), M["roses_white"])

# poles + string lights (catenaries) + far bokeh
POLE_ANGLES = [25, 75, 105, 155, 205, 335]
POLE_R = 3.38
POLE_H = 2.65
pole_top = {}
for ang in POLE_ANGLES:
    a = math.radians(ang)
    x, y = POLE_R * math.cos(a), POLE_R * math.sin(a)
    cone(f"pole{ang}", 0.022, 0.018, POLE_H, M["pole"], seg=12, loc=(x, y, POLE_H / 2))
    sphere(f"pole{ang}_cap", 0.03, M["gold"], seg=12, rings=8, loc=(x, y, POLE_H))
    pole_top[ang] = Vector((x, y, POLE_H - 0.05))

STRANDS = [(25, 75, 0.3), (75, 105, 0.22), (105, 155, 0.3), (155, 205, 0.35), (335, 25, 0.3),
           (75, 205, 0.55), (105, 335, 0.55)]
bulb_i = 0
for a0, a1, sag in STRANDS:
    p0, p1 = pole_top[a0], pole_top[a1]
    length = (p1 - p0).length
    n = max(8, int(length / 0.27))
    pts = []
    for k in range(n + 1):
        t = k / n
        p = p0.lerp(p1, t)
        p.z -= sag * 4 * t * (1 - t)
        pts.append(p)
    curve_obj(f"wire_{a0}_{a1}", pts, 0.0035, M["wire"], res=2)
    for k in range(1, n):
        p = pts[k] - Vector((0, 0, 0.035))
        sphere(f"bulb{bulb_i}", 0.024, M["bulb"], seg=10, rings=6, loc=p)
        bulb_i += 1

for i in range(46):
    ang = random.uniform(math.radians(10), math.radians(170)) if i < 34 else random.uniform(math.radians(190), math.radians(350))
    dist = random.uniform(8.5, 15.0)
    p = (dist * math.cos(ang), dist * math.sin(ang), random.uniform(0.2, 3.8))
    sphere(f"bokeh{i}", random.uniform(0.04, 0.07), M["bokeh"] if i % 3 else M["bokeh2"], seg=10, rings=6, loc=p)

# --------------------------------------------------------------------------------------
# the flower arch with the "Better Together" neon
# --------------------------------------------------------------------------------------
ARCH_R = 0.86
POST_H = 1.18
arch_root = empty("arch", (0, ARCH_Y, 0))
for sx in (-1, 1):
    cone(f"arch_post{sx}", 0.02, 0.02, POST_H, M["gold"], seg=16, loc=(sx * ARCH_R, 0, POST_H / 2), parent=arch_root)
    # satin drape around the post: folds that pool on the floor
    verts, faces = [], []
    rings, seg = 26, 56
    for i in range(rings + 1):
        v = i / rings
        z = v * 1.32
        base = 0.055 + 0.27 * (1 - v) ** 2.4
        for j in range(seg):
            th = 2 * math.pi * j / seg
            fold = 1 + (0.2 * (1 - v) + 0.05) * math.sin(7 * th + v * 2.5 + sx) + 0.06 * math.sin(13 * th - v * 4)
            r = base * fold
            verts.append((sx * ARCH_R + r * math.cos(th), r * math.sin(th) * 0.85, z))
    for i in range(rings):
        for j in range(seg):
            j2 = (j + 1) % seg
            faces.append((i * seg + j, i * seg + j2, (i + 1) * seg + j2, (i + 1) * seg + j))
    from_pydata(f"arch_drape{sx}", verts, faces, M["satin"], parent=arch_root, subsurf=1)
semi = [(ARCH_R * math.cos(a), 0, POST_H + ARCH_R * math.sin(a)) for a in [math.pi * k / 64 for k in range(65)]]
curve_obj("arch_ring", semi, 0.02, M["gold"], parent=arch_root)

n_roses = 0
for k in range(230):
    t = random.uniform(0.02, 0.98)
    if random.random() < 0.35:
        t = random.uniform(0.25, 0.75)  # denser on the crown
    a = math.pi * t
    radial = Vector((math.cos(a), 0, math.sin(a)))
    jitter = random.uniform(-0.05, 0.09)
    depth = random.uniform(-0.11, 0.1)
    p = Vector((0, 0, POST_H)) + radial * (ARCH_R + jitter) + Vector((0, depth, 0))
    nrm = (radial * random.uniform(0.3, 0.8) + Vector((0, -1, 0)) * (0.9 if depth < 0.03 else -0.3)
           + Vector((random.uniform(-.3, .3), 0, random.uniform(-.3, .3)))).normalized()
    rose(f"arch_rose{k}", p, nrm, random.uniform(0.03, 0.062), M["roses_arch"], parent=arch_root)
    n_roses += 1
# roses also run down the top of each post a little + baby's breath + eucalyptus
for sx in (-1, 1):
    for k in range(18):
        z = random.uniform(0.75, POST_H + 0.05)
        p = Vector((sx * (ARCH_R + random.uniform(-0.06, 0.06)), random.uniform(-0.1, 0.08), z))
        rose(f"arch_post_rose{sx}_{k}", p, Vector((sx * 0.5, -1, random.uniform(-.2, .4))), random.uniform(0.028, 0.052), M["roses_arch"], parent=arch_root)
for k in range(70):
    t = random.uniform(0.0, 1.0)
    a = math.pi * t
    radial = Vector((math.cos(a), 0, math.sin(a)))
    p = Vector((0, 0, POST_H)) + radial * (ARCH_R + random.uniform(0.02, 0.16)) + Vector((0, random.uniform(-0.12, 0.12), 0))
    leaf(f"arch_leaf{k}", p, radial + Vector((random.uniform(-.6, .6), random.uniform(-.5, .5), random.uniform(-.6, .6))),
         random.uniform(0.08, 0.13), parent=arch_root)

neon1 = text_obj("neon_better", "Better", "Caveat-Medium", 0.29, M["neon"], extrude=0.006,
                 parent=arch_root, loc=(-0.16, -0.02, 1.83), rot=FACE_MINUS_Y)
neon2 = text_obj("neon_together", "Together", "Caveat-Medium", 0.29, M["neon"], extrude=0.006,
                 parent=arch_root, loc=(0.1, -0.02, 1.57), rot=FACE_MINUS_Y)
for ob in (neon1, neon2):
    ob.data.offset = 0.0035  # makes the strokes a touch bolder, like neon tube

# --------------------------------------------------------------------------------------
# the Gran Vía booth
# --------------------------------------------------------------------------------------
booth = empty("booth", (0, BOOTH_Y, 0))
box("booth_cabinet", (0.62, 0.42, 0.66), M["wood"], bevel=0.012, parent=booth, loc=(0, 0, 0.83))
box("booth_pedestal", (0.5, 0.36, 0.5), M["wood"], bevel=0.01, parent=booth, loc=(0, 0, 0.25))
box("booth_plinth", (0.54, 0.4, 0.03), M["ink"], bevel=0.004, parent=booth, loc=(0, 0, 0.015))
box("booth_bezel", (0.475, 0.012, 0.315), M["black"], bevel=0.008, parent=booth, loc=(0, 0.21, 0.865))
SCREEN_W, SCREEN_H = 0.44, 0.293
screen_seq_len = RENDER_FRAMES
screen_mat = image_material("booth_screen", os.path.join(TEX, "screen", "screen_0001.png"), rough=0.3,
                            emit_str=2.4, coat=0.25, sequence_len=screen_seq_len)
sw, sh = SCREEN_W / 2, SCREEN_H / 2
from_pydata("booth_screen", [(-sw, -sh, 0), (sw, -sh, 0), (sw, sh, 0), (-sw, sh, 0)], [(0, 1, 2, 3)], screen_mat,
            uvs=[(0, 0), (1, 0), (1, 1), (0, 1)], smooth=False, parent=booth, loc=(0, 0.2165, 0.865), rot=FACE_PLUS_Y)
# engraved logo under the screen: "GRAN VÍA | STUDIO" (as on the real booth)
lg1 = text_obj("booth_logo_granvia", "GRAN VÍA", "PlayfairDisplay-Bold", 0.036, M["ink"], extrude=0.0008, align="RIGHT",
               parent=booth, rot=FACE_PLUS_Y)
lg2 = text_obj("booth_logo_bar", "|", "Montserrat-Regular", 0.03, M["ink"], extrude=0.0008,
               parent=booth, rot=FACE_PLUS_Y)
lg3 = text_obj("booth_logo_studio", "STUDIO", "Montserrat-Medium", 0.0165, M["ink"], extrude=0.0008, spacing=1.35,
               align="LEFT", parent=booth, rot=FACE_PLUS_Y)
bpy.context.view_layer.update()
_gap = 0.013
_shift = (lg1.dimensions.x - lg3.dimensions.x) / 2      # centre the whole lock-up
lg1.location = (_gap - _shift, 0.2112, 0.672)
lg2.location = (-_shift, 0.2112, 0.673)
lg3.location = (-_gap - _shift, 0.2112, 0.6735)
# printer slot + lip
box("booth_slot", (0.15, 0.01, 0.016), M["black"], parent=booth, loc=(0, 0.2085, 0.585))
box("booth_slot_lip", (0.17, 0.035, 0.007), M["gold"], bevel=0.002, parent=booth, loc=(0, 0.225, 0.572))
# camera lens + flash window
cone("booth_lens_barrel", 0.042, 0.04, 0.035, M["black"], seg=32, parent=booth, loc=(0, 0.22, 1.09), rot=(math.radians(90), 0, 0))
sphere("booth_lens_glass", 0.03, M["lens_glass"], seg=32, rings=16, parent=booth, loc=(0, 0.236, 1.09), scale=(1, 0.35, 1))
torus("booth_lens_ring", 0.036, 0.004, M["gold"], seg=40, rseg=8, parent=booth, loc=(0, 0.238, 1.09), rot=(math.radians(90), 0, 0))
_fw = principled("flash_window", "#F4F1EC", rough=0.3, emit="#FFFFFF", emit_str=0.4)
_fw.cycles.emission_sampling = "NONE"
box("booth_flash_window", (0.16, 0.006, 0.035), _fw,
    bevel=0.004, parent=booth, loc=(0, 0.212, 1.145))
# flowers on top of the cabinet, like the real booth
flower_cluster("booth_top_l", (0.22, 0.12, 1.19), 9, 0.1, M["roses_white"], size=(0.022, 0.036), parent=booth)
flower_cluster("booth_top_r", (-0.25, 0.1, 1.18), 5, 0.07, M["roses_white"], size=(0.02, 0.032), parent=booth)

# ring light on its stand
RING_POS = Vector((-0.52, BOOTH_Y + 0.06, 1.42))
cone("ring_pole", 0.011, 0.011, 1.36, M["pole"], seg=10, loc=(RING_POS.x, RING_POS.y, 0.68))
for k in range(3):
    a = k * 2.094 + 0.4
    cone(f"ring_leg{k}", 0.008, 0.008, 0.36, M["pole"], seg=8,
         loc=(RING_POS.x + 0.12 * math.cos(a), RING_POS.y + 0.12 * math.sin(a), 0.1),
         rot=Euler((0, math.radians(55), a)).to_quaternion().to_euler())
torus("ring_light", 0.13, 0.017, M["ring"], seg=64, rseg=12, loc=RING_POS, rot=(math.radians(90), 0, 0))
torus("ring_housing", 0.13, 0.021, M["black"], seg=64, rseg=12, loc=RING_POS + Vector((0, -0.012, 0)), rot=(math.radians(90), 0, 0))


# --------------------------------------------------------------------------------------
# tables (guestbook + audio guestbook)
# --------------------------------------------------------------------------------------
def draped_table(name, pos, top_z, radius):
    root = empty(name, pos)
    verts, faces = [], []
    rings, seg = 18, 72
    for i in range(rings + 1):
        v = i / rings
        z = top_z * (1 - v)
        r = radius + 0.01 + 0.07 * v ** 1.8
        for j in range(seg):
            th = 2 * math.pi * j / seg
            fold = 1 + 0.06 * v ** 1.3 * math.sin(9 * th + 0.5) + 0.02 * v * math.sin(17 * th)
            verts.append((r * fold * math.cos(th), r * fold * math.sin(th), z))
    for i in range(rings):
        for j in range(seg):
            j2 = (j + 1) % seg
            faces.append((i * seg + j, i * seg + j2, (i + 1) * seg + j2, (i + 1) * seg + j))
    from_pydata(name + "_cloth", verts, faces, M["linen"], parent=root, subsurf=1)
    bm = bmesh.new()
    bmesh.ops.create_circle(bm, cap_ends=True, segments=72, radius=radius + 0.01)
    finish_mesh(name + "_top", bm, M["linen"], smooth=True, parent=root, loc=(0, 0, top_z))
    torus(name + "_edge", radius + 0.01, 0.012, M["linen"], seg=72, rseg=10, parent=root, loc=(0, 0, top_z - 0.006))
    return root


gb_table = draped_table("guestbook_table", GB, GB_TOP, 0.42)
ph_table = draped_table("phone_table", PH, PH_TOP, 0.33)

# guestbook: linen cover + two page blocks with a soft dip at the spine
book = empty("guestbook", (GB.x, GB.y, GB_TOP), rot=(0, 0, 0))
box("book_cover", (0.49, 0.335, 0.01), M["cover"], bevel=0.004, parent=book, loc=(0, 0, 0.005))
for side in (-1, 1):
    verts, faces = [], []
    nx, ny = 28, 2
    for i in range(nx + 1):
        u = i / nx * 0.232
        z = 0.022 - 0.011 * math.exp(-u / 0.018)
        for j in range(ny + 1):
            y = -0.155 + 0.31 * j / ny
            verts.append((side * (u + 0.003), y, z))
    for i in range(nx):
        for j in range(ny):
            a = i * (ny + 1) + j
            f = (a, a + ny + 1, a + ny + 2, a + 1)
            faces.append(f if side > 0 else f[::-1])
    pg = from_pydata(f"book_pages{side}", verts, faces, M["paper"], parent=book)
    so = pg.modifiers.new("solid", "SOLIDIFY")
    so.thickness = 0.012
    so.offset = -1
text_obj("book_header", "Ana i Marko · 12.6.2027.", "PlayfairDisplay-Italic", 0.0105, principled("taupe_ink", "#9C8476"),
         parent=book, loc=(0.118, 0.128, 0.0225))
text_obj("book_header_l", "Knjiga gostiju", "PlayfairDisplay-Italic", 0.0105, bpy.data.materials["taupe_ink"],
         parent=book, loc=(-0.118, 0.128, 0.0225))
# vase with roses on the guestbook table
vase = cone("gb_vase", 0.035, 0.045, 0.11, M["ceramic"], seg=40, loc=(GB.x + 0.24, GB.y + 0.25, GB_TOP + 0.055))
flower_cluster("gb_vase_fl", (GB.x + 0.24, GB.y + 0.25, GB_TOP + 0.14), 7, 0.07, M["roses_blush"], size=(0.02, 0.03), leaves=5)
for k in range(6):  # loose petals on the cloth
    p = (GB.x + random.uniform(-0.35, 0.35), GB.y + random.uniform(-0.3, 0.3), GB_TOP + 0.002)
    if abs(p[0] - GB.x) < 0.26 and abs(p[1] - GB.y) < 0.19:
        continue
    sphere(f"petal{k}", 0.012, M["sash"], seg=10, rings=6, loc=p, scale=(1, 0.7, 0.15), rot=(0, 0, random.uniform(0, 3)))

# the strip that lands in the book (S3/S5) — UV upright for a top-down camera
strip_tex = os.path.join(TEX, "strip.png")
M["strip"] = image_material("photo_strip", strip_tex, rough=0.28, coat=0.6)
hw, hl = STRIP_W / 2, STRIP_L / 2
strip_book = from_pydata("strip_book", [(-hw, -hl, 0), (hw, -hl, 0), (hw, hl, 0), (-hw, hl, 0)], [(0, 1, 2, 3)],
                         M["strip"], uvs=[(0, 0), (1, 0), (1, 1), (0, 1)], smooth=False)
so = strip_book.modifiers.new("solid", "SOLIDIFY")
so.thickness = 0.0008
# the strip coming out of the printer (S2): origin at trailing edge, extends along +Y.
# Image top points back toward the booth so it reads upright for the S2 camera.
strip_print = from_pydata("strip_print", [(-hw, 0, 0), (hw, 0, 0), (hw, STRIP_L, 0), (-hw, STRIP_L, 0)], [(0, 1, 2, 3)],
                          M["strip"], uvs=[(1, 1), (0, 1), (0, 0), (1, 0)], smooth=False)
so = strip_print.modifiers.new("solid", "SOLIDIFY")
so.thickness = 0.0008

# handwriting on the right page
TXT_Z = GB_TOP + 0.0232
line1 = text_obj("write_zivjeli", "Živjeli", "Caveat-Medium", 0.052, M["writing"], align="LEFT", valign="BOTTOM_BASELINE",
                 loc=(GB.x + 0.04, GB.y + 0.035, TXT_Z))
line2 = text_obj("write_mladenci", "mladenci!", "Caveat-Medium", 0.052, M["writing"], align="LEFT", valign="BOTTOM_BASELINE",
                 loc=(GB.x + 0.058, GB.y - 0.03, TXT_Z))
for ob in (line1, line2):
    ob.data.offset = 0.0004
heart_pts = []
for k in range(90):
    t = 2 * math.pi * k / 89
    x = 16 * math.sin(t) ** 3
    y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
    heart_pts.append((x * 0.0011, y * 0.0011, 0))
# start the doodle at the top notch so it draws like a pen would
HEART_ORIGIN = Vector((GB.x + 0.185, GB.y - 0.068, TXT_Z))
heart_doodle = curve_obj("write_heart", heart_pts, 0.0011, M["heart_ink"], loc=HEART_ORIGIN, res=2)
heart_doodle.data.bevel_factor_mapping_end = "SPLINE"

# gold pen + a guest's hand
pen = empty("pen")
cone("pen_body", 0.0065, 0.0065, 0.12, M["gold"], seg=20, parent=pen, loc=(0, 0, 0.075))
cone("pen_tip", 0.0012, 0.0065, 0.016, M["black"], seg=20, parent=pen, loc=(0, 0, 0.008))
sphere("pen_cap", 0.0068, M["black"], seg=16, rings=8, parent=pen, loc=(0, 0, 0.135))
guest_hand = sphere("guest_hand", 0.036, M["skin_b"], seg=32, rings=16, scale=(1, 0.85, 1.1))
sphere("guest_thumb", 0.014, M["skin_b"], seg=16, rings=8, parent=guest_hand, loc=(-0.026, -0.016, -0.014))

# --------------------------------------------------------------------------------------
# rotary phone (audio guestbook)
# --------------------------------------------------------------------------------------
S_ = PHONE_S
phone = empty("phone", (PH.x, PH.y - 0.04, PH_TOP))
bm = bmesh.new()
bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=4, radius1=0.15 * S_, radius2=0.085 * S_, depth=0.075 * S_)
bmesh.ops.rotate(bm, verts=bm.verts, cent=(0, 0, 0), matrix=Matrix.Rotation(math.radians(45), 3, "Z"))
bmesh.ops.translate(bm, verts=bm.verts, vec=(0, 0, 0.0375 * S_))
ph_body = finish_mesh("phone_body", bm, M["phone"], parent=phone, subsurf=2, smooth=True)
bv = ph_body.modifiers.new("bevel", "BEVEL")
bv.width = 0.018 * S_
bv.segments = 2
ph_body.modifiers.move(1, 0)
tilt = math.atan2((0.15 - 0.085) * math.cos(math.radians(45)), 0.075)
dial_c = Vector((0, -((0.15 + 0.085) / 2) * math.cos(math.radians(45)) * S_ - 0.004, 0.036 * S_))
dial = empty("phone_dial", dial_c, parent=phone, rot=(math.pi / 2 - tilt, 0, 0))
cone("dial_plate", 0.043 * S_, 0.043 * S_, 0.006, M["gold"], seg=48, parent=dial, rot=(0, 0, 0))
cone("dial_face", 0.026 * S_, 0.026 * S_, 0.008, M["phone"], seg=40, parent=dial, loc=(0, 0, 0.002))
for k in range(10):
    a = math.radians(-60 + k * 28)
    sphere(f"dial_hole{k}", 0.0065 * S_, M["black"], seg=12, rings=6, parent=dial,
           loc=(0.034 * S_ * math.cos(a), 0.034 * S_ * math.sin(a), 0.003), scale=(1, 1, 0.35))
for sx in (-1, 1):
    cone(f"cradle{sx}", 0.008 * S_, 0.006 * S_, 0.03 * S_, M["phone"], seg=12, parent=phone,
         loc=(sx * 0.055 * S_, 0, 0.083 * S_))

handset = empty("handset")
hs_pts = []
for k in range(25):
    x = -0.11 + 0.22 * k / 24
    hs_pts.append((x * S_, 0, (0.03 * (1 - (x / 0.11) ** 2) + 0.008) * S_))
curve_obj("handset_handle", hs_pts, 0.016 * S_, M["phone"], parent=handset, res=6)
for sx in (-1, 1):
    cone(f"handset_cup{sx}", 0.032 * S_, 0.026 * S_, 0.035 * S_, M["phone"], seg=32, parent=handset,
         loc=(sx * 0.11 * S_, 0, -0.004 * S_), subsurf=1)
HANDSET_REST = (Vector((PH.x, PH.y - 0.04, PH_TOP + 0.087 * S_)), Euler((0, 0, 0)))

# coiled cord: a POLY curve whose points are baked per frame
CORD_N = 150
cord = curve_obj("phone_cord", [(0, 0, 0)] * CORD_N, 0.0032, M["phone"], res=2)

# --------------------------------------------------------------------------------------
# characters
# --------------------------------------------------------------------------------------
HEAD_R = 0.22
NECK_Z = 0.68
HEAD_CZ = 0.215


def on_head(az, el, r=HEAD_R):
    return Vector((r * math.sin(az) * math.cos(el), -r * math.cos(az) * math.cos(el), r * math.sin(el) + HEAD_CZ))


class Character:
    def __init__(self, name, kind, skin, hair_mat):
        self.name, self.kind = name, kind
        self.root = empty(name)
        self.body = empty(name + "_body", parent=self.root)
        self.head = empty(name + "_head", (0, 0, NECK_Z), parent=self.body)
        self.hands = {}
        self.skin = skin
        self._build_head(hair_mat)
        self._build_body()
        for side, sx in (("sl", -1), ("sr", 1)):
            h = empty(f"{name}_hand_{side}", (sx * 0.27, -0.02, 0.4), parent=self.root)
            sphere(f"{name}_mitt_{side}", 0.056, skin, seg=32, rings=16, parent=h, scale=(1, 0.82, 1.08))
            sphere(f"{name}_thumb_{side}", 0.021, skin, seg=16, rings=8, parent=h, loc=(-sx * 0.04, -0.03, 0.012))
            self.hands[side] = h

    def _build_head(self, hair_mat):
        n, H = self.name, self.head
        sphere(n + "_skull", HEAD_R, self.skin, seg=72, rings=36, parent=H, loc=(0, 0, HEAD_CZ))
        self.eyes = []
        for sx in (-1, 1):
            p = on_head(sx * 0.37, 0.02, HEAD_R - 0.005)
            e = sphere(f"{n}_eye{sx}", 0.03, M["eye"], seg=24, rings=12, parent=H, loc=p,
                       rot=(-0.02, 0, sx * 0.37), scale=(0.8, 0.45, 1.15))
            sphere(f"{n}_catch{sx}", 0.0085, M["catch"], seg=12, rings=6, parent=e, loc=(-0.009, -0.024, 0.011))
            self.eyes.append(e)
            pc = on_head(sx * 0.66, -0.2, HEAD_R - 0.012)
            sphere(f"{n}_cheek{sx}", 0.042, M["blush"], seg=20, rings=10, parent=H, loc=pc,
                   rot=(0.2, 0, sx * 0.66), scale=(1, 0.32, 0.62))
        c = on_head(0, -0.3, HEAD_R + 0.002)
        pts = []
        for k in range(15):
            s = -1 + 2 * k / 14
            pts.append(on_head(s * 0.21, -0.28 - 0.075 * (1 - s * s), HEAD_R + 0.003) - c)
        self.smile = curve_obj(n + "_smile", pts, 0.0075, M["mouth"], parent=H, loc=c, res=3)
        pm = on_head(0, -0.33, HEAD_R - 0.012)
        self.mouth = sphere(n + "_mouth_open", 0.042, M["mouth_in"], seg=24, rings=12, parent=H, loc=pm,
                            rot=(0.33, 0, 0), scale=(0.0, 0.0, 0.0))
        if self.kind == "bride":
            sphere(n + "_hair", 0.232, hair_mat, seg=72, rings=36, parent=H, loc=(0, 0.026, HEAD_CZ + 0.04))
            for sx in (-1, 1):
                sphere(f"{n}_fringe{sx}", 0.1, hair_mat, seg=40, rings=20, parent=H,
                       loc=(sx * 0.075, -0.145, HEAD_CZ + 0.125), rot=(0.55, -sx * 0.5, sx * 0.35),
                       scale=(1.25, 0.55, 0.6))
            sphere(n + "_bun", 0.085, hair_mat, seg=32, rings=16, parent=H, loc=(0, 0.13, HEAD_CZ + 0.17))
            for k in range(7):  # little flower crown around the bun
                a = 2 * math.pi * k / 7
                rose(f"{n}_crown{k}", (0.085 * math.cos(a), 0.13 + 0.085 * math.sin(a) * 0.6, HEAD_CZ + 0.17 - 0.03 * math.sin(a)),
                     (math.cos(a), 0.3 + math.sin(a) * 0.5, 0.4), 0.016, M["roses_blush"], parent=H)
            verts, faces = [], []
            rings, seg = 14, 28
            for i in range(rings + 1):
                v = i / rings
                for j in range(seg + 1):
                    th = math.pi * (0.15 + 0.7 * j / seg)
                    r = 0.07 + 0.25 * v ** 0.9
                    verts.append((r * math.cos(th), 0.14 + r * math.sin(th) * 0.75 + 0.02 * v, HEAD_CZ + 0.15 - 0.78 * v))
            for i in range(rings):
                for j in range(seg):
                    a = i * (seg + 1) + j
                    faces.append((a, a + 1, a + seg + 2, a + seg + 1))
            vl = from_pydata(n + "_veil", verts, faces, M["veil"], parent=H)
            so = vl.modifiers.new("solid", "SOLIDIFY")
            so.thickness = 0.003
        elif self.kind == "groom":
            sphere(n + "_hair", 0.235, hair_mat, seg=72, rings=36, parent=H, loc=(0, 0.03, HEAD_CZ + 0.052))
            sphere(n + "_quiff", 0.08, hair_mat, seg=32, rings=16, parent=H, loc=(0.03, -0.1, HEAD_CZ + 0.185),
                   rot=(0.4, 0, 0.2), scale=(1.5, 1.0, 0.6))
        else:
            sphere(n + "_hair", HEAD_R + 0.026, hair_mat, seg=72, rings=36, parent=H, loc=(0, 0.045, HEAD_CZ + 0.0),
                   scale=(1.0, 1.0, 1.04))
            sphere(n + "_fringe", 0.1, hair_mat, seg=32, rings=16, parent=H, loc=(-0.05, -0.135, HEAD_CZ + 0.15),
                   rot=(0.5, 0.3, 0.3), scale=(1.5, 0.7, 0.55))

    def _build_body(self):
        n, B = self.name, self.body
        if self.kind == "bride":
            sk = cone(n + "_skirt", 0.4, 0.14, 0.58, M["dress"], seg=64, parent=B, loc=(0, 0, 0.29), subsurf=1)
            sphere(n + "_bodice", 0.155, M["dress"], seg=40, rings=20, parent=B, loc=(0, 0, 0.6), scale=(1, 0.86, 1.12))
            torus(n + "_sash", 0.148, 0.02, M["sash"], seg=48, rseg=10, parent=B, loc=(0, 0, 0.53), scale=(1, 0.88, 1))
        elif self.kind == "groom":
            for sx in (-1, 1):
                cone(f"{n}_leg{sx}", 0.056, 0.056, 0.2, M["suit"], seg=24, parent=B, loc=(sx * 0.075, 0, 0.11))
                sphere(f"{n}_shoe{sx}", 0.065, M["black"], seg=24, rings=12, parent=B, loc=(sx * 0.078, -0.03, 0.028),
                       scale=(0.9, 1.4, 0.55))
            sphere(n + "_torso", 0.2, M["suit"], seg=40, rings=20, parent=B, loc=(0, 0, 0.44), scale=(1.0, 0.83, 1.22))
            sphere(n + "_shirt", 0.1, M["shirt"], seg=32, rings=16, parent=B, loc=(0, -0.112, 0.55), scale=(0.55, 0.42, 1.0))
            for sx in (-1, 1):
                cone(f"{n}_bow{sx}", 0.0, 0.026, 0.038, M["black"], seg=4, parent=B, loc=(sx * 0.02, -0.163, 0.628),
                     rot=(0, sx * math.pi / 2, 0))
            sphere(n + "_knot", 0.011, M["black"], seg=12, rings=8, parent=B, loc=(0, -0.168, 0.628))
            rose(n + "_boutonniere", (0.1, -0.142, 0.54), (0.2, -1, 0.3), 0.015, M["roses_blush"], parent=B)
            for k in range(3):
                sphere(f"{n}_button{k}", 0.008, M["black"], seg=10, rings=6, parent=B, loc=(0, -0.168, 0.46 - 0.06 * k))
        else:
            for sx in (-1, 1):
                cone(f"{n}_leg{sx}", 0.032, 0.032, 0.14, self.skin, seg=16, parent=B, loc=(sx * 0.07, 0, 0.075))
                sphere(f"{n}_shoe{sx}", 0.045, M["tan"], seg=20, rings=10, parent=B, loc=(sx * 0.072, -0.02, 0.022),
                       scale=(0.9, 1.4, 0.55))
            cone(n + "_skirt", 0.3, 0.13, 0.42, M["teal"], seg=56, parent=B, loc=(0, 0, 0.34), subsurf=1)
            sphere(n + "_bodice", 0.14, M["teal"], seg=40, rings=20, parent=B, loc=(0, 0, 0.58), scale=(1, 0.86, 1.1))


bride = Character("bride", "bride", M["skin_a"], M["hair_bride"])
groom = Character("groom", "groom", M["skin_b"], M["hair_groom"])
friend = Character("friend", "friend", M["skin_c"], M["hair_friend"])

# props
bouquet = empty("bouquet", (0, -0.02, 0.03), parent=bride.hands["sl"])
flower_cluster("bouquet", (0, -0.03, 0.05), 7, 0.06, M["roses_blush"], size=(0.017, 0.025), parent=bouquet, leaves=5)
cone("bouquet_stems", 0.012, 0.025, 0.09, M["leaf"], seg=12, parent=bouquet, loc=(0, 0.0, -0.02))
torus("bouquet_ribbon", 0.02, 0.006, M["satin"], seg=24, rseg=8, parent=bouquet, loc=(0, 0, -0.02))

mustache = empty("mustache_prop", (0, -0.01, 0.0), parent=groom.hands["sl"])
cone("mustache_stick", 0.005, 0.005, 0.2, M["gold"], seg=10, parent=mustache, loc=(0, 0, 0.1))
for sx in (-1, 1):
    sphere(f"mustache_half{sx}", 0.05, M["black"], seg=24, rings=12, parent=mustache, loc=(sx * 0.042, -0.006, 0.205),
           rot=(0, sx * 0.35, 0), scale=(1.0, 0.3, 0.38))
    sphere(f"mustache_curl{sx}", 0.018, M["black"], seg=16, rings=8, parent=mustache, loc=(sx * 0.088, -0.006, 0.222),
           scale=(1, 0.5, 1))

sign = empty("sign_prop", (0, -0.01, 0.0), parent=friend.hands["sr"])
cone("sign_stick", 0.006, 0.006, 0.2, M["gold"], seg=10, parent=sign, loc=(0, 0.004, 0.1))
box("sign_card", (0.38, 0.014, 0.16), M["card"], bevel=0.008, parent=sign, loc=(0, 0, 0.27))
_st = text_obj("sign_text", "Živjeli!", "Caveat-Medium", 0.125, M["sign_ink"], extrude=0.0012, parent=sign,
               loc=(0, -0.0085, 0.258), rot=FACE_MINUS_Y)
_st.data.offset = 0.0035

heart_pop = empty("heart_pop", (0.16, 1.28, 1.25))
bm = bmesh.new()
hverts = []
for k in range(48):
    t = 2 * math.pi * k / 48
    x = 16 * math.sin(t) ** 3
    y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
    hverts.append(bm.verts.new((x * 0.0055, 0, y * 0.0055)))
fce = bm.faces.new(hverts)
ext = bmesh.ops.extrude_face_region(bm, geom=[fce])
bmesh.ops.translate(bm, verts=[v for v in ext["geom"] if isinstance(v, bmesh.types.BMVert)], vec=(0, 0.04, 0))
bmesh.ops.translate(bm, verts=bm.verts, vec=(0, -0.02, 0))
hm = finish_mesh("heart_pop_mesh", bm, M["heart"], parent=heart_pop, smooth=False, subsurf=2)
bvh = hm.modifiers.new("bevel", "BEVEL")
bvh.width = 0.012
bvh.segments = 3
hm.modifiers.move(1, 0)

# sound rings for the audio guestbook
RINGS = []
for k in range(3):
    pair = empty(f"sound_wave{k}")
    for side in (0, 1):
        torus(f"sound_arc{k}_{side}", 0.1, 0.0035, M["glow"], seg=24, rseg=6, arc=math.radians(70), parent=pair,
              rot=(0, 0, math.radians(-35) + side * math.pi))
    RINGS.append(pair)

# --------------------------------------------------------------------------------------
# lights
# --------------------------------------------------------------------------------------
def area(name, loc, target, power, color, size=1.0, size_y=None, shape="RECTANGLE"):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy = power
    ld.color = srgb(color)[:3]
    ld.shape = shape
    ld.size = size
    if size_y is not None:
        ld.size_y = size_y
    ob = bpy.data.objects.new(name, ld)
    ob.location = loc
    ob.rotation_euler = look_rot(loc, target)
    link(ob)
    return ob


def point(name, loc, power, color, radius=0.1):
    ld = bpy.data.lights.new(name, "POINT")
    ld.energy = power
    ld.color = srgb(color)[:3]
    ld.shadow_soft_size = radius
    ob = bpy.data.objects.new(name, ld)
    ob.location = loc
    return link(ob)


area("key", (-3.0, -2.6, 4.1), (0, 0.9, 0.6), 260, "#FFCF9A", size=2.6, size_y=2.6)
area("fill", (3.6, -1.8, 2.8), (0, 0.5, 0.8), 70, "#9FB2E3", size=3.0, size_y=3.0)
area("rim", (0.6, 4.4, 2.9), (0, 1.3, 0.9), 320, "#FFC48A", size=2.2, size_y=1.2)
booth_fill = area("booth_fill", RING_POS + Vector((0.0, 0.05, 0)), (0, 1.35, 0.8), 40, "#FFF2E2", size=0.28, shape="DISK")
flash = area("flash", (0, BOOTH_FRONT + 0.06, 1.15), (0, 1.35, 0.85), 0, "#FFFFFF", size=0.3, size_y=0.12)
point("neon_glow", (0, ARCH_Y - 0.28, 1.68), 32, "#FFC88A", radius=0.3)
area("gb_light", (GB.x + 0.5, GB.y - 0.55, 1.75), (GB.x, GB.y, GB_TOP), 13, "#FFE1BE", size=0.9, size_y=0.9)
area("ph_light", (PH.x - 0.6, PH.y - 0.7, 1.9), (PH.x, PH.y + 0.4, 0.8), 38, "#FFE1BE", size=0.9, size_y=0.9)
area("screen_glow", (0, BOOTH_FRONT + 0.4, 0.9), (0, 1.0, 0.9), 2.5, "#DCE6FF", size=0.4, size_y=0.27)
area("printer_spot", (0.25, BOOTH_FRONT + 0.6, 1.25), (0, BOOTH_FRONT + 0.1, 0.58), 9, "#FFE6C8", size=0.4, size_y=0.4)

# --------------------------------------------------------------------------------------
# ANIMATION
# --------------------------------------------------------------------------------------
BASE = dict(sl=(-0.27, -0.02, 0.4), sr=(0.27, -0.02, 0.4), head=(0.0, 0.0, 0.0), lean=(0.0, 0.0),
            squash=1.0, eyes=1.0, mouth=0.0, wave_sl=0.0, wave_sr=0.0, prop=0.0, run=0.0)


def pose_at(keys, f):
    cur = dict(BASE)
    for (k0, over, dur, ease) in keys:
        if f < k0:
            break
        tgt = {**cur, **over}
        t = 1.0 if dur <= 0 else (f - k0) / dur
        if t < 1.0:
            e = ease(clamp(t))
            return {k: lerp(cur[k], tgt[k], e) for k in cur}
        cur = tgt
    return cur


HANDS_UP = dict(sl=(-0.34, -0.07, 0.72), sr=(0.34, -0.07, 0.72))
CROUCH = dict(squash=0.8, sl=(-0.3, 0.06, 0.28), sr=(0.3, 0.06, 0.28), lean=(0.12, 0.0), head=(0.1, 0, 0), eyes=1.0,
              mouth=0.0, prop=0.0)
TAKEOFF = dict(squash=1.14, sl=(-0.24, -0.05, 1.06), sr=(0.24, -0.05, 1.06), lean=(-0.05, 0.0), head=(-0.15, 0, 0),
               mouth=0.7)
AIR = dict(squash=1.0)
LAND = dict(squash=0.84, sl=(-0.33, -0.05, 0.62), sr=(0.33, -0.05, 0.62), mouth=0.3)
SETTLE = dict(squash=1.0, sl=(-0.27, -0.02, 0.4), sr=(0.27, -0.02, 0.4), lean=(0, 0), head=(0, 0, 0), mouth=0.0)


def jump_keys():
    return [(92, CROUCH, 6, in_out), (98, TAKEOFF, 4, out_cubic), (102, AIR, 4, in_out), (112, LAND, 3, out_cubic),
            (115, SETTLE, 8, out_back_soft)]


WAVE_BOTH = dict(sl=(-0.33, -0.08, 0.8), sr=(0.33, -0.08, 0.8), wave_sl=1.0, wave_sr=1.0, mouth=0.35, head=(-0.08, 0, 0))

POSES = {
    "groom": [
        (15, dict(sr=(0.3, -0.06, 0.82), wave_sr=1.0, head=(0, -0.12, -0.35), mouth=0.4), 6, out_back),
        (40, dict(**HANDS_UP, wave_sr=0.0, head=(0, 0.12, 0.0), mouth=0.0), 7, out_back),
        (62, dict(sl=(0.0, -0.255, 0.645), sr=(0.3, -0.08, 0.42), prop=1.0, head=(-0.06, 0.0, 0.0), eyes=1.2), 9, out_back),
        (77, dict(lean=(0.0, -0.06), head=(0.0, -0.1, -0.15), eyes=0.08, prop=0.0, sl=(-0.3, -0.04, 0.5),
                  sr=(0.3, -0.05, 0.5)), 8, out_back_soft),
        *jump_keys(),
        (170, dict(head=(0, -0.1, -0.5), mouth=0.3), 10, in_out),          # chatting with the bride (seen in S4)
        (330, dict(head=(0.05, 0.15, -0.45), mouth=0.0, sr=(0.2, -0.12, 0.6)), 10, in_out),
        (392, WAVE_BOTH, 8, out_back),
    ],
    "bride": [
        (0, dict(sr=(0.17, -0.34, 0.82)), 0, linear),
        (1, dict(sr=(0.12, -0.425, 0.84)), 5, out_cubic),                  # tap!
        (8, dict(sr=(0.22, -0.26, 0.66)), 6, in_out),
        (15, dict(run=1.0, sl=(-0.3, 0.0, 0.5), sr=(0.3, 0.0, 0.5), mouth=0.5), 4, in_out),
        (35, dict(run=0.0, squash=0.86, mouth=0.0), 3, out_cubic),
        (38, dict(squash=1.0, sl=(-0.3, -0.13, 0.74), sr=(0.34, -0.07, 0.74), head=(0.0, 0.1, 0.0), wave_sr=0.6), 7, out_back),
        (52, dict(wave_sr=0.0), 4, in_out),
        (62, dict(sl=(-0.2, -0.21, 0.88), sr=(0.2, -0.21, 0.88), head=(0.0, 0.22, 0.0), mouth=0.75, eyes=1.15), 9, out_back),
        (77, dict(lean=(0.0, 0.06), head=(0.0, 0.1, 0.15), eyes=0.08, mouth=0.0, sl=(-0.3, -0.06, 0.45),
                  sr=(0.33, -0.06, 0.62)), 8, out_back_soft),
        *jump_keys(),
        (170, dict(head=(0, 0.1, 0.5), mouth=0.2), 10, in_out),
        (330, dict(head=(-0.05, -0.1, 0.4), mouth=0.0), 10, in_out),
        (392, WAVE_BOTH, 8, out_back),
    ],
    "friend": [
        (15, dict(sl=(-0.32, -0.07, 0.84), wave_sl=1.0, mouth=0.5, head=(0, 0.1, 0.25)), 6, out_back),
        (40, dict(**HANDS_UP, wave_sl=0.0, head=(0, -0.12, 0.0), mouth=0.0), 7, out_back),
        (62, dict(sr=(0.02, -0.1, 1.02), sl=(-0.31, -0.06, 0.52), prop=1.0, head=(-0.12, -0.12, 0.0), mouth=0.55), 9, out_back),
        (77, dict(sl=(-0.15, -0.21, 0.78), sr=(0.15, -0.21, 0.78), prop=0.0, head=(0.0, 0.15, 0.0), eyes=1.15, mouth=0.45), 8,
         out_back_soft),
        *jump_keys(),
        # S4: audio guestbook (hands overridden while holding the handset)
        (300, dict(sl=(-0.3, -0.02, 0.4), sr=(0.27, -0.02, 0.4), head=(0.1, 0, 0.1), mouth=0.0, eyes=1.0), 4, in_out),
        (318, dict(head=(0.02, -0.12, 0.0), sr=(0.3, -0.14, 0.62)), 6, out_back),
        (340, dict(head=(-0.04, 0.1, 0.05), sr=(0.34, -0.12, 0.76), wave_sr=0.5), 8, out_back),
        (362, dict(head=(-0.25, -0.05, 0.0), eyes=0.12, wave_sr=0.0, sr=(0.24, -0.16, 0.58)), 6, out_back),  # laugh
        (378, dict(head=(0.0, -0.1, 0.0), eyes=1.0), 8, in_out),
        (392, dict(sr=(0.34, -0.08, 0.82), wave_sr=1.0), 8, out_back),
    ],
}

BLINKS = {"bride": [52, 136, 336, 402], "groom": [33, 51, 344, 410], "friend": [27, 54, 325, 351, 398]}
PHASE = {"bride": 0.0, "groom": 1.7, "friend": 3.1}


# root motion (x, y, rot_z) per character
def root_keys(name):
    if name == "bride":
        return [(0, (-0.12, -0.86, 0.0)), (15, (-0.12, -0.86, 0.0)), (19, (-0.1, -0.78, math.pi), out_cubic),
                (34, (-0.12, 1.31, math.pi), in_out), (40, (-0.12, 1.31, 2 * math.pi), out_back_soft)]
    if name == "groom":
        return [(0, (0.44, 1.36, 0.0)), (300, (0.44, 1.36, 0.0)), (330, (0.44, 1.36, 0.25)), (392, (0.44, 1.36, 0.0))]
    return [(0, (-0.62, 1.42, 0.0)), (111, (-0.62, 1.42, 0.0)), (112, (PH.x + 0.03, PH.y + 0.6, 0.0), linear)]


def hop_z(name, f):
    z = 0.0
    if name == "bride" and 17 <= f <= 35:
        t = (f - 17) / 18
        z += 0.1 * abs(math.sin(math.pi * 5 * t))
    if 98 <= f <= 112:  # the jump on the 4th photo
        t = (f - 98) / 14
        z += 0.36 * 4 * t * (1 - t)
    return z


@lru_cache(maxsize=None)
def state(name, f):
    p = pose_at(POSES[name], f)
    rk = root_keys(name)
    x, y, rz = tween(f, [(k[0], k[1], k[2] if len(k) > 2 else in_out) for k in rk])
    ph = PHASE[name]
    breath = 1 + 0.014 * math.sin(2 * math.pi * f / 52 + ph)
    sq = p["squash"] * breath
    eyes = p["eyes"]
    for b in BLINKS[name]:
        d = f - b
        if 0 <= d <= 3:
            eyes = min(eyes, [0.45, 0.08, 0.3, 0.75][d])
    sl = list(p["sl"])
    sr = list(p["sr"])
    bob = 0.012 * math.sin(2 * math.pi * f / 40 + ph * 2)
    sl[2] += bob
    sr[2] += 0.012 * math.sin(2 * math.pi * f / 40 + ph * 2 + 1.1)
    if p["wave_sl"] > 0:
        sl[0] += 0.055 * p["wave_sl"] * math.sin(2 * math.pi * f / 8)
        sl[2] += 0.02 * p["wave_sl"] * math.cos(2 * math.pi * f / 8)
    if p["wave_sr"] > 0:
        sr[0] += 0.055 * p["wave_sr"] * math.sin(2 * math.pi * f / 8 + 1)
        sr[2] += 0.02 * p["wave_sr"] * math.cos(2 * math.pi * f / 8 + 1)
    if p["run"] > 0:
        sw = math.sin(2 * math.pi * (f - 17) / 7.2)
        sl[2] += 0.12 * sw * p["run"]
        sr[2] -= 0.12 * sw * p["run"]
    mouth = p["mouth"]
    head = list(p["head"])
    lean = list(p["lean"])
    if name == "friend" and 320 <= f < 388:   # talking into the audio guestbook
        syl = 0.5 + 0.5 * math.sin(2 * math.pi * f / 4.3 + 0.7 * math.sin(f * 0.37))
        talk = 0.25 + 0.6 * syl
        if 362 <= f < 376:                      # laugh: wide open, bouncing
            talk = 0.75 + 0.2 * math.sin(2 * math.pi * f / 3)
        mouth = max(mouth, talk)
        head[1] += 0.04 * math.sin(2 * math.pi * f / 23)
        lean[1] += 0.02 * math.sin(2 * math.pi * f / 31)
    return dict(x=x, y=y, rz=rz, z=hop_z(name, f), sq=sq, eyes=eyes, sl=sl, sr=sr, head=head, lean=lean,
                mouth=mouth, prop=p["prop"])


# ---- phone / handset (S4) ----------------------------------------------------------
FRIEND_PHONE_ROOT = Vector((PH.x + 0.03, PH.y + 0.6, 0.0))


def handset_world(f):
    """(location, euler) of the handset."""
    rest_loc, rest_rot = HANDSET_REST
    ear = FRIEND_PHONE_ROOT + Vector((-0.235, 0.005, NECK_Z + 0.2))
    mouth_p = FRIEND_PHONE_ROOT + Vector((-0.13, -0.2, NECK_Z + 0.06))
    xa = (mouth_p - ear).normalized()
    za = Vector((1, 0, 0))
    za = (za - xa * za.dot(xa)).normalized() * -1   # cups (local -Z) face the head (+X)
    ya = za.cross(xa)
    mat = Matrix((xa, ya, za)).transposed()
    held_rot = mat.to_euler()
    held_loc = (ear + mouth_p) / 2 - za * 0.012
    if f < 306 or f >= S5[1] + 100:
        return rest_loc, rest_rot
    t = clamp((f - 306) / 10)
    e = in_out(t)
    loc = rest_loc.lerp(held_loc, e) + Vector((0, 0, 0.1 * math.sin(math.pi * e)))
    q = rest_rot.to_quaternion().slerp(held_rot.to_quaternion(), e)
    if f >= 320:
        loc = held_loc + Vector((0, 0, 0.006 * math.sin(2 * math.pi * f / 23)))
    return loc, q.to_euler()


def handset_grip(f):
    loc, rot = handset_world(f)
    m = rot.to_matrix()
    return loc + m @ Vector((0.02, 0, 0.035 * PHONE_S))


def hand_local(name, side, f):
    s = state(name, f)
    if name == "friend" and side == "sl" and 300 <= f:
        rest = Vector(s["sl"])
        grip = handset_grip(f) - FRIEND_PHONE_ROOT
        if f < 306:
            return tuple(rest.lerp(grip, out_cubic(clamp((f - 300) / 6))))
        return tuple(grip)
    return tuple(s[side])


for ch in (bride, groom, friend):
    nm = ch.name
    animate(ch.root, "location", lambda f, nm=nm: (state(nm, f)["x"], state(nm, f)["y"], state(nm, f)["z"]),
            constant=(nm == "friend"))  # friend teleports to the phone between shots: no motion-blur smear
    animate(ch.root, "rotation_euler", lambda f, nm=nm: (0.0, 0.0, state(nm, f)["rz"]))

    def body_scale(f, nm=nm):
        sq = state(nm, f)["sq"]
        w = 1 / math.sqrt(max(sq, 0.2))
        return (w, w, sq)

    animate(ch.body, "scale", body_scale)
    animate(ch.body, "rotation_euler", lambda f, nm=nm: (state(nm, f)["lean"][0], state(nm, f)["lean"][1], 0.0))
    animate(ch.head, "rotation_euler", lambda f, nm=nm: tuple(state(nm, f)["head"]))
    for side in ("sl", "sr"):
        animate(ch.hands[side], "location", lambda f, nm=nm, side=side: hand_local(nm, side, f))
    for e in ch.eyes:
        animate(e, "scale", lambda f, nm=nm: (0.8, 0.45, 1.15 * max(0.07, state(nm, f)["eyes"])))
    animate(ch.mouth, "scale", lambda f, nm=nm: (0.62 + 0.38 * state(nm, f)["mouth"], 0.42, 0.85 * state(nm, f)["mouth"]))
    animate(ch.smile, "scale", lambda f, nm=nm: (max(0.0, 1 - 1.3 * state(nm, f)["mouth"]),) * 3)


def prop_scale(nm):
    def fn(f):
        s = clamp(state(nm, f)["prop"], 0, 1.3)
        return (s, s, s)
    return fn


animate(mustache, "scale", prop_scale("groom"))
animate(sign, "scale", prop_scale("friend"))


def heart_scale(f):
    if f < 79 or f > 98:
        return (0.0, 0.0, 0.0)
    s = out_back(clamp((f - 79) / 7), 2.4) if f < 93 else 1 - in_cubic(clamp((f - 93) / 5))
    return (s, s, s)


animate(heart_pop.children[0], "scale", heart_scale)
animate(heart_pop, "location", lambda f: (0.16, 1.28, 1.2 + 0.01 * max(0, f - 79)))
animate(heart_pop, "rotation_euler", lambda f: (0, 0.12 * math.sin(f * 0.5), 0))

# ---- handset, cord, sound rings ----------------------------------------------------
animate(handset, "location", lambda f: tuple(handset_world(f)[0]))
animate(handset, "rotation_euler", lambda f: tuple(handset_world(f)[1]))

CORD_A = Vector((PH.x + 0.1 * PHONE_S, PH.y - 0.04, PH_TOP + 0.02))


@lru_cache(maxsize=None)
def cord_points(f):
    loc, rot = handset_world(f)
    b = loc + rot.to_matrix() @ Vector((0.11 * PHONE_S, 0, -0.03 * PHONE_S))
    a = CORD_A
    mid = (a + b) / 2 + Vector((0.07, -0.04, -0.12 - 0.05 * (b - a).length))
    pts = []
    for i in range(CORD_N):
        t = i / (CORD_N - 1)
        p = a * (1 - t) ** 2 + mid * 2 * t * (1 - t) + b * t * t
        tan = (2 * (1 - t) * (mid - a) + 2 * t * (b - mid)).normalized()
        n1 = tan.cross(Vector((0, 0, 1)))
        if n1.length < 1e-3:
            n1 = Vector((1, 0, 0))
        n1.normalize()
        n2 = tan.cross(n1)
        th = 2 * math.pi * 24 * t
        coil = 0.0085 * min(1.0, 8 * t, 8 * (1 - t))
        pts.append(p + (n1 * math.cos(th) + n2 * math.sin(th)) * coil)
    return pts


for i in range(CORD_N):
    animate(cord.data, f"splines[0].points[{i}].co", lambda f, i=i: (*cord_points(f)[i], 1.0), n=4)

RING_ORIGIN = FRIEND_PHONE_ROOT + Vector((0.0, -0.12, NECK_Z + 0.2))
for k, r in enumerate(RINGS):
    def ring_state(f, k=k):
        best = None
        for start in range(322 + k * 9, 384, 27):
            if start <= f < start + 22:
                best = (f - start) / 22
        return best

    def ring_scale(f, k=k):
        t = ring_state(f, k)
        if t is None:
            return (0.0, 0.0, 0.0)
        s = 2.6 + 1.8 * out_cubic(t)
        return (s, s, s)

    def ring_color(f, k=k):
        t = ring_state(f, k)
        a = 0.0 if t is None else 0.8 * (1 - t) ** 1.5 * min(1.0, t * 5)
        return (1.0, 1.0, 1.0, a)

    r.location = RING_ORIGIN
    r.rotation_euler = (Vector((1.53, -0.92, 1.04)) - RING_ORIGIN).to_track_quat("Z", "Y").to_euler()
    animate(r, "scale", ring_scale)
    for ch_ in r.children:
        animate(ch_, "color", ring_color, n=4)

# ---- the printed strip (S2) ---------------------------------------------------------
SLOT = Vector((0, BOOTH_FRONT + 0.004, 0.588))


def strip_print_xf(f):
    e_t = clamp((f - 122) / (168 - 122))
    ext = STRIP_L * (e_t - 0.8 * math.sin(2 * math.pi * 6 * e_t) / (2 * math.pi * 6))
    droop = -0.05 * e_t
    loc = SLOT + Vector((0, ext - STRIP_L, 0))
    if f >= 177:
        t = clamp((f - 177) / 14)
        droop = lerp(-0.05, -1.35, in_quad(t))
        loc = SLOT + Vector((0, 0.004, -0.002))
        if f >= 188:
            loc += Vector((0, 0.025 * (f - 188), -0.0026 * (f - 188) ** 2))
    return loc, (droop, 0.0, 0.0)


strip_print.location = SLOT
animate(strip_print, "location", lambda f: tuple(strip_print_xf(f)[0]))
animate(strip_print, "rotation_euler", lambda f: strip_print_xf(f)[1])
visible_during(strip_print, (S2[0], S3[0]))

# ---- the strip landing in the guestbook (S3+) ---------------------------------------
STRIP_REST = Vector((GB.x - 0.118, GB.y - 0.005, GB_TOP + 0.0235))


def strip_book_xf(f):
    if f < S3[0]:
        return STRIP_REST + Vector((0, 0, 1.0)), (0, 0, 0)
    t = clamp((f - 210) / 12)
    z = lerp(0.55, 0.0, in_quad(t))
    if 222 <= f < 230:
        z = 0.012 * math.sin(math.pi * (f - 222) / 8) * (1 - (f - 222) / 10)
    rz = lerp(0.55, -0.075, out_cubic(clamp((f - 210) / 16)))
    wob = (1 - clamp((f - 210) / 18))
    rx = 0.35 * wob * math.sin(f * 0.9)
    ry = 0.25 * wob * math.cos(f * 0.7)
    return STRIP_REST + Vector((0.03 * wob, 0.02 * wob, z)), (rx, ry, rz)


animate(strip_book, "location", lambda f: tuple(strip_book_xf(f)[0]))
animate(strip_book, "rotation_euler", lambda f: strip_book_xf(f)[1])
visible_during(strip_book, (S3[0], RENDER_FRAMES + 1))

# ---- handwriting + pen (S3) --------------------------------------------------------
bpy.context.view_layer.update()
W1 = line1.dimensions.x + 0.004
W2 = line2.dimensions.x + 0.004
L1 = (230, 252)
L2 = (256, 279)
HEART_T = (282, 293)


def reveal(f, rng, width):
    t = clamp((f - rng[0]) / (rng[1] - rng[0]))
    t = t + 0.03 * math.sin(t * math.pi * 7) * (1 - t) * t * 4   # uneven hand speed
    return -0.005 + (width + 0.008) * clamp(t)


animate(line1, "color", lambda f: (reveal(f, L1, W1), 0.0, 0.0, 1.0), n=4)
animate(line2, "color", lambda f: (reveal(f, L2, W2), 0.0, 0.0, 1.0), n=4)
animate(heart_doodle.data, "bevel_factor_end",
        lambda f: max(0.0, clamp((f - HEART_T[0]) / (HEART_T[1] - HEART_T[0]))), n=1)

heart_cum = [0.0]
for a_, b_ in zip(heart_pts[:-1], heart_pts[1:]):
    heart_cum.append(heart_cum[-1] + (Vector(b_) - Vector(a_)).length)


def heart_point(u):
    target = u * heart_cum[-1]
    for i in range(1, len(heart_cum)):
        if heart_cum[i] >= target:
            seg = heart_cum[i] - heart_cum[i - 1]
            t = 0 if seg == 0 else (target - heart_cum[i - 1]) / seg
            return HEART_ORIGIN + Vector(heart_pts[i - 1]).lerp(Vector(heart_pts[i]), t)
    return HEART_ORIGIN + Vector(heart_pts[-1])


PEN_DIR = Vector((0.42, -0.38, 0.82)).normalized()
PEN_REST = (Vector((GB.x + 0.2, GB.y - 0.11, GB_TOP + 0.03)), Vector((-0.35, 1, 0.04)).normalized())
OFF = Vector((GB.x + 0.45, GB.y - 0.55, GB_TOP + 0.25))


@lru_cache(maxsize=None)
def pen_tip(f):
    """returns (tip position, writing?)"""
    def on_line(line, rng, width, f):
        x = reveal(f, rng, width)
        base = Vector(line.location)
        stroke = 0.5 + 0.5 * math.sin(2 * math.pi * x / 0.0115)
        return base + Vector((x, 0.004 + 0.022 * stroke, 0.0))

    if f < 224:
        return OFF, False
    if f < L1[0]:
        p0 = on_line(line1, L1, W1, L1[0]) + Vector((0, 0, 0.02))
        return OFF.lerp(p0, out_cubic((f - 224) / (L1[0] - 224))), False
    if f <= L1[1]:
        return on_line(line1, L1, W1, f), True
    if f < L2[0]:
        a = on_line(line1, L1, W1, L1[1])
        b = on_line(line2, L2, W2, L2[0])
        t = in_out((f - L1[1]) / (L2[0] - L1[1]))
        return a.lerp(b, t) + Vector((0, 0, 0.018 * math.sin(math.pi * t))), False
    if f <= L2[1]:
        return on_line(line2, L2, W2, f), True
    if f < HEART_T[0]:
        a = on_line(line2, L2, W2, L2[1])
        b = heart_point(0.0)
        t = in_out((f - L2[1]) / (HEART_T[0] - L2[1]))
        return a.lerp(b, t) + Vector((0, 0, 0.018 * math.sin(math.pi * t))), False
    if f <= HEART_T[1]:
        return heart_point(clamp((f - HEART_T[0]) / (HEART_T[1] - HEART_T[0]))), True
    t = out_cubic(clamp((f - HEART_T[1]) / 8))
    return heart_point(1.0).lerp(OFF, in_cubic(clamp((f - HEART_T[1] - 2) / 6))) + Vector((0, 0, 0.03 * t)), False


def pen_xf(f):
    if S3[0] <= f < S3[1]:
        tip, writing = pen_tip(f)
        tip = tip + Vector((0, 0, 0.0 if writing else 0.004))
        d = (PEN_DIR + Vector((0.04 * math.sin(f * 0.6), 0.03 * math.cos(f * 0.45), 0))).normalized()
        return tip, d.to_track_quat("Z", "Y").to_euler()
    loc, d = PEN_REST
    q = d.to_track_quat("Z", "Y") @ Quaternion((0, 0, 1), 0.3)
    return loc, q.to_euler()


animate(pen, "location", lambda f: tuple(pen_xf(f)[0]), constant=True)
animate(pen, "rotation_euler", lambda f: tuple(pen_xf(f)[1]), constant=True)


def guest_hand_loc(f):
    tip, rot = pen_xf(f)
    d = rot.to_matrix() @ Vector((0, 0, 1))
    return tuple(tip + d * 0.098 + Vector((0.01, -0.006, 0.0)))


animate(guest_hand, "location", guest_hand_loc, constant=True)
visible_during(guest_hand, S3)

# ---- lights ----------------------------------------------------------------------
animate(flash.data, "energy", lambda f: sum(420 * pulse(f, fl, 0, 1.6) for fl in FLASHES), n=1)

# --------------------------------------------------------------------------------------
# cameras
# --------------------------------------------------------------------------------------
def make_cam(name, hfov_deg, fstop, clip=0.01):
    cd = bpy.data.cameras.new(name)
    cd.sensor_fit = "HORIZONTAL"
    cd.sensor_width = 36
    cd.lens = 18 / math.tan(math.radians(hfov_deg) / 2)
    cd.clip_start = clip
    cd.clip_end = 120
    cd.dof.use_dof = True
    cd.dof.aperture_fstop = fstop
    cd.dof.aperture_blades = 7
    ob = bpy.data.objects.new(name, cd)
    return link(ob)


def lens_for(hfov):
    return 18 / math.tan(math.radians(hfov) / 2)


def cam_anim(cam, fn, fov_fn=None, focus_fn=None, fstop_fn=None):
    prev = [None]

    def rot(f):
        pos, tgt, roll = fn(f)
        e = look_rot(pos, tgt, roll)
        return tuple(e)

    animate(cam, "location", lambda f: tuple(fn(f)[0]))
    animate(cam, "rotation_euler", rot)
    if fov_fn:
        animate(cam.data, "lens", lambda f: lens_for(fov_fn(f)), n=1)

    def focus(f):
        if focus_fn:
            return focus_fn(f)
        pos, tgt, _ = fn(f)
        return (Vector(tgt) - Vector(pos)).length

    animate(cam.data, "dof.focus_distance", focus, n=1)
    if fstop_fn:
        animate(cam.data, "dof.aperture_fstop", fstop_fn, n=1)


CAMS = {}

# S1a: over the bride's shoulder, her mitten taps the screen
CAMS["s1a"] = make_cam("cam_s1a_tap", 44, 2.0)
cam_anim(CAMS["s1a"], lambda f: (Vector((0.4, -0.72, 0.97)).lerp(Vector((0.36, -0.78, 0.96)), f / 15)
                                 + Vector((wobble(f, 1, 0.003), 0, wobble(f, 2, 0.002))),
                                 Vector((0.02, BOOTH_FRONT, 0.97)), 0.0),
         focus_fn=lambda f: (Vector((0.02, BOOTH_FRONT, 0.85)) - Vector((0.38, -0.73, 1.09))).length)

# S1b: the booth's own camera (POV of the photobooth)
BOOTH_CAM_POS = Vector((0.0, BOOTH_FRONT + 0.03, 1.08))
CAMS["s1b"] = make_cam("cam_s1b_booth_pov", 39, 4.5)
cam_anim(CAMS["s1b"], lambda f: (BOOTH_CAM_POS, Vector((-0.09, 1.35, 1.13)), 0.0),
         fov_fn=lambda f: lerp(39.5, 37.0, smooth(clamp((f - 15) / 97))),
         focus_fn=lambda f: 2.66)

# cell camera for the 4 strip photos (landscape 3:2) — rendered separately
CAMS["cell"] = make_cam("cam_cell_strip", 60, 5.6)
CAMS["cell"].location = BOOTH_CAM_POS
CAMS["cell"].rotation_euler = look_rot(BOOTH_CAM_POS, Vector((-0.09, 1.35, 0.98)))
CAMS["cell"].data.dof.focus_distance = 2.66


# S2: printer slot, ends in a whip-pan down
def s2_cam(f):
    t = clamp((f - S2[0]) / (193 - S2[0]))
    pos = Vector((0.1, -0.96, 0.86)).lerp(Vector((0.07, -1.01, 0.83)), smooth(t))
    tgt = Vector((0.0, BOOTH_FRONT + 0.1, 0.565))
    w = clamp((f - 193) / 17)
    tgt = tgt + Vector((0, 0.18 * in_cubic(w), -0.9 * in_cubic(w)))
    pos = pos + Vector((wobble(f, 3, 0.004), wobble(f, 4, 0.003), wobble(f, 5, 0.003)))
    return pos, tgt, 0.0


CAMS["s2"] = make_cam("cam_s2_print", 40, 2.4)
cam_anim(CAMS["s2"], s2_cam, focus_fn=lambda f: (Vector((0.0, BOOTH_FRONT + 0.14, 0.585)) - s2_cam(f)[0]).length)


# S3: top-down over the guestbook
def s3_cam(f):
    t = clamp((f - S3[0]) / (S3[1] - S3[0]))
    h = lerp(0.6, 0.5, in_out(t))
    c = Vector((GB.x + 0.005, GB.y - 0.01, GB_TOP))
    pos = c + Vector((wobble(f, 6, 0.004), wobble(f, 7, 0.004), h))
    tgt = c + Vector((0, 0.012, 0))
    return pos, tgt, lerp(0.035, -0.02, smooth(t))


CAMS["s3"] = make_cam("cam_s3_guestbook", 47, 3.2)
cam_anim(CAMS["s3"], s3_cam, focus_fn=lambda f: s3_cam(f)[0].z - GB_TOP - 0.02)


# S4: audio guestbook phone
def s4_cam(f):
    t = clamp((f - S4[0]) / (S4[1] - S4[0]))
    pos = Vector((1.58, -0.74, 0.96)).lerp(Vector((1.63, -0.64, 0.94)), smooth(t))
    pos += Vector((wobble(f, 8, 0.004), wobble(f, 9, 0.004), wobble(f, 10, 0.003)))
    tgt = Vector((PH.x - 0.02, PH.y + 0.38, 0.95))
    return pos, tgt, 0.0


CAMS["s4"] = make_cam("cam_s4_phone", 42, 2.8)
cam_anim(CAMS["s4"], s4_cam, focus_fn=lambda f: (Vector((PH.x - 0.05, PH.y + 0.45, 0.85)) - s4_cam(f)[0]).length)


# S5: crane pull-back to the whole corner
def s5_cam(f):
    t = in_out(clamp((f - S5[0]) / (S5[1] - S5[0])))
    pos = Vector((0.14, 0.32, 1.0)).lerp(Vector((0.0, -5.5, 4.1)), t)
    pos.z = 1.0 + (4.1 - 1.0) * t ** 0.75
    tgt = Vector((0.14, 1.4, 0.92)).lerp(Vector((0.0, 0.35, 0.5)), t)
    return pos, tgt, 0.0


CAMS["s5"] = make_cam("cam_s5_reveal", 40, 4.0)
cam_anim(CAMS["s5"], s5_cam, fov_fn=lambda f: lerp(40, 50, in_out(clamp((f - S5[0]) / 45))),
         fstop_fn=lambda f: lerp(4.0, 14.0, clamp((f - S5[0]) / 40)))

for key, start in (("s1a", S1A[0]), ("s1b", S1B[0]), ("s2", S2[0]), ("s3", S3[0]), ("s4", S4[0]), ("s5", S5[0])):
    mk = SCENE.timeline_markers.new(key, frame=sf(start))
    mk.camera = CAMS[key]
SCENE.camera = CAMS["s1a"]

# --------------------------------------------------------------------------------------
# bake + render settings + save
# --------------------------------------------------------------------------------------
bake()

SCENE.frame_start = 1
SCENE.frame_end = RENDER_FRAMES
SCENE.render.fps = FPS
SCENE.render.resolution_x = 1080
SCENE.render.resolution_y = 1920
SCENE.render.resolution_percentage = 80
SCENE.render.engine = "CYCLES"
cy = SCENE.cycles
cy.device = "CPU"
cy.samples = 14
cy.use_adaptive_sampling = True
cy.adaptive_threshold = 0.025
cy.adaptive_min_samples = 8
cy.use_denoising = True
cy.denoiser = "OPENIMAGEDENOISE"
cy.denoising_input_passes = "RGB_ALBEDO_NORMAL"
cy.max_bounces = 4
cy.diffuse_bounces = 2
cy.glossy_bounces = 2
cy.transmission_bounces = 4
cy.transparent_max_bounces = 10
cy.volume_bounces = 0
cy.caustics_reflective = False
cy.caustics_refractive = False
cy.sample_clamp_indirect = 6.0
cy.use_light_tree = False  # few real lights left; ~20% faster, no visible difference
SCENE.render.use_motion_blur = True
SCENE.render.motion_blur_shutter = 0.45
SCENE.render.use_persistent_data = True
SCENE.view_settings.view_transform = "AgX"
SCENE.view_settings.look = "AgX - Medium High Contrast"
SCENE.view_settings.exposure = -0.2
SCENE.render.film_transparent = False
SCENE.render.image_settings.file_format = "PNG"
SCENE.render.image_settings.color_depth = "8"
SCENE.render.image_settings.color_mode = "RGB"

os.makedirs(os.path.dirname(os.path.abspath(ARGS.out)), exist_ok=True)
bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(ARGS.out))
print("SAVED", ARGS.out, "objects:", len(bpy.data.objects), "anim channels:", len(ANIM))
