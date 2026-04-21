#!/usr/bin/env python3
"""30-second 1920x1080 teaser video for GraspAuto.

Each object is shown for ~6 seconds with TWO distinct grasps at different tap
locations, and the camera is fixed per-object (computed once from the object's
bbox, not per-frame from obj+hand bbox) so the view stays rock-steady through
all the fades and rotations.

Object timeline (6 s each):
  0.0-0.4 s  mesh appears (fade in on the object only)
  0.4-0.6 s  grasp-1 sphere fades in
  0.6-1.2 s  grasp-1 hand fades in and materialises
  1.2-2.6 s  hold grasp-1 (1.4 s), camera orbits 120 degrees
  2.6-2.9 s  grasp-1 hand + sphere fade out
  2.9-3.1 s  grasp-2 sphere fades in
  3.1-3.7 s  grasp-2 hand fades in
  3.7-6.0 s  hold grasp-2 (2.3 s), camera orbits 360 degrees

Output: paper/teaser_video.mp4
"""
from pathlib import Path
import os, sys, json, subprocess
# Resolve project root from this file's location (works from any cwd)
REPO = Path(__file__).resolve().parent.parent
os.chdir(str(REPO))
sys.path.insert(0, 'src')

import numpy as np
import torch
import open3d as o3d
import trimesh
import cv2
from PIL import Image, ImageDraw, ImageFont

from graspauto.mano_decoder import MangoMANODecoder

EVAL = 'outputs/graspauto_sphere_r047/eval_bo64sel/per_sample.json'
VAL  = 'outputs/stage3_contact_graph/val_sphere.pt'
MESH_DIR = 'data/contactpose/data_raw/contactpose_ply_files_mm'
OUT_DIR  = 'paper/teaser_video_frames'
VIDEO_OUT = 'paper/teaser_video.mp4'

W, H = 1920, 1080
FPS  = 30
PER_OBJ_SECONDS = 6.0
FRAMES_PER_OBJ = int(PER_OBJ_SECONDS * FPS)   # 180

# Frame milestones for a single object (inclusive-exclusive). Phases B (sphere
# fade in) and C (hand materialise) are both lengthened relative to the
# earlier timings so the tap-to-grasp transition reads deliberately, not rushed.
F_A_END  = int(0.4 * FPS)   # 12  mesh appears
F_B1_END = int(0.9 * FPS)   # 27  sphere 1 fades in (0.5 s)
F_C1_END = int(1.9 * FPS)   # 57  hand 1 materialises (1.0 s)
F_D1_END = int(2.8 * FPS)   # 84  hold grasp 1 (0.9 s)
F_X_END  = int(3.1 * FPS)   # 93  grasp 1 fades out (0.3 s)
F_B2_END = int(3.6 * FPS)   # 108 sphere 2 fades in (0.5 s)
F_C2_END = int(4.6 * FPS)   # 138 hand 2 materialises (1.0 s)
# Phase D2: F_C2_END .. FRAMES_PER_OBJ  — hold grasp 2 (1.4 s)

# Two grasps per object: (label, val_idx)
OBJECTS = [
    ('mouse', False, [('Palm centre grip', 67),
                       ('Side-of-mouse grip', 46)]),
    ('mug',   False, [('Handle-side grasp', 47),
                       ('Body-side grasp',   116)]),
    ('hammer', False, [('Handle grip', 160),
                        ('Head grip',   135)]),
    ('flashlight', False, [('Near end grip', 110),
                            ('Mid-body grip', 159)]),
    ('bowl', True, [('Rim grip (side A)', 18),
                     ('Rim grip (side B)', 56)]),
]

# Per-object camera (azim, elev) — tuned so both taps are visible.
CAMERA = {
    'mouse':      (-90.0, 28.0),
    'mug':        (-110.0, 18.0),
    'hammer':     (90.0,  22.0),
    'flashlight': (-30.0, 18.0),
    'bowl':       (60.0,  32.0),
}

# --- Data + model preload -----------------------------------------------------
recs = {r['sample_index']: r for r in json.load(open(EVAL))['records']}
val = torch.load(VAL, map_location='cpu', weights_only=False)

decoder = MangoMANODecoder()
hand_faces = decoder.faces.cpu().numpy()


def decode_mano(x1_54):
    x = torch.tensor(x1_54).unsqueeze(0).float()
    with torch.no_grad():
        out = decoder(x)
    return out['vertices'][0].numpy()


def load_obj_mesh(name):
    m = trimesh.load(os.path.join(MESH_DIR, f'{name}.ply'), force='mesh')
    v = np.asarray(m.vertices) * 0.001
    f = np.asarray(m.faces)
    if len(f) > 50000:
        m2 = m.simplify_quadric_decimation(face_count=15000)
        v = np.asarray(m2.vertices) * 0.001
        f = np.asarray(m2.faces)
    return v, f


def sphere_params(idx):
    c = val['unified_centroid'][idx].numpy()
    r = float(val['unified_spread'][idx].norm())
    return c, r


# --- Renderer ---------------------------------------------------------------
VIEW_W, VIEW_H = 900, 900
renderer = o3d.visualization.rendering.OffscreenRenderer(VIEW_W, VIEW_H)
renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])


def snap_white(img):
    r, g, b = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
    brightness = (r + g + b) / 3
    spread = np.max([np.abs(r - g), np.abs(g - b), np.abs(r - b)], axis=0)
    mask = (brightness >= 220) & (spread <= 5)
    img[mask] = 255
    return img


def fixed_camera(obj_v, azim_deg, elev_deg):
    """Compute a fixed camera from the object bbox alone, so nothing jitters
    when hand/sphere fade in and out."""
    ctr = obj_v.mean(0)
    diag = np.linalg.norm(obj_v.max(0) - obj_v.min(0))
    # Allow extra room so a grasping hand (~= half object diag) fits the frame.
    dist = max(0.24, diag * 1.55)
    az, el = np.deg2rad(azim_deg), np.deg2rad(elev_deg)
    direction = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    cam_pos = ctr + direction * dist
    return ctr, cam_pos


def rot_z(deg):
    t = np.deg2rad(deg)
    return np.array([[np.cos(t), -np.sin(t), 0.0],
                     [np.sin(t),  np.cos(t), 0.0],
                     [0.0,        0.0,       1.0]])


def render_scene(obj_v, obj_f, hand_v, sph_center, sph_radius,
                  cam_ctr, cam_pos, rotation_deg,
                  hand_alpha, sphere_alpha):
    """Render an RGB frame. cam_ctr / cam_pos are fixed for the whole object.
    Sphere is rendered LAST so it's always visible even through the hand (acts
    like an AR marker over the grasp). This matters for objects like the mouse
    where the hand would otherwise fully occlude the tap location."""
    Rz = rot_z(rotation_deg)
    obj_pivot = obj_v.mean(0)

    def rot(pts): return (pts - obj_pivot) @ Rz.T + obj_pivot

    obj_v_r = rot(obj_v)

    # Object-only pass (baseline).
    renderer.scene.clear_geometry()
    obj_mesh = o3d.geometry.TriangleMesh()
    obj_mesh.vertices = o3d.utility.Vector3dVector(obj_v_r)
    obj_mesh.triangles = o3d.utility.Vector3iVector(obj_f)
    obj_mesh.compute_vertex_normals()
    obj_mesh.paint_uniform_color([0.78, 0.78, 0.82])
    mat_o = o3d.visualization.rendering.MaterialRecord(); mat_o.shader = 'defaultLit'
    renderer.scene.add_geometry('obj', obj_mesh, mat_o)
    renderer.setup_camera(45.0, cam_ctr.tolist(), cam_pos.tolist(), [0, 0, 1])
    img_obj = snap_white(np.asarray(renderer.render_to_image()).copy()).astype(np.uint8)

    # Hand pass (composite onto obj).
    out = img_obj.copy()
    if hand_v is not None and hand_alpha > 0.01:
        renderer.scene.clear_geometry()
        renderer.scene.add_geometry('obj', obj_mesh, mat_o)
        hand_rot = rot(hand_v)
        hand_mesh = o3d.geometry.TriangleMesh()
        hand_mesh.vertices = o3d.utility.Vector3dVector(hand_rot)
        hand_mesh.triangles = o3d.utility.Vector3iVector(hand_faces)
        hand_mesh.compute_vertex_normals()
        hand_mesh.paint_uniform_color([0.95, 0.55, 0.55])
        mat_h = o3d.visualization.rendering.MaterialRecord(); mat_h.shader = 'defaultLit'
        renderer.scene.add_geometry('hand', hand_mesh, mat_h)
        renderer.setup_camera(45.0, cam_ctr.tolist(), cam_pos.tolist(), [0, 0, 1])
        img_hand = snap_white(np.asarray(renderer.render_to_image()).copy())
        diff = np.abs(img_hand.astype(int) - img_obj.astype(int)).sum(axis=-1)
        hand_mask = diff > 10
        out[hand_mask] = (
            hand_alpha * img_hand[hand_mask].astype(float)
            + (1 - hand_alpha) * img_obj[hand_mask]
        ).astype(np.uint8)

    # Sphere pass (drawn LAST, sits on top of hand so AR tap marker is always visible).
    if sphere_alpha > 0.01:
        renderer.scene.clear_geometry()
        sph_r = float(sph_radius) * 0.40
        sph_c_rot = rot(np.array(sph_center)[None])[0]
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sph_r, resolution=24)
        sphere.translate(sph_c_rot.tolist())
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color([0.98, 0.82, 0.10])
        mat_s = o3d.visualization.rendering.MaterialRecord(); mat_s.shader = 'defaultUnlit'
        renderer.scene.add_geometry('sphere', sphere, mat_s)
        renderer.setup_camera(45.0, cam_ctr.tolist(), cam_pos.tolist(), [0, 0, 1])
        img_sphere_only = np.asarray(renderer.render_to_image()).copy()
        r = img_sphere_only[..., 0].astype(int)
        g = img_sphere_only[..., 1].astype(int)
        b = img_sphere_only[..., 2].astype(int)
        brightness = (r + g + b) / 3
        spread = np.max([np.abs(r - g), np.abs(g - b), np.abs(r - b)], axis=0)
        sphere_mask = ~((brightness >= 220) & (spread <= 5))
        out_f = out.astype(float)
        out_f[sphere_mask] = (
            sphere_alpha * img_sphere_only[sphere_mask].astype(float)
            + (1 - sphere_alpha) * out_f[sphere_mask]
        )
        out = out_f.astype(np.uint8)

    return out


# --- Label overlay via Pillow -------------------------------------------------
try:
    FONT_TITLE = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 72)
    FONT_SUB   = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 42)
    FONT_TAG   = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 46)
    FONT_FOOT  = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 34)
except Exception:
    FONT_TITLE = ImageFont.load_default()
    FONT_SUB = ImageFont.load_default()
    FONT_TAG = ImageFont.load_default()
    FONT_FOOT = ImageFont.load_default()


def compose_frame(scene_img, obj_name, phase_label, alpha_title, is_unseen):
    """Layout: scene centred full-height, object name + optional phase label
    tucked into the bottom-right corner. No logo / footer / side panel — the
    project page already owns that branding."""
    frame = Image.new('RGB', (W, H), (255, 255, 255))
    # Scene: render at full 1080x1080 (square) and centre horizontally.
    scene = Image.fromarray(scene_img).resize((H, H), Image.LANCZOS)   # 1080x1080
    frame.paste(scene, ((W - H) // 2, 0))                               # centred

    draw = ImageDraw.Draw(frame)

    # Bottom-right caption block (right-aligned via anchor='rs' = right-baseline).
    margin_r = 60
    line_y = H - 60
    obj_color = (142, 76, 42) if is_unseen else (42, 93, 142)
    draw.text((W - margin_r, line_y), obj_name,
              fill=obj_color, font=FONT_TITLE, anchor='rs')
    if is_unseen:
        draw.text((W - margin_r, line_y - 78),
                  'UNSEEN object class',
                  fill=(142, 76, 42), font=FONT_SUB, anchor='rs')
    if phase_label and alpha_title > 0.01:
        a = float(max(0.0, min(1.0, alpha_title)))
        r0, g0, b0 = 70, 70, 80
        tag_color = tuple(int(round(a * c + (1 - a) * 255)) for c in (r0, g0, b0))
        offset = 78 + (78 if is_unseen else 0)
        draw.text((W - margin_r, line_y - offset),
                  phase_label, fill=tag_color, font=FONT_SUB, anchor='rs')
    return np.asarray(frame)


def ease_in_out(t):
    t = float(max(0.0, min(1.0, t)))
    return t * t * (3 - 2 * t)


# --- Main loop ---------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
# Clean stale frames so we don't muxing old content into the new video.
for p in os.listdir(OUT_DIR):
    if p.endswith('.png'):
        os.remove(os.path.join(OUT_DIR, p))

print('pre-decoding grasps and meshes...')
meshes = {name: load_obj_mesh(name) for name, _, _ in OBJECTS}
grasps = {}
for name, _unseen, pairs in OBJECTS:
    grasps[name] = []
    for label, idx in pairs:
        hv = decode_mano(recs[idx]['picked_x1'])
        sc, sr = sphere_params(idx)
        grasps[name].append((label, hv, sc, sr))

frame_idx = 0
for obj_i, (name, is_unseen, pairs) in enumerate(OBJECTS):
    print(f'[{obj_i+1}/{len(OBJECTS)}] rendering {name} ({FRAMES_PER_OBJ} frames)')
    obj_v, obj_f = meshes[name]
    azim, elev = CAMERA[name]
    cam_ctr, cam_pos = fixed_camera(obj_v, azim, elev)

    (lbl1, hv1, sc1, sr1), (lbl2, hv2, sc2, sr2) = grasps[name]

    # Continuous monotonic rotation across the whole 6 s object clip.
    # Angular velocity is constant (~60 deg/s) so there are no visible "stops"
    # between phases. Phase transitions only change alpha layers, not rotation.
    TOTAL_ROT_DEG = 360.0
    for k in range(FRAMES_PER_OBJ):
        rotation = TOTAL_ROT_DEG * (k / max(1, FRAMES_PER_OBJ - 1))

        hand_v = None; sph_c = sc1; sph_r = sr1
        hand_alpha = 0.0; sphere_alpha = 0.0
        phase_label = None; label_alpha = 0.0

        if k < F_A_END:       # Phase A: mesh fades in (alpha via object-only render)
            tA = ease_in_out(k / max(1, F_A_END))
            phase_label = 'Input: 3D object mesh'; label_alpha = tA
        elif k < F_B1_END:    # Phase B1: sphere 1 fades in
            tB = (k - F_A_END) / max(1, F_B1_END - F_A_END)
            sphere_alpha = ease_in_out(tB)
            phase_label = 'Tap 1: ' + lbl1; label_alpha = 1.0
        elif k < F_C1_END:    # Phase C1: hand 1 fades in; sphere fades OUT in sync
            tC = (k - F_B1_END) / max(1, F_C1_END - F_B1_END)
            hand_v = hv1
            hand_alpha = ease_in_out(tC)
            sphere_alpha = 1.0 - ease_in_out(tC)    # fully gone by the time the hand is fully there
            phase_label = 'Tap 1: ' + lbl1; label_alpha = 1.0
        elif k < F_D1_END:    # Phase D1: hold grasp 1 — sphere entirely removed
            hand_v = hv1
            hand_alpha = 1.0
            sphere_alpha = 0.0
            phase_label = 'Tap 1: ' + lbl1; label_alpha = 1.0
        elif k < F_X_END:     # Phase X: grasp 1 hand fades out (sphere already gone)
            tX = (k - F_D1_END) / max(1, F_X_END - F_D1_END)
            hand_v = hv1
            hand_alpha = 1.0 - ease_in_out(tX)
            sphere_alpha = 0.0
            phase_label = None; label_alpha = 0.0
        elif k < F_B2_END:    # Phase B2: sphere 2 fades in
            tB = (k - F_X_END) / max(1, F_B2_END - F_X_END)
            sph_c, sph_r = sc2, sr2
            sphere_alpha = ease_in_out(tB)
            phase_label = 'Tap 2: ' + lbl2; label_alpha = 1.0
        elif k < F_C2_END:    # Phase C2: hand 2 fades in; sphere fades OUT in sync
            tC = (k - F_B2_END) / max(1, F_C2_END - F_B2_END)
            sph_c, sph_r = sc2, sr2
            hand_v = hv2
            hand_alpha = ease_in_out(tC)
            sphere_alpha = 1.0 - ease_in_out(tC)
            phase_label = 'Tap 2: ' + lbl2; label_alpha = 1.0
        else:                 # Phase D2: hold grasp 2, sphere entirely removed
            sph_c, sph_r = sc2, sr2
            hand_v = hv2
            hand_alpha = 1.0
            sphere_alpha = 0.0
            phase_label = 'Tap 2: ' + lbl2; label_alpha = 1.0

        scene = render_scene(obj_v, obj_f, hand_v, sph_c, sph_r,
                              cam_ctr, cam_pos, rotation,
                              hand_alpha=hand_alpha, sphere_alpha=sphere_alpha)
        frame = compose_frame(scene, name, phase_label, label_alpha, is_unseen)
        cv2.imwrite(os.path.join(OUT_DIR, f'frame_{frame_idx:05d}.png'),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frame_idx += 1

print(f'rendered {frame_idx} frames into {OUT_DIR}')
print('encoding video via ffmpeg...')
cmd = [
    'ffmpeg', '-y', '-framerate', str(FPS),
    '-i', os.path.join(OUT_DIR, 'frame_%05d.png'),
    '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
    '-crf', '18', '-preset', 'slow',
    '-movflags', '+faststart',
    VIDEO_OUT,
]
subprocess.run(cmd, check=True)
print(f'saved: {VIDEO_OUT}')
