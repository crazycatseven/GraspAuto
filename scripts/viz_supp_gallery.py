#!/usr/bin/env python3
"""Regenerate the supplementary per-object gallery using the teaser-quality
rendering pipeline (mesh object + hand mesh + white bg + consistent camera).

Output: paper/figures/gallery_full_25.png
"""
from pathlib import Path
import os, json, sys
# Resolve project root from this file's location (works from any cwd)
REPO = Path(__file__).resolve().parent.parent
os.chdir(str(REPO))
sys.path.insert(0, 'src')

import torch
import numpy as np
import open3d as o3d
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from graspauto.mano_decoder import MangoMANODecoder

EVAL = 'outputs/graspauto_sphere_r047/eval_bo1/per_sample.json'
VAL = 'outputs/stage3_contact_graph/val_sphere.pt'
MESH_DIR = 'data/contactpose/data_raw/contactpose_ply_files_mm'
OUT = 'paper/figures/gallery_full_25.png'

HOLDOUT = {'bowl', 'headphones', 'toothbrush', 'wine_glass'}

recs = json.load(open(EVAL))['records']
val = torch.load(VAL, map_location='cpu', weights_only=False)
obj_names = list(val['object_name'])

from collections import defaultdict
by_obj = defaultdict(list)
for r in recs:
    by_obj[obj_names[r['sample_index']]].append(r)

# Best per object (lowest mc); only keep objects we have both eval data AND mesh for
all_objs = sorted(by_obj.keys())
best_per_obj = {}
for name in all_objs:
    mesh_path = os.path.join(MESH_DIR, f'{name}.ply')
    if not os.path.exists(mesh_path):
        continue
    best_per_obj[name] = min(by_obj[name], key=lambda r: r['mode_cover_mm'])

objects = sorted(best_per_obj.keys())
print(f'rendering {len(objects)} objects: {objects}')

decoder = MangoMANODecoder()
hand_faces = decoder.faces.numpy()

def decode_mano(x1_54):
    x = torch.tensor(x1_54).unsqueeze(0).float()
    with torch.no_grad():
        out = decoder(x)
    return out['vertices'][0].numpy()


def load_obj_mesh(name):
    path = os.path.join(MESH_DIR, f'{name}.ply')
    m = trimesh.load(path, force='mesh')
    verts_m = np.asarray(m.vertices) * 0.001
    faces = np.asarray(m.faces)
    if len(faces) > 50000:
        target = 15000
        m_simpl = m.simplify_quadric_decimation(face_count=target)
        verts_m = np.asarray(m_simpl.vertices) * 0.001
        faces = np.asarray(m_simpl.faces)
    return verts_m, faces


# Per-object camera overrides (same convention as teaser v3)
CAMERA_OVERRIDES = {
    'headphones': dict(azim_deg=140.0, elev_deg=35.0),
    'toothbrush': dict(azim_deg=-60.0, elev_deg=40.0),
    'banana':     dict(azim_deg=-30.0, elev_deg=45.0),
    'eyeglasses': dict(azim_deg=-50.0, elev_deg=45.0),
    'flashlight': dict(azim_deg=-30.0, elev_deg=30.0),
    'knife':      dict(azim_deg=20.0,  elev_deg=35.0),
}


renderer = o3d.visualization.rendering.OffscreenRenderer(520, 520)
renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])


def render_panel(obj_verts, obj_faces, hand_verts, name=None):
    renderer.scene.clear_geometry()

    obj_mesh = o3d.geometry.TriangleMesh()
    obj_mesh.vertices = o3d.utility.Vector3dVector(obj_verts)
    obj_mesh.triangles = o3d.utility.Vector3iVector(obj_faces)
    obj_mesh.compute_vertex_normals()
    obj_mesh.paint_uniform_color([0.78, 0.78, 0.82])
    mat_obj = o3d.visualization.rendering.MaterialRecord(); mat_obj.shader = 'defaultLit'
    renderer.scene.add_geometry('obj', obj_mesh, mat_obj)

    hand_mesh = o3d.geometry.TriangleMesh()
    hand_mesh.vertices = o3d.utility.Vector3dVector(hand_verts)
    hand_mesh.triangles = o3d.utility.Vector3iVector(hand_faces)
    hand_mesh.compute_vertex_normals()
    hand_mesh.paint_uniform_color([0.95, 0.55, 0.55])
    mat_h = o3d.visualization.rendering.MaterialRecord(); mat_h.shader = 'defaultLit'
    renderer.scene.add_geometry('hand', hand_mesh, mat_h)

    all_p = np.vstack([obj_verts, hand_verts])
    ctr = all_p.mean(0)
    CAM_DIST_M = 0.35
    azim_deg = 30.0; elev_deg = 25.0
    if name in CAMERA_OVERRIDES:
        ov = CAMERA_OVERRIDES[name]
        azim_deg = ov['azim_deg']; elev_deg = ov['elev_deg']
    az, el = np.deg2rad(azim_deg), np.deg2rad(elev_deg)
    direction = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    cam_pos = ctr + direction * CAM_DIST_M
    renderer.setup_camera(45.0, ctr.tolist(), cam_pos.tolist(), [0, 0, 1])
    img = np.asarray(renderer.render_to_image()).copy()
    r, g, b = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
    brightness = (r + g + b) / 3
    rgb_spread = np.max([np.abs(r-g), np.abs(g-b), np.abs(r-b)], axis=0)
    mask = (brightness >= 220) & (rgb_spread <= 5)
    img[mask] = 255
    return img


# Layout: 5 cols × ceil(N/5) rows
ncols = 5
nrows = (len(objects) + ncols - 1) // ncols
fig = plt.figure(figsize=(ncols * 3.3, nrows * 3.3), facecolor='white')
gs = fig.add_gridspec(nrows, ncols, wspace=0.02, hspace=0.12)

for i, name in enumerate(objects):
    row, col = i // ncols, i % ncols
    ax = fig.add_subplot(gs[row, col])
    r = best_per_obj[name]
    mc = r['mode_cover_mm']
    pred_verts = decode_mano(r['picked_x1'])
    obj_v, obj_f = load_obj_mesh(name)
    img = render_panel(obj_v, obj_f, pred_verts, name=name)
    ax.imshow(img)
    ax.set_facecolor('white')
    ax.axis('off')
    for s in ax.spines.values():
        s.set_visible(False)
    is_unseen = name in HOLDOUT
    color = '#8e4c2a' if is_unseen else '#2a5d8e'
    label_text = f'{name}\n{mc:.1f}\u2009mm'
    if is_unseen:
        label_text += '  (UNSEEN)'
    ax.text(0.03, 0.05, label_text,
             transform=ax.transAxes,
             fontsize=11, fontweight='bold',
             color=color,
             verticalalignment='bottom', horizontalalignment='left',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='#bbb', alpha=0.9))

plt.savefig(OUT, dpi=120, bbox_inches='tight', facecolor='white', pad_inches=0.1)
print(f'saved: {OUT}')
