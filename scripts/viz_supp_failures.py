#!/usr/bin/env python3
"""Render the worst-case grasp generations for the supplementary failure gallery.

We sort val samples by mode-cover error from r047's bo-1 CFG 1.5 eval and
render the top-N highest-error samples as (object + generated hand + GT
hand) triplets using the teaser rendering pipeline.
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
OUT = 'paper/figures/failure_cases.png'

N_FAILURES = 3  # top-N worst (kept tight for single-column layout)
HOLDOUT = {'bowl', 'headphones', 'toothbrush', 'wine_glass'}

recs = json.load(open(EVAL))['records']
val = torch.load(VAL, map_location='cpu', weights_only=False)
obj_names = list(val['object_name'])

# Rank by descending mode_cover, take top N
recs_sorted = sorted(recs, key=lambda r: -r['mode_cover_mm'])
failures = []
for r in recs_sorted:
    name = obj_names[r['sample_index']]
    mesh_path = os.path.join(MESH_DIR, f'{name}.ply')
    if not os.path.exists(mesh_path):
        continue
    failures.append((r, name))
    if len(failures) >= N_FAILURES:
        break

print(f'top-{len(failures)} failures:')
for r, name in failures:
    print(f'  {name} ({"UNSEEN" if name in HOLDOUT else "SEEN"}): mc={r["mode_cover_mm"]:.1f}mm')

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
        m_simpl = m.simplify_quadric_decimation(face_count=15000)
        verts_m = np.asarray(m_simpl.vertices) * 0.001
        faces = np.asarray(m_simpl.faces)
    return verts_m, faces

CAMERA_OVERRIDES = {
    'headphones': dict(azim_deg=140.0, elev_deg=35.0),
    'toothbrush': dict(azim_deg=-60.0,  elev_deg=40.0),
    'banana':     dict(azim_deg=-30.0,  elev_deg=45.0),
    'flashlight': dict(azim_deg=-30.0,  elev_deg=30.0),
}

renderer = o3d.visualization.rendering.OffscreenRenderer(520, 520)
renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])


def render_panel(obj_verts, obj_faces, hand_verts, name=None, hand_color=(0.95, 0.55, 0.55)):
    renderer.scene.clear_geometry()
    obj_mesh = o3d.geometry.TriangleMesh()
    obj_mesh.vertices = o3d.utility.Vector3dVector(obj_verts)
    obj_mesh.triangles = o3d.utility.Vector3iVector(obj_faces)
    obj_mesh.compute_vertex_normals()
    obj_mesh.paint_uniform_color([0.78, 0.78, 0.82])
    mat_o = o3d.visualization.rendering.MaterialRecord(); mat_o.shader = 'defaultLit'
    renderer.scene.add_geometry('obj', obj_mesh, mat_o)

    hand_mesh = o3d.geometry.TriangleMesh()
    hand_mesh.vertices = o3d.utility.Vector3dVector(hand_verts)
    hand_mesh.triangles = o3d.utility.Vector3iVector(hand_faces)
    hand_mesh.compute_vertex_normals()
    hand_mesh.paint_uniform_color(list(hand_color))
    mat_h = o3d.visualization.rendering.MaterialRecord(); mat_h.shader = 'defaultLit'
    renderer.scene.add_geometry('hand', hand_mesh, mat_h)

    all_p = np.vstack([obj_verts, hand_verts])
    ctr = all_p.mean(0)
    CAM_DIST_M = 0.35
    azim_deg = 30.0; elev_deg = 25.0
    if name in CAMERA_OVERRIDES:
        azim_deg = CAMERA_OVERRIDES[name]['azim_deg']
        elev_deg = CAMERA_OVERRIDES[name]['elev_deg']
    az, el = np.deg2rad(azim_deg), np.deg2rad(elev_deg)
    direction = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    cam_pos = ctr + direction * CAM_DIST_M
    renderer.setup_camera(45.0, ctr.tolist(), cam_pos.tolist(), [0, 0, 1])
    img = np.asarray(renderer.render_to_image()).copy()
    r, g, b = img[...,0].astype(int), img[...,1].astype(int), img[...,2].astype(int)
    brightness = (r+g+b)/3
    spread = np.max([np.abs(r-g), np.abs(g-b), np.abs(r-b)], axis=0)
    mask = (brightness >= 220) & (spread <= 5)
    img[mask] = 255
    return img


# Layout: 2 rows × N_FAILURES cols (top = generated, bottom = GT reference)
fig = plt.figure(figsize=(3.3 * N_FAILURES, 6.2), facecolor='white')
gs = fig.add_gridspec(2, N_FAILURES, wspace=0.02, hspace=0.05)

for i, (r, name) in enumerate(failures):
    idx = r['sample_index']
    pred_verts = decode_mano(r['picked_x1'])
    gt_verts = val['gt_world_verts'][idx].numpy()
    obj_v, obj_f = load_obj_mesh(name)

    is_unseen = name in HOLDOUT
    tag = 'UNSEEN' if is_unseen else 'SEEN'
    label_color = '#8e4c2a' if is_unseen else '#2a5d8e'

    # Top row: generated (salmon)
    ax_g = fig.add_subplot(gs[0, i])
    img_g = render_panel(obj_v, obj_f, pred_verts, name=name, hand_color=(0.95, 0.55, 0.55))
    ax_g.imshow(img_g); ax_g.axis('off')
    for s in ax_g.spines.values(): s.set_visible(False)
    ax_g.text(0.03, 0.05,
               f'{name} ({tag})\ngenerated · {r["mode_cover_mm"]:.1f}\u2009mm',
               transform=ax_g.transAxes, fontsize=10, fontweight='bold',
               color=label_color, verticalalignment='bottom',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='#bbb', alpha=0.9))

    # Bottom row: GT reference (green)
    ax_r = fig.add_subplot(gs[1, i])
    img_r = render_panel(obj_v, obj_f, gt_verts, name=name, hand_color=(0.65, 0.82, 0.65))
    ax_r.imshow(img_r); ax_r.axis('off')
    for s in ax_r.spines.values(): s.set_visible(False)
    ax_r.text(0.03, 0.05, f'{name} ({tag})\nGT reference',
               transform=ax_r.transAxes, fontsize=10, fontweight='bold',
               color='#3a6e3a', verticalalignment='bottom',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor='#bbb', alpha=0.9))

plt.savefig(OUT, dpi=120, bbox_inches='tight', facecolor='white', pad_inches=0.1)
print(f'saved: {OUT}')
