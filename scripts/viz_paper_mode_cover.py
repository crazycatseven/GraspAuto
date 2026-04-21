#!/usr/bin/env python3
"""Controllability figure for the main paper:
  6 different tap locations on a single mug → 6 different generated grasps.

Each panel shows the mug + a translucent blue grip-sphere at the user's tap
location + the model's grasp for that sphere condition. The grasp is the
best-of-64 selector-picked sample from r047's eval_bo64sel pool.

Visual message: the sphere condition gives the user fine-grained control over
WHERE to grasp; the model produces a plausible hand pose for each.
"""
from pathlib import Path
import os, sys, json
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
from graspauto.velocity_network import VelocityNetwork
from graspauto.conditioning import ContactGraphConditioningAdapter, compute_patch_weight_from_point
from graspauto.flow_matching import ConditionalFlowMatching
from graspauto.mano_autoencoder import ResidualMANOAutoEncoder

EVAL = 'outputs/graspauto_sphere_r047/eval_bo64sel/per_sample.json'
VAL = 'outputs/stage3_contact_graph/val_sphere.pt'
OBJ_CACHE = 'outputs/stage3_contact_graph/object_m2ae_cache.pt'
R047_CKPT = 'outputs/graspauto_sphere_r047/best.pt'
AE_CKPT = 'outputs/graspauto_ae_joint/best.pt'
MESH_DIR = 'data/contactpose/data_raw/contactpose_ply_files_mm'
OUT = 'paper/figures/mode_cover.png'

OBJECTS = ['mug', 'hammer']
TAPS_PER_OBJECT = 3
# Use real val samples so the grasps are exactly the selector-picked grasps
# from r047's bo-64 pool (high quality, in-distribution). For cases where the
# centroid falls INSIDE the mug body (handle grips), we project the DISPLAY
# marker in the inset to the nearest mug surface point so it reads as a wall
# tap rather than floating in the cavity.
TAP_INDICES = {
    'mug':    [47, 68, 91],      # handle side, body wall, top rim
    'hammer': [135, 43, 86],     # head, handle middle, handle tip
}
# Which mug panels should have their inset sphere projected onto the mug
# outer surface (because the real centroid lies inside the body cavity).
MUG_PROJECT_INSET_IDXS = {47}

CANONICAL_VIEW = {
    'mug':    (45.0, 55.0),
    'hammer': (90.0, 40.0),
}
MAIN_VIEW = {
    'mug':    'auto',
    'hammer': (90.0, 22.0),
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Resolve panel specs (object, tap idx, sphere, picked grasp) --------------
recs = json.load(open(EVAL))['records']
val = torch.load(VAL, map_location='cpu', weights_only=False)
obj_names = list(val['object_name'])

def make_panel(idx):
    rec = next(r for r in recs if r['sample_index'] == idx)
    c = val['unified_centroid'][idx].numpy()
    radius = float(val['unified_spread'][idx].norm())
    hTm_rot = val['hTm_rot'][idx].numpy()
    approach = hTm_rot @ np.array([0.0, 0.0, 1.0])
    return (idx, rec['picked_x1'], c, radius, approach, rec['mode_cover_mm'])

object_panels = {obj: [make_panel(i) for i in TAP_INDICES[obj]] for obj in OBJECTS}

print('selected panels:')
for obj, ps in object_panels.items():
    for p in ps:
        print(f'  {obj:6s}  idx={p[0]:3d}  centre=[{p[2][0]:+.3f},{p[2][1]:+.3f},{p[2][2]:+.3f}]'
              f'  r={p[3]*1000:4.1f}mm  mc={p[5]:.1f}mm')

# --- Decode the selector-picked grasps ---------------------------------------
decoder = MangoMANODecoder().to(device).eval()
hand_faces = decoder.faces.cpu().numpy()

def decode_mano(x1_54):
    x = torch.tensor(x1_54).unsqueeze(0).float().to(device)
    with torch.no_grad():
        out = decoder(x)
    return out['vertices'][0].cpu().numpy()

object_hands = {obj: [decode_mano(p[1]) for p in object_panels[obj]] for obj in OBJECTS}

# --- Load mesh per object (scaled to metres, simplified if large) -------------
def load_mesh(name):
    m = trimesh.load(os.path.join(MESH_DIR, f'{name}.ply'), force='mesh')
    v = np.asarray(m.vertices) * 0.001
    f = np.asarray(m.faces)
    if len(f) > 50000:
        m_simpl = m.simplify_quadric_decimation(face_count=15000)
        v = np.asarray(m_simpl.vertices) * 0.001
        f = np.asarray(m_simpl.faces)
    return v, f

object_meshes = {obj: load_mesh(obj) for obj in OBJECTS}

renderer = o3d.visualization.rendering.OffscreenRenderer(620, 620)
renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

# Smaller renderer for the inset mini-map (canonical mug view).
inset_renderer = o3d.visualization.rendering.OffscreenRenderer(260, 260)
inset_renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])


def rotation_from_z_to(n):
    n = n / (np.linalg.norm(n) + 1e-9)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z, n); s = np.linalg.norm(axis); c = float(np.dot(z, n))
    if s < 1e-6:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    axis = axis / s
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + s * K + (1 - c) * K @ K


def snap_white(img):
    r, g, b = img[..., 0].astype(int), img[..., 1].astype(int), img[..., 2].astype(int)
    brightness = (r + g + b) / 3
    spread = np.max([np.abs(r - g), np.abs(g - b), np.abs(r - b)], axis=0)
    mask = (brightness >= 220) & (spread <= 5)
    img[mask] = 255
    return img


SPHERE_COLOR = [0.98, 0.82, 0.10]   # bright warm yellow


def render_panel(obj_v, obj_f, hand_verts, azim_deg=25.0, elev_deg=20.0):
    """Main panel: object + generated hand, no sphere. Camera distance is
    auto-scaled so both fit the frame regardless of object size."""
    renderer.scene.clear_geometry()

    obj_mesh = o3d.geometry.TriangleMesh()
    obj_mesh.vertices = o3d.utility.Vector3dVector(obj_v)
    obj_mesh.triangles = o3d.utility.Vector3iVector(obj_f)
    obj_mesh.compute_vertex_normals()
    obj_mesh.paint_uniform_color([0.78, 0.78, 0.82])
    mat_o = o3d.visualization.rendering.MaterialRecord(); mat_o.shader = 'defaultLit'
    renderer.scene.add_geometry('obj', obj_mesh, mat_o)

    hand_mesh = o3d.geometry.TriangleMesh()
    hand_mesh.vertices = o3d.utility.Vector3dVector(hand_verts)
    hand_mesh.triangles = o3d.utility.Vector3iVector(hand_faces)
    hand_mesh.compute_vertex_normals()
    hand_mesh.paint_uniform_color([0.95, 0.55, 0.55])
    mat_h = o3d.visualization.rendering.MaterialRecord(); mat_h.shader = 'defaultLit'
    renderer.scene.add_geometry('hand', hand_mesh, mat_h)

    # Auto-distance: use the combined bbox diagonal to keep framing consistent.
    all_p = np.vstack([obj_v, hand_verts])
    bbox_diag = np.linalg.norm(all_p.max(0) - all_p.min(0))
    dist_m = max(0.22, bbox_diag * 1.25)   # floor for very small objects
    ctr = all_p.mean(0)
    el = max(15.0, float(elev_deg))        # clamp elevation so camera isn't too low
    az_r, el_r = np.deg2rad(azim_deg), np.deg2rad(el)
    direction = np.array([np.cos(el_r) * np.cos(az_r), np.cos(el_r) * np.sin(az_r), np.sin(el_r)])
    cam_pos = ctr + direction * dist_m
    renderer.setup_camera(45.0, ctr.tolist(), cam_pos.tolist(), [0, 0, 1])
    return snap_white(np.asarray(renderer.render_to_image()).copy())


SPHERE_VISUAL_SCALE = 0.40  # fraction of actual sphere radius to render as marker


def _camera_for_inset(obj_v, azim_deg, elev_deg):
    bbox_diag = np.linalg.norm(obj_v.max(0) - obj_v.min(0))
    dist_m = max(0.16, bbox_diag * 1.1)
    ctr = obj_v.mean(0)
    az, el = np.deg2rad(azim_deg), np.deg2rad(elev_deg)
    direction = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return ctr, ctr + direction * dist_m


def render_inset(obj_v, obj_f, sph_center, sph_radius,
                  azim_deg=45.0, elev_deg=55.0):
    """Inset mini-map: semi-transparent object + a flat-shaded yellow sphere
    drawn ALWAYS on top (two-pass render + 2D composite) so it is never
    occluded. The sphere is a small marker (SPHERE_VISUAL_SCALE * actual
    radius) so it reads as a tap location, not a large ball eating the
    object."""
    ctr, cam_pos = _camera_for_inset(obj_v, azim_deg, elev_deg)

    # Pass A: object only.
    inset_renderer.scene.clear_geometry()
    obj_mesh = o3d.geometry.TriangleMesh()
    obj_mesh.vertices = o3d.utility.Vector3dVector(obj_v)
    obj_mesh.triangles = o3d.utility.Vector3iVector(obj_f)
    obj_mesh.compute_vertex_normals()
    obj_mesh.paint_uniform_color([0.80, 0.80, 0.84])
    mat_o = o3d.visualization.rendering.MaterialRecord(); mat_o.shader = 'defaultLit'
    inset_renderer.scene.add_geometry('obj', obj_mesh, mat_o)
    inset_renderer.setup_camera(45.0, ctr.tolist(), cam_pos.tolist(), [0, 0, 1])
    img_A = snap_white(np.asarray(inset_renderer.render_to_image()).copy())

    # Pass B: sphere only (flat unlit yellow).
    inset_renderer.scene.clear_geometry()
    sphere = o3d.geometry.TriangleMesh.create_sphere(
        radius=float(sph_radius) * SPHERE_VISUAL_SCALE, resolution=28)
    sphere.translate(list(sph_center))
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(SPHERE_COLOR)
    mat_s = o3d.visualization.rendering.MaterialRecord(); mat_s.shader = 'defaultUnlit'
    inset_renderer.scene.add_geometry('sphere', sphere, mat_s)
    inset_renderer.setup_camera(45.0, ctr.tolist(), cam_pos.tolist(), [0, 0, 1])
    img_B = np.asarray(inset_renderer.render_to_image()).copy()

    # Composite: pass B's non-white pixels overwrite pass A.
    r, g, b = img_B[..., 0].astype(int), img_B[..., 1].astype(int), img_B[..., 2].astype(int)
    brightness = (r + g + b) / 3
    spread = np.max([np.abs(r - g), np.abs(g - b), np.abs(r - b)], axis=0)
    sphere_mask = ~((brightness >= 220) & (spread <= 5))
    out = img_A.copy()
    out[sphere_mask] = img_B[sphere_mask]
    # Draw a thin dark outline around the sphere silhouette for extra punch.
    try:
        from scipy.ndimage import binary_dilation
        outline = binary_dilation(sphere_mask, iterations=1) & ~sphere_mask
        out[outline] = [80, 55, 10]
    except Exception:
        pass
    return out


def best_azim_for_panel(hand_verts, obj_v):
    """Pick camera azimuth so the hand is on the camera-near side of the object."""
    offset = hand_verts.mean(0) - obj_v.mean(0)
    return float(np.degrees(np.arctan2(offset[1], offset[0])))


panel_grid = []  # 2D list indexed [row][col] of (main_img, inset_img, panel_spec, obj_name)
for obj in OBJECTS:
    obj_v, obj_f = object_meshes[obj]
    az_inset, el_inset = CANONICAL_VIEW[obj]
    row_cells = []
    main_view = MAIN_VIEW.get(obj, 'auto')
    for panel, hv in zip(object_panels[obj], object_hands[obj]):
        if main_view == 'auto':
            az_main, el_main = best_azim_for_panel(hv, obj_v), 20.0
        else:
            az_main, el_main = main_view
        main_img = render_panel(obj_v, obj_f, hv,
                                 azim_deg=az_main, elev_deg=el_main)

        # For taps whose real centroid lies inside the mug cavity, push the
        # display marker radially outward to the outer cylindrical wall so
        # the inset clearly reads as "tap the wall here".
        disp_center = panel[2]
        disp_radius = panel[3]
        if obj == 'mug' and panel[0] in MUG_PROJECT_INSET_IDXS:
            xy = np.array(panel[2][:2])
            xy_norm = np.linalg.norm(xy)
            if xy_norm > 1e-4:
                OUTER_R = 0.042
                xy_wall = xy * (OUTER_R / xy_norm)
                disp_center = np.array([xy_wall[0], xy_wall[1], panel[2][2]])
            disp_radius = 0.028   # visual shrink for projected markers

        # For mug: per-panel inset view so each tap's marker sits on the near
        # side (so the viewer sees it clearly, not through the mug). For
        # hammer: keep the fixed canonical side-view so the three inset views
        # align (hammer is long and narrow — a single side view works for all).
        if obj == 'mug':
            tap_xy = np.array(disp_center[:2])
            if np.linalg.norm(tap_xy) < 0.015 and disp_center[2] > 0.070:
                inset_az_eff, inset_el_eff = 45.0, 55.0   # top-rim special case
            else:
                inset_az_eff = float(np.degrees(np.arctan2(tap_xy[1], tap_xy[0])))
                inset_el_eff = 25.0
        else:
            inset_az_eff, inset_el_eff = az_inset, el_inset
        inset_img = render_inset(obj_v, obj_f, disp_center, disp_radius,
                                  azim_deg=inset_az_eff, elev_deg=inset_el_eff)
        row_cells.append((main_img, inset_img, panel, obj))
    panel_grid.append(row_cells)

# --- Matplotlib layout: 2 rows (mug, mouse) × 3 cols (3 tap locations) -------
fig = plt.figure(figsize=(13.5, 7.8), facecolor='white')
gs = fig.add_gridspec(len(OBJECTS), TAPS_PER_OBJECT, wspace=0.005, hspace=0.05)

for row_i, row_cells in enumerate(panel_grid):
    for col_i, (main, inset, (idx, _x1, c, r, n, mc), obj) in enumerate(row_cells):
        ax = fig.add_subplot(gs[row_i, col_i])
        ax.imshow(main, aspect='equal')
        ax.axis('off')
        for s in ax.spines.values():
            s.set_visible(False)

        ax_in = ax.inset_axes([0.65, 0.62, 0.35, 0.35])
        ax_in.imshow(inset, aspect='equal')
        ax_in.set_xticks([]); ax_in.set_yticks([])
        for s in ax_in.spines.values():
            s.set_edgecolor('#4a6e9a'); s.set_linewidth(1.2)

        label = f'{obj}  \u00B7  tap {col_i + 1}  \u00B7  r = {r * 1000:.0f}\u2009mm'
        ax.text(0.03, 0.05, label,
                 transform=ax.transAxes, fontsize=10.5, fontweight='bold',
                 color='#2a5d8e', verticalalignment='bottom',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           edgecolor='#bbb', alpha=0.9))

plt.savefig(OUT, dpi=130, bbox_inches='tight', facecolor='white', pad_inches=0.04)
print(f'saved: {OUT}')
