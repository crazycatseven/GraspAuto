# Data setup

The repository does **not** bundle datasets, trained checkpoints, or the MANO model (license-restricted). Follow the steps below to reproduce the paper numbers.

All paths below are relative to the repository root.

---

## 0. Point-M2AE pretrained weights (required only to regenerate the object feature cache)

If you want to re-extract object features from scratch (e.g., after adding a new dataset or changing meshes) you need the Point-M2AE pretraining checkpoint at `external/pretrained/point_m2ae_pretrain.pth` (183 MB). Download from <https://github.com/ZrrSkywalker/Point-M2AE>.

For normal training and evaluation you can skip this — the pipeline reads a pre-computed `object_m2ae_cache.pt` (see §2) that we'll release with the other checkpoints.

---

## 1. MANO model weights (required for anything that decodes poses)

`manotorch.ManoLayer` expects the official MANO pickles in `assets/mano_v1_2/`:

```
assets/mano_v1_2/
└── models/
    ├── MANO_RIGHT.pkl
    └── MANO_LEFT.pkl
```

1. Register at <https://mano.is.tue.mpg.de/> (free, academic).
2. Download **MANO v1.2** and extract so the two `.pkl` files land at the paths above.

Without these files any script that instantiates `MangoMANODecoder` (training, eval, every viz script) will abort at start-up with a clear message.

---

## 2. ContactPose (primary training/eval data)

ContactPose provides hand-object grasps with thermal contact maps across 25 object classes.

1. Obtain the raw dataset from <https://contactpose.cc.gatech.edu/> and place the PLY meshes at:

   ```
   data/contactpose/data_raw/contactpose_ply_files_mm/<object>.ply
   ```

2. Preprocess into the intermediate tensors GraspAuto expects:

   ```bash
   # Stage-3 contact-graph cache (train + val samples with MANO poses)
   python src/preprocess_grip_sphere.py \
       --source contactpose \
       --out outputs/stage3_contact_graph
   ```

   This writes:

   ```
   outputs/stage3_contact_graph/
   ├── train.pt         # pose + object-id + participant
   ├── val.pt
   ├── train_sphere.pt  # adds unified_centroid / unified_normal / unified_spread / palm_entropy
   ├── val_sphere.pt
   └── object_m2ae_cache.pt       # per-object 64 patch tokens (Point-M2AE run once)
   ```

   The val split follows standard ContactPose practice (participants disjoint from train). Our reporting further splits val by object class:
   - **TRUE SEEN** = 21 classes (N=179 samples)
   - **CP-UNSEEN** = 4 classes (bowl, headphones, toothbrush, wine_glass; 37 samples)
     — these classes appear in the CP training set from different participants; the split is a per-class reporting convention, not a true class-level holdout.

---

## 3. OakInk-Shape (sphere-era mixing, optional)

OakInk provides a real-MoCap hand-object subset used by the sphere-conditioned checkpoints.

1. Download OakInk-Shape from <https://oakink.net/> (academic agreement).
2. Place the shape dataset under `data/oakink/`.
3. Run the same preprocessing with `--source oakink` to produce a second `train_sphere.pt` segment, then point the trainer at the combined data via its `--data-mix` flag.

OakInk is optional: training with ContactPose alone (`sphere, CP-only` row in Main Table 1) reproduces 12.36 mm TRUE SEEN without it. The shipped 10.64 mm checkpoint uses `CP + OakInk contact-filtered`.

---

## 4. GraspXL (truly-novel-geometry eval, optional)

Only needed to reproduce the 56.7 mm number in Supplementary S6 (true cross-class generalisation).

1. Download from <https://github.com/zdchan/graspxl>.
2. Run `eval/eval_on_graspxl_unseen.py` (script not yet included in this trim; lifted from the upstream research repo). 36 objects, 150 samples.

---

## 5. Pretrained checkpoints

Checkpoints are not yet attached to this repo (28.9 MB Mode A bundle, 90.3 MB Mode B including the Point-M2AE encoder; see the paper for details). They will be added as a GitHub Release together with the Unity Sentis sample.

Expected layout used by the scripts:

```
outputs/
├── graspauto_ae_joint/best.pt                # MANO residual autoencoder (1.08 M)
├── graspauto_sphere_r042/best.pt             # 8-member sphere pool
├── graspauto_sphere_r043/best.pt
├── graspauto_sphere_r044/best.pt
├── graspauto_sphere_r045/best.pt
├── graspauto_sphere_r046/best.pt
└── graspauto_sphere_r047/                    # shipped single-checkpoint
    ├── best.pt
    ├── eval_bo1/per_sample.json
    └── eval_bo64sel/per_sample.json
```

(The full pool also adds two members — real-mocap-only and min-radius ablation — following Supp. Table S2.)

---

## 6. Reproducing each paper figure

Once the above is in place:

| Figure                                | Script                                             |
| ------------------------------------- | -------------------------------------------------- |
| **Fig. 3** (controllability: mug + hammer) | `python scripts/viz_paper_mode_cover.py`      |
| **Supp. S8 gallery** (25 objects)     | `python scripts/viz_supp_gallery.py`               |
| **Supp. S9 failures** (top-3)         | `python scripts/viz_supp_failures.py`              |
| **Teaser video** (60 s, 1920×1080)    | `python scripts/make_teaser_video.py`              |

Fig. 1 (teaser) and Fig. 2 (pipeline diagram) were hand-assembled in Figma; their source art lives in `paper/figures/`.

Each script expects the outputs listed above and aborts early if a file is missing.

---

## What's in / out of the repo

In (tracked):
- `src/graspauto/`, `src/graspauto/` — model & data-graph code
- `train/`, `eval/`, `scripts/` — entry points
- `paper/` — PDFs, figures, teaser video
- `docs/` — GitHub Pages landing page
- `requirements.txt`, `README.md`, `LICENSE`, `DATA.md`

Out (git-ignored — too large or license-restricted):
- `outputs/` — intermediate tensors and checkpoints (Section 2–5 above)
- `data/` — raw datasets (Sections 2–4)
- `assets/mano_v1_2/` — MANO weights (Section 1)
