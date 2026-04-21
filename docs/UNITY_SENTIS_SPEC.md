# Unity Sentis Integration Spec

Everything the Quest 3 / Unity Sentis port needs to know about the ONNX files, tensor shapes, pre/post-processing, and how to sanity-check numerics against the PyTorch reference.

Keep this file in sync with `scripts/validate_onnx.py` and `eval/eval.py`.

---

## 1. Shipped ONNX files

Mode A (ship cached object features, recommended for Quest 3 real-time):

| File                 | fp32 size | fp16 size | Role                                           |
| -------------------- | --------- | --------- | ---------------------------------------------- |
| `adapter.onnx`       | 0.48 MB   | 0.24 MB   | 7-D grip-sphere + 64 object patches → 65-token conditioning bundle |
| `velocity_net.onnx`  | 24.08 MB  | 12.04 MB  | CFM velocity net (one step)                    |
| `ae_decoder.onnx`    | 4.32 MB   | 2.16 MB   | 32-D latent → 54-D MANO params                 |
| **total**            | **28.88 MB** | **~14.4 MB** |                                           |

Mode B adds `point_m2ae.onnx` (61.4 MB fp32 / ~30 MB fp16) for on-device feature extraction of user-imported meshes.

---

## 2. Per-model I/O (Sentis tensor names must match exactly)

### 2.1 `adapter.onnx`

Inputs:

| Name                 | Shape      | Dtype   | Meaning                                                                  |
| -------------------- | ---------- | ------- | ------------------------------------------------------------------------ |
| `m2ae_local`         | `[1,64,384]` | float32 | Per-object Point-M2AE patch features (cached from precompute in Mode A). |
| `patch_centers`      | `[1,64,3]` | float32 | Object-frame 3-D centres of the 64 patches (cached).                     |
| `tap_point`          | `[1,3]`    | float32 | Grip-sphere centre `c ∈ ℝ³` in object frame (metres).                    |
| `palm_normal`        | `[1,3]`    | float32 | Approach direction `n ∈ ℝ³` (unit-length, object frame).                 |
| `palm_spread`        | `[1,3]`    | float32 | Grip aperture broadcast to 3 dims: `[r, r, r]` (metres).                 |
| `palm_entropy`       | `[1]`      | float32 | Soft-contact entropy proxy. At inference this can be `0.0` (the model has learned to tolerate 0 here for sphere-era runs). |
| `palm_mass`          | `[1]`      | float32 | Contact mass proxy. Same: `0.0` is fine for inference.                   |

Output:

| Name             | Shape        | Dtype   | Meaning                                                 |
| ---------------- | ------------ | ------- | ------------------------------------------------------- |
| `bundle_tokens`  | `[1,65,256]` | float32 | 65 tokens (64 object patches after tap-centered gating + 1 grip-sphere token) in hidden dim 256. Feed straight to `velocity_net`. |

### 2.2 `velocity_net.onnx`

Inputs:

| Name            | Shape        | Dtype   | Meaning                                     |
| --------------- | ------------ | ------- | ------------------------------------------- |
| `xt`            | `[1,32]`     | float32 | Current latent state along the Euler path.  |
| `t`             | `[1]`        | float32 | Flow time in `[0, 1]`.                      |
| `bundle_tokens` | `[1,65,256]` | float32 | Output of `adapter.onnx` (or the null bundle — all zeros of the same shape — for CFG unconditional pass). |

Output:

| Name       | Shape    | Dtype   | Meaning                          |
| ---------- | -------- | ------- | -------------------------------- |
| `velocity` | `[1,32]` | float32 | Instantaneous latent velocity.   |

### 2.3 `ae_decoder.onnx`

Input:

| Name | Shape    | Dtype   | Meaning                             |
| ---- | -------- | ------- | ----------------------------------- |
| `z`  | `[1,32]` | float32 | Final latent `z_1` after CFM steps. |

Output:

| Name          | Shape    | Dtype   | Meaning                                           |
| ------------- | -------- | ------- | ------------------------------------------------- |
| `mano_params` | `[1,54]` | float32 | Normalised 54-D MANO vector. **Must be denormalised** (see §4.3). |

### 2.4 `point_m2ae.onnx` (Mode B on-device feature extraction)

Input:

| Name     | Shape        | Dtype   | Meaning                              |
| -------- | ------------ | ------- | ------------------------------------ |
| `obj_pc` | `[B,2048,3]` | float32 | Exactly 2048 points sampled from the object mesh (object frame, metres). The ONNX export bakes in a deterministic farthest-point-sampling that starts at index 0, so upstream Unity sampling must be reproducible: either (a) pass 2048 points already sampled offline, or (b) downsample the mesh with FPS starting from a fixed index on the C# side so the same mesh always yields the same 2048-point subset. Random starts will produce non-reproducible features and break consistency across sessions. |

Outputs (Sentis will resolve dynamic dims):

| Name             | Shape        | Dtype   | Meaning                         |
| ---------------- | ------------ | ------- | ------------------------------- |
| `m2ae_global`    | `[B,1024]`   | float32 | Global object embedding (not used by Mode A inference). |
| `m2ae_local`     | `[B,64,384]` | float32 | Per-patch features (feed into adapter). |
| `patch_centers`  | `[B,64,3]`   | float32 | Object-frame patch centres.    |

---

## 3. Reference tensors for Unity-side sanity checks

`scripts/validate_onnx.py` writes the following to `artifacts/onnx_validation/`:

```
adapter_io.npz       m2ae_local, patch_centers, tap_point, palm_normal,
                     palm_spread, palm_entropy, palm_mass, output
velocity_net_io.npz  xt, t, bundle_tokens, output
ae_decoder_io.npz    z, output
validation_summary.json  max/mean abs & rel diff per model
```

On the Unity side, load the same `.npz` (or convert to `.bytes`), feed the inputs through Sentis, and compare against the `output` array. Acceptable drift for PyTorch → ONNX → Sentis is roughly `max_abs ≤ 5e-4` per model (adapter tends to be tighter, point_m2ae looser).

---

## 4. Inference pipeline

### 4.1 Conditioning bundle (once per tap)

```
in:  grip-sphere (c, r, n), object_id
out: bundle_tokens [1,65,256]
```

1. Look up the cached `m2ae_local` and `patch_centers` for `object_id` (precomputed per mesh).
2. Pack grip-sphere into the four adapter inputs:
   - `tap_point = c`
   - `palm_normal = n / ||n||`
   - `palm_spread = [r, r, r]`
   - `palm_entropy = 0.0`, `palm_mass = 0.0`
3. Run `adapter.onnx`.

### 4.2 CFM sampling (10 Euler steps, classifier-free guidance)

```
in:  bundle_tokens; seed for x_0
out: z_1 [1,32]
```

Pseudocode:

```csharp
const int NUM_STEPS = 10;
const float CFG_SCALE = 1.5f;
float dt = 1f / NUM_STEPS;

Tensor zt = RandnTensor(1, 32, seed);                 // x_0 ~ N(0, I)
for (int step = 0; step < NUM_STEPS; step++) {
    float t = step * dt;
    Tensor vCond = velocityNet.Execute(zt, t, bundleTokens);
    Tensor vUncond = velocityNet.Execute(zt, t, zeroBundleTokens);   // CFG null
    Tensor v = vUncond + CFG_SCALE * (vCond - vUncond);
    zt = zt + v * dt;                                 // Euler step
}
// zt is now z_1
```

The "null bundle" is a tensor of the same shape as `bundle_tokens` but all zeros — compute it once per session.

### 4.3 Latent → MANO parameters → hand pose

```
in:  z_1 [1,32]
out: mano_verts [778,3], mano_joints [21,3]
```

1. `ae_decoder.onnx(z_1)` → `mano_params [1,54]`. Denormalisation (`* train_std + train_mean`) is already baked into the ONNX graph — the output is ready to use.
2. Parse `mano_params [1,54]`:
   - `global_rot_6d = mano_params[:, 0:6]`
   - `translation   = mano_params[:, 6:9]`
   - `pose_45       = mano_params[:, 9:54]`
4. Convert the 6-D rotation to a 3×3 matrix following Zhou et al. (2019):

   ```
   a1 = mano_params[:, 0:3]
   a2 = mano_params[:, 3:6]
   b1 = a1 / ||a1||
   b2 = (a2 - <b1,a2> b1) / ||a2 - <b1,a2> b1||
   b3 = cross(b1, b2)
   R = [b1 | b2 | b3]      // 3×3
   ```
5. Feed `pose_45` through the MANO layer (mean pose + PCA or axis-angle — see `graspauto.mano_decoder.MangoMANODecoder` for the reference implementation) to get 778 template vertices and 21 joints in MANO-local frame.
6. Apply `R` and `translation` to the vertices/joints to bring them into object frame.
7. Optional: run inverse kinematics to the 26-joint OpenXR / 24-bone Quest SDK skeleton for XR Hand output.

---

## 5. Latency measurement methodology

Measure each model's `.Execute()` latency separately and end-to-end, both with and without an initial warm-up pass (Sentis compiles shaders on first run — include warm-up).

Recommended protocol:

```
for model in [adapter, velocity_net, ae_decoder]:
    warm-up: 5 runs
    timed:   50 runs, discard top/bottom 10%, report mean + p50 + p95

end-to-end:
    warm-up: 3 taps
    timed:   20 taps, same stats
```

On-device breakdown to report in the paper:

```
adapter        : ~_ ms
velocity_net   : ~_ ms × 20 passes (10 cond + 10 uncond for CFG)
ae_decoder     : ~_ ms
post-processing: ~_ ms  (6-D → rotmat, MANO FK)
total / tap    : ~_ ms
```

Current paper claims 60 ms/tap; if the Quest 3 measurement comes out different we update the numbers and the abstract before the arXiv upload.

---

## 6. Gotchas worth knowing before writing C#

- **No contact-head path on device.** The Unity port only needs the four ONNX files above. Everything upstream (contact prediction, palm/finger graph) is a training-time thing; at inference the user's tap fully determines the sphere.
- **`bundle_tokens` is already post-gated.** The adapter applies top-`K_patch` tap-centered locality gating internally (zeros out patches far from the tap), so Sentis doesn't need to reimplement that gate.
- **CFG doubles velocity-net cost.** Each of the 10 Euler steps runs `velocity_net` twice (conditional + unconditional). Budget accordingly.
- **Seeding.** MANO grasp diversity comes from `x_0`. For a single "best-of-1" grasp the seed is arbitrary. For multiple candidates or the consensus ranker, draw `x_0` from different seeds and keep all `z_1` outputs.
- **fp16 quantisation.** Run `validate_onnx.py` on the fp16 build too before shipping — quantisation can add another 1–5× to the drift.
- **MANO layer.** `manotorch` is the PyTorch reference. On C# side the MANO forward pass has to be reimplemented (or the 778 template vertices baked and skinned to the 21 computed joint transforms). Either is fine; just make sure it matches the reference.

---

## 7. Validation checklist before declaring "it works"

- [ ] `scripts/validate_onnx.py` passes with `max_abs ≤ 1e-4` per model (tolerance in the script).
- [ ] Sentis-side forward pass of each ONNX on `artifacts/onnx_validation/*.npz` reproduces the reference output within `5e-4`.
- [ ] 10-step CFM sample in Sentis, with the same `x_0` seed, lands within `1e-3` L2 of the PyTorch sample on the same tap (the gap is dominated by the velocity-net drift times 10 steps).
- [ ] Decoded MANO vertices match the PyTorch ones to within `0.5 mm` on 778 verts (visually indistinguishable).
- [ ] Per-model and end-to-end latencies measured on device, reported in README / paper.
- [ ] A short clip (30 s) of the Unity app running the full pipeline, single-tap → grasp, on Quest 3.
