# SAM 3D Body — Measurement & Serverless Deployment

> This document describes **the work built on top of** Meta's [SAM 3D Body](https://github.com/facebookresearch/sam-3d-body).
> For the upstream model card, architecture, and license, see [`README.md`](./README.md).

## What this project does

Given a **single photo of a person** and their **height in centimeters**, this service returns their
**body measurements** (chest, waist, hips, torso/spinal length, arm length, etc.) as JSON.

It does this by:

1. Running Meta's **SAM 3D Body** model to recover a 3D human mesh from the image.
2. Rebuilding that mesh in a neutral **T-pose** from the recovered shape parameters.
3. **Scaling** the mesh so its height matches the user-supplied real-world height (this converts the
   model's arbitrary units into real centimeters).
4. **Slicing / tracing** the mesh geometry to measure circumferences and lengths like a virtual tape measure.

The whole thing is packaged to run as a **serverless GPU worker on RunPod**.

## Pipeline at a glance

```
image + height ──► SAM 3D Body inference ──► T-pose mesh ──► scale to height ──► measure ──► JSON
   (handler)          (sam_3d_body)          (measure.py)     (measure.py)     (measure.py)  (handler)
```

## The code I wrote (vs. upstream)

Everything under `sam_3d_body/` is **Meta's model code, vendored into this repo** (see the note on the
fork below). The files below are the custom work built around it:

| File | Purpose |
|------|---------|
| **`main.py`** | Core orchestration. `load_model()` loads the estimator once (warm start); `pipe()` runs the full image → measurements pipeline. |
| **`measure.py`** | The measurement engine. Extracts the T-pose mesh from shape params (`get_tpose_mesh_final`), scales it to the target height (`scale_model`), and computes measurements (`get_measurements` / `calculate_body_metrics`). Circumferences are measured by taking a horizontal cross-section of the mesh and summing the perimeter of the largest loop (`get_largest_slice_loop`); lengths are measured as polylines through joints. |
| **`joints_definitions.py`** | Configuration: `MHR_JOINT_MAP` maps body landmarks to MHR joint indices, and `MEASUREMENT_CONFIG` declaratively defines each measurement (type, landmarks, offsets, colors). Add a measurement here without touching the engine. |
| **`utils.py`** | `setup_sam_3d_body()` — loads the model from a **local checkpoint** (`./checkpoints/...`) instead of pulling from HuggingFace at runtime, plus optional detector/segmentor/FOV components. |
| **`handler.py`** | The **RunPod serverless entry point**. Decodes the incoming base64 image, runs `pipe()`, and maps the internal measurement names to the snake_case JSON output contract. Model is loaded once at cold start and reused across requests. |
| **`test_handler.py`** | Local test harness — encodes a local image to base64 and calls `handler()` directly so you can test without deploying. |
| **`Dockerfile`** | Multi-stage CUDA 12.1 image. Builder stage compiles deps into a venv; runtime stage is slimmer (no build tools), forces headless EGL rendering for PyRender, runs as a non-root user. |
| **`requirements.txt`** | Python dependencies (PyTorch stack, trimesh/geometry, `runpod`). |

## Measurement definitions

Measurements are configured declaratively in `joints_definitions.py`. Three types are supported:

- **`circumference`** — takes a horizontal slice of the mesh at a target height (derived from named
  joints + an offset) and measures the perimeter of the largest resulting loop. Used for chest, waist, hips.
- **`polyline`** — sums the distances along a chain of joints, so it can follow a bend (e.g. arm length
  through shoulder → elbow → wrist, or spinal length down the back).
- **`linear`** — a straight point-to-point distance between two joints.

To add or tune a measurement, edit `MEASUREMENT_CONFIG` — no changes to `measure.py` needed.

## API contract

**Input** (RunPod job):

```json
{
  "input": {
    "image": "<base64 string or data URI>",
    "height": 163
  }
}
```

**Output**:

```json
{
  "chest_circumference": 92.5,
  "waist_circumference": 74.0,
  "hips_circumference": 98.1,
  "torso_length": 51.2,
  "arms_length": 58.7,
  "inside_leg_length": 78.3,
  "height": 163
}
```

`height` defaults to `165` if not provided. On failure the handler returns `{"error": "..."}`.

## Running it

**Locally (single image test):**

```bash
python test_handler.py     # edit img_path / height inside the file first
```

**As a serverless worker:**

```bash
python handler.py          # starts runpod.serverless worker
```

**Prerequisite — model checkpoints** must be present locally (not fetched at runtime):

```bash
hf download facebook/sam-3d-body-dinov3 --local-dir checkpoints/sam-3d-body-dinov3
```

`utils.py` expects:
- `./checkpoints/sam-3d-body-dinov3/model.ckpt`
- `./checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt`

## A note on the "sam3d fork" dependency

The SAM 3D Body model code lives **directly inside this repo** at [`sam_3d_body/`](./sam_3d_body/) —
it is **vendored (committed in-tree)**, not installed from an external fork or a git/pip dependency.
Because of that, this project does **not** depend on any separate GitHub fork at runtime, and nothing
here breaks if such a fork is deleted. Upstream updates were historically pulled in via the
`update-fork` branch (which mirrors `facebookresearch/sam-3d-body`) and merged in — see the
`chore: merge with updated fork` commit. To restore that "sync from upstream" capability, simply
re-fork [`facebookresearch/sam-3d-body`](https://github.com/facebookresearch/sam-3d-body) on GitHub.
</content>
