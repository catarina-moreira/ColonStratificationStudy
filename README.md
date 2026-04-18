# SATO Colon Straightening Pipeline

End-to-end pipeline for straightening the colon from abdominal CT scans using:
- **TotalSegmentator** for automatic colon segmentation
- **3D skeletonization + graph traversal** for centerline extraction
- **SATO** (Straighten Any 3D Tubular Object) for curved planar reformation

Based on the paper:
> Zhou et al., *SATO: Straighten Any 3D Tubular Object*, IEEE TMI, 2025.
> Code: https://github.com/Yanfeng-Zhou/SATO

---

## Repository Structure

```
SATO/
├── pipeline.py       ← full pipeline (DICOM → mask → centerline → straighten)
├── visualize.py      ← visualisation functions (CT slices, mask, centerline, results)
├── demo.ipynb        ← interactive notebook with maths, code and plots
├── requirements.txt  ← Python dependencies
├── repo/             ← original SATO repository (cloned from GitHub)
│   └── straighten/
│       ├── straighten_img.py
│       ├── straighten_seg.py
│       └── parallel_straighten.py
├── data/             ← CT Colonography DICOM series
└── output/           ← generated files (created on first run)
    ├── ct.nii.gz
    ├── segmentation/colon.nii.gz
    ├── centerline.npy
    ├── straightened_image.tif
    └── straightened_mask.tif
```

---

## Setup

```bash
# 1. Clone this repo / navigate to the project directory
cd /path/to/SATO

# 2. Create Python virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# (CPU-only PyTorch — already the default in requirements.txt)
# For GPU: pip install torch torchvision  # uses CUDA by default
```

---

## Run the Full Pipeline

```bash
source venv/bin/activate

python pipeline.py \
  --dicom_dir  "data/manifest-1776479825092/CT COLONOGRAPHY/1.3.6.1.4.1.9328.50.4.0518/01-01-2000-1-CT COLONOGRAPHY-83932/2.000000-SUPINE-83936" \
  --output_dir output
```

### Optional flags

| Flag | Default | Description |
|------|---------|-------------|
| `--crop_radius N` | auto (5×max_radius) | Cross-section half-width in voxels |
| `--no_fast` | — | Use full-resolution TotalSegmentator (slower, more accurate) |
| `--skip_totalseg` | — | Skip segmentation (reuse existing `colon.nii.gz`) |
| `--skip_centerline` | — | Skip centerline (reuse existing `centerline.npy`) |

### Expected runtimes (CPU, M-series Mac)

| Step | Time |
|------|------|
| DICOM → NIfTI | ~10 s |
| TotalSegmentator (`--fast`) | ~5–15 min |
| Centerline extraction | ~1–3 min |
| Straightening (image + mask) | ~2–5 min |

---

## Interactive Notebook

```bash
source venv/bin/activate
jupyter notebook demo.ipynb
```

The notebook walks through every step with:
- mathematical derivation of Zhou's Swept Frame
- inline visualisations of the CT, mask, centerline, and straightened result
- annotated code cells

---

## Centerline Format (SATO `.npy` schema)

The centerline is saved as a NumPy object array containing a Python dict:

```python
{
    "edge_length":      float,           # arc length in voxels → number of output slices
    "edge_width":       float,           # maximum local radius (voxels)
    "start_coordinate": np.int16[3],     # [Z, Y, X] voxel index of start
    "end_coordinate":   np.int16[3],     # [Z, Y, X] voxel index of end
    "point": [
        {
            "coordinate": np.int16[3],   # [Z, Y, X] voxel index
            "width":      float          # local tube radius (voxels, from EDT)
        },
        ...                              # one entry per skeleton voxel
    ]
}
```

**Coordinate convention**: indices match the NumPy array axes after
`sitk.GetArrayFromImage()`, i.e. `[axis-0 = Z, axis-1 = Y, axis-2 = X]`.

---

## What is Implemented vs What the Paper Claims

| Component | Status |
|-----------|--------|
| Zhou's Swept Frame (core algorithm) | ✅ fully implemented (`pipeline.py`) |
| Image volume straightening | ✅ fully implemented |
| Segmentation mask straightening | ✅ fully implemented |
| Colon segmentation (TotalSegmentator) | ✅ integrated |
| Centerline extraction | ✅ implemented (skeletonize + BFS) |
| Parallel batch processing | ✅ available (`repo/straighten/parallel_straighten.py`) |
| Downstream morphological analysis | ❌ not included |
| Full paper experimental suite | ❌ requires additional datasets |

---

## Reproducibility Notes

- **Segmentation quality directly affects centerline quality.** If TotalSegmentator
  misses a segment of the colon, the centerline will be truncated there.
- **Centerline extraction** uses 3-D skeletonization (Lee 1994) which can produce
  spurious branches at haustral folds; these are suppressed by the longest-path
  BFS traversal.
- **Coordinate system**: centerline voxel indices must match the array order of the
  image as loaded by SimpleITK. The pipeline guarantees this by computing both
  from the same NIfTI file.
- The original SATO requirements pin `numpy==1.21.6`, which does not install on
  Python 3.13. This pipeline uses current library versions, which are compatible.
