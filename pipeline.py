"""
SATO colon straightening pipeline.

Steps:
  1. Load DICOM series → save as NIfTI
  2. Run TotalSegmentator → extract colon binary mask
  3. Compute colon centerline from mask (skeletonize + graph traversal)
  4. Straighten image and mask with SATO (Zhou's swept frame)

Usage:
  cd /Users/162191/Documents/GitHub/SATO
  source venv/bin/activate
  python pipeline.py [--output_dir output] [--crop_radius 40]
"""

import argparse
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.interpolate import splprep, splev, interp1d
from scipy.spatial.transform import Rotation
from skimage.morphology import skeletonize, remove_small_holes

# ---------------------------------------------------------------------------
# Step 1: DICOM → NIfTI
# ---------------------------------------------------------------------------

def dicom_to_nifti(dicom_dir: str, out_path: str) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in {dicom_dir}")
    files = reader.GetGDCMSeriesFileNames(dicom_dir, series_ids[0])
    reader.SetFileNames(files)
    image = reader.Execute()
    image = sitk.Cast(image, sitk.sitkInt16)
    sitk.WriteImage(image, out_path)
    arr = sitk.GetArrayFromImage(image)
    print(f"[1] NIfTI saved: {out_path}  shape={arr.shape}  "
          f"spacing={image.GetSpacing()}")
    return image


# ---------------------------------------------------------------------------
# Step 2: TotalSegmentator → colon mask
# ---------------------------------------------------------------------------

def run_totalsegmentator(nifti_path: str, seg_dir: str, fast: bool = True):
    """Segment colon with TotalSegmentator; saves colon.nii.gz in seg_dir."""
    from totalsegmentator.python_api import totalsegmentator

    colon_path = os.path.join(seg_dir, "colon.nii.gz")
    if os.path.exists(colon_path):
        print(f"[2] Colon mask already exists, skipping.")
        return

    os.makedirs(seg_dir, exist_ok=True)
    print(f"[2] Running TotalSegmentator (fast={fast}) on CPU — this takes ~5-15 min...")
    totalsegmentator(
        input=nifti_path,
        output=seg_dir,
        roi_subset=["colon"],
        fast=fast,
        device="cpu",
        quiet=True,
    )
    if not os.path.exists(colon_path):
        raise RuntimeError(f"colon.nii.gz not found in {seg_dir} after TotalSegmentator.")
    print(f"[2] Colon mask saved: {colon_path}")


# ---------------------------------------------------------------------------
# Step 3: Centerline extraction from binary mask
# ---------------------------------------------------------------------------

def _build_graph(skel: np.ndarray):
    """26-connected adjacency list for skeleton voxels.
    Returns (adj dict {node_idx: [neighbour_idxs]}, coords array (N,3))."""
    coords = np.argwhere(skel)          # (N, 3) in [Z, Y, X] voxel order
    if len(coords) == 0:
        return {}, coords

    shape = skel.shape
    # map flat index → node index
    flat = np.ravel_multi_index(coords.T, shape)
    idx_of = {int(f): i for i, f in enumerate(flat)}

    offsets = [
        (dz, dy, dx)
        for dz in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
        if not (dz == 0 and dy == 0 and dx == 0)
    ]

    adj = {i: [] for i in range(len(coords))}
    for i, (z, y, x) in enumerate(coords):
        for dz, dy, dx in offsets:
            nz, ny, nx = z + dz, y + dy, x + dx
            if 0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]:
                nf = int(nz * shape[1] * shape[2] + ny * shape[2] + nx)
                if nf in idx_of:
                    adj[i].append(idx_of[nf])
    return adj, coords


def _bfs(adj: dict, start: int):
    """BFS; returns (furthest_node, dist_dict, parent_dict)."""
    from collections import deque
    dist = {start: 0}
    parent = {start: -1}
    q = deque([start])
    furthest = start
    while q:
        node = q.popleft()
        for nb in adj[node]:
            if nb not in dist:
                dist[nb] = dist[node] + 1
                parent[nb] = node
                q.append(nb)
                if dist[nb] > dist[furthest]:
                    furthest = nb
    return furthest, dist, parent


def _trace(parent: dict, end: int) -> list:
    path = []
    n = end
    while n != -1:
        path.append(n)
        n = parent[n]
    return path[::-1]


def compute_centerline(mask_arr: np.ndarray) -> dict:
    """
    Extract the colon centreline from a binary mask.

    Returns a SATO-format dict:
      edge_length  – arc length in voxels (float)
      edge_width   – maximum radius (float)
      point        – list of {'coordinate': np.int16[3], 'width': float}
                     coordinates are in voxel array order [Z, Y, X]
    """
    print("[3] Computing centerline...")

    # --- keep largest connected component ---
    labeled, n_comp = ndimage.label(mask_arr > 0)
    if n_comp == 0:
        raise RuntimeError("Colon mask is empty.")
    sizes = ndimage.sum(mask_arr > 0, labeled, range(1, n_comp + 1))
    binary = (labeled == int(np.argmax(sizes)) + 1).astype(np.uint8)
    print(f"    Mask voxels (largest CC): {binary.sum():,}")

    # --- distance transform (radius at each voxel) ---
    dist_map = ndimage.distance_transform_edt(binary)

    # --- 3D skeletonize (Lee 1994 via skimage) ---
    print("    Skeletonizing…")
    skel = skeletonize(binary.astype(bool))
    print(f"    Skeleton voxels: {int(skel.sum()):,}")

    # --- build graph and find largest connected component ---
    adj, coords = _build_graph(skel)

    visited = set()
    components = []
    for seed in range(len(coords)):
        if seed not in visited:
            comp, stack = [], [seed]
            while stack:
                n = stack.pop()
                if n in visited:
                    continue
                visited.add(n)
                comp.append(n)
                stack.extend(adj[n])
            components.append(comp)

    main_nodes = set(max(components, key=len))
    adj_main = {n: [nb for nb in adj[n] if nb in main_nodes]
                for n in main_nodes}

    # --- find longest path via double BFS ---
    ep0 = next(iter(main_nodes))
    far1, _, _       = _bfs(adj_main, ep0)
    far2, _, parent  = _bfs(adj_main, far1)
    main_path = _trace(parent, far2)
    print(f"    Main path: {len(main_path)} nodes")

    # --- build per-point info ---
    path_coords = coords[main_path]                      # (L, 3) [Z,Y,X]
    radii = dist_map[path_coords[:, 0],
                     path_coords[:, 1],
                     path_coords[:, 2]]

    diffs = np.diff(path_coords.astype(float), axis=0)
    arc_len = float(np.sum(np.linalg.norm(diffs, axis=1)))
    print(f"    Arc length: {arc_len:.1f} voxels, max radius: {radii.max():.1f} vx")

    points = [
        {"coordinate": path_coords[i].astype(np.int16), "width": float(radii[i])}
        for i in range(len(path_coords))
    ]

    return {
        "start_point":      0,
        "end_point":        len(points) - 1,
        "start_coordinate": path_coords[0].astype(np.int16),
        "end_coordinate":   path_coords[-1].astype(np.int16),
        "edge_length":      arc_len,
        "edge_width":       float(radii.max()),
        "point":            points,
    }


# ---------------------------------------------------------------------------
# Step 4: SATO straightening (exactly mirrors the original scripts)
# ---------------------------------------------------------------------------

def straighten_volume(image_arr: np.ndarray,
                      centerline: dict,
                      crop_radius: int | None = None,
                      is_seg: bool = False) -> np.ndarray:
    """
    Straighten image_arr along the centerline using Zhou's swept frame.

    Parameters
    ----------
    image_arr   : np.ndarray (Z, Y, X)
    centerline  : SATO dict (output of compute_centerline)
    crop_radius : half-width of output cross-section in voxels;
                  default = int(5 × max_width) capped at 80
    is_seg      : nearest-neighbour interp + binary postprocessing

    Returns
    -------
    np.ndarray (L, 2R+1, 2R+1)
    """
    pts    = centerline["point"]
    length = int(centerline["edge_length"])

    # --- interpolate centerline with cubic spline ---
    raw_coords = np.array([p["coordinate"] for p in pts], dtype=float)
    raw_radii  = np.array([p["width"]      for p in pts], dtype=float)

    tck, u = splprep(raw_coords.T.tolist(), s=0)
    u_new  = np.linspace(u.min(), u.max(), length)

    dx, dy, dz           = splev(u_new, tck, der=1)
    x_c, y_c, z_c        = splev(u_new, tck)
    radii_interp          = interp1d(
        np.linspace(0, 1, len(raw_radii)), raw_radii, kind="linear"
    )(np.linspace(0, 1, length))

    if crop_radius is None:
        crop_radius = min(int(5 * radii_interp.max()), 80)

    R = crop_radius
    print(f"    crop_radius={R}, slices={length}, patch={2*R+1}×{2*R+1}")

    # --- pre-build the local cross-section grid (4 × (2R+1)²) ---
    lin   = np.linspace(-R, R, 2*R+1)
    gu    = np.tile(lin, 2*R+1)
    gv    = np.repeat(np.arange(R, -R-1, -1), 2*R+1)
    gz    = np.zeros((2*R+1)**2)
    gones = np.ones((2*R+1)**2)
    local_grid = np.stack([gu, gv, gz, gones], axis=0)   # (4, N)

    order = 0 if is_seg else 1

    # --- swept-frame loop (identical logic to original SATO scripts) ---
    slices = []
    ux_before = uy_before = uz_before = None

    for m in range(length):
        uz = np.array([dx[m], dy[m], dz[m]])

        if m == 0:
            if dx[m] == 0:
                ux = np.array([0.0, -dz[m], dy[m]])
            elif dy[m] == 0:
                ux = np.array([-dz[m], 0.0, dx[m]])
            else:
                ux = np.array([-dy[m], dx[m], 0.0])
            uy = np.cross(uz, ux)
        else:
            if np.allclose(uz, uz_before):
                ux = ux_before
                uy = uy_before
            else:
                intersect = np.cross(uz_before, uz)
                intersect_n = intersect / np.linalg.norm(intersect)
                cos_a = np.clip(
                    np.dot(uz, uz_before)
                    / (np.linalg.norm(uz) * np.linalg.norm(uz_before)),
                    -1.0, 1.0,
                )
                theta = np.arccos(cos_a)
                rot = Rotation.from_rotvec(theta * intersect_n)
                ux  = rot.apply(ux_before)
                uy  = rot.apply(uy_before)

        ux_n = ux / np.linalg.norm(ux)
        uy_n = uy / np.linalg.norm(uy)
        uz_n = uz / np.linalg.norm(uz)

        ux_before = ux_n
        uy_before = uy_n
        uz_before = uz_n   # store normalised (matches original uz_before = uz_normal)

        T = np.array([
            [ux_n[0], uy_n[0], uz_n[0], x_c[m]],
            [ux_n[1], uy_n[1], uz_n[1], y_c[m]],
            [ux_n[2], uy_n[2], uz_n[2], z_c[m]],
            [0,       0,       0,       1      ],
        ])

        world = (T @ local_grid)[:3, :]   # (3, N)
        vals  = ndimage.map_coordinates(image_arr, world, order=order)
        slices.append(vals.reshape(2*R+1, 2*R+1))

    result = np.stack(slices, axis=0)   # (L, 2R+1, 2R+1)

    if is_seg:
        result = (result >= 0.5).astype(bool)
        result = remove_small_holes(result, area_threshold=200)
        result = ndimage.median_filter(result.astype(np.uint8), size=3)
        result = result.astype(np.uint8)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DICOM_DIR = (
    "data/manifest-1776479825092/CT COLONOGRAPHY/"
    "1.3.6.1.4.1.9328.50.4.0518/"
    "01-01-2000-1-CT COLONOGRAPHY-83932/"
    "2.000000-SUPINE-83936"
)


def main():
    parser = argparse.ArgumentParser(description="SATO colon straightening pipeline")
    parser.add_argument("--dicom_dir",  default=DICOM_DIR)
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--crop_radius", type=int, default=None,
                        help="Cross-section half-width in voxels (default 5×max_radius)")
    parser.add_argument("--fast",    action="store_true", default=True,
                        help="TotalSegmentator --fast (3 mm model, ~5 min CPU)")
    parser.add_argument("--no_fast", action="store_true", default=False,
                        help="Use full-resolution TotalSegmentator model")
    parser.add_argument("--skip_totalseg",   action="store_true")
    parser.add_argument("--skip_centerline", action="store_true")
    args = parser.parse_args()

    base = Path(__file__).parent
    out  = base / args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "nifti":        out / "ct.nii.gz",
        "colon_mask":   out / "segmentation" / "colon.nii.gz",
        "seg_dir":      out / "segmentation",
        "centerline":   out / "centerline.npy",
        "straight_img": out / "straightened_image.tif",
        "straight_seg": out / "straightened_mask.tif",
    }

    # Step 1 ----------------------------------------------------------------
    if not paths["nifti"].exists():
        dicom_to_nifti(str(base / args.dicom_dir), str(paths["nifti"]))
    else:
        print(f"[1] NIfTI exists: {paths['nifti']}")

    # Step 2 ----------------------------------------------------------------
    if not args.skip_totalseg:
        run_totalsegmentator(
            str(paths["nifti"]),
            str(paths["seg_dir"]),
            fast=not args.no_fast,
        )
    else:
        print("[2] Skipping TotalSegmentator.")

    # Step 3 ----------------------------------------------------------------
    if not args.skip_centerline or not paths["centerline"].exists():
        mask_itk = sitk.ReadImage(str(paths["colon_mask"]))
        mask_arr = sitk.GetArrayFromImage(mask_itk).astype(np.uint8)
        print(f"    Mask shape: {mask_arr.shape}, nonzero voxels: {(mask_arr>0).sum():,}")
        cl = compute_centerline(mask_arr)
        np.save(str(paths["centerline"]),
                np.array(cl, dtype=object), allow_pickle=True)
        print(f"[3] Centerline: {len(cl['point'])} pts, "
              f"arc={cl['edge_length']:.0f} vx, max_r={cl['edge_width']:.1f} vx")
    else:
        print(f"[3] Loading existing centerline: {paths['centerline']}")
        cl = np.load(str(paths["centerline"]), allow_pickle=True).tolist()

    # Step 4a: straighten raw image -----------------------------------------
    print("[4a] Straightening CT image…")
    img_arr = sitk.GetArrayFromImage(
        sitk.ReadImage(str(paths["nifti"]))
    ).astype(np.float32)

    s_img = straighten_volume(img_arr, cl,
                              crop_radius=args.crop_radius, is_seg=False)
    sitk.WriteImage(sitk.GetImageFromArray(s_img.astype(np.int16)),
                    str(paths["straight_img"]))
    print(f"    Saved: {paths['straight_img']}  shape={s_img.shape}")

    # Step 4b: straighten mask -----------------------------------------------
    print("[4b] Straightening colon mask…")
    mask_arr2 = sitk.GetArrayFromImage(
        sitk.ReadImage(str(paths["colon_mask"]))
    ).astype(np.float32)

    s_seg = straighten_volume(mask_arr2, cl,
                              crop_radius=args.crop_radius, is_seg=True)
    sitk.WriteImage(sitk.GetImageFromArray(s_seg),
                    str(paths["straight_seg"]))
    print(f"    Saved: {paths['straight_seg']}  shape={s_seg.shape}")

    print("\n=== Done ===")
    for k, p in paths.items():
        print(f"  {k:15s}: {p}")


if __name__ == "__main__":
    main()
