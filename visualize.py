"""
Visualisation utilities for the SATO colon straightening pipeline.

All functions return the matplotlib Figure so callers can call
fig.savefig(...) or fig.show() as needed.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D   # noqa: F401 (needed for 3-D projection)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _mid(arr: np.ndarray, axis: int) -> int:
    return arr.shape[axis] // 2


def _window(arr: np.ndarray, wl: float = 40, ww: float = 400):
    """Apply CT window (window-level / window-width)."""
    lo = wl - ww / 2
    hi = wl + ww / 2
    return np.clip((arr.astype(float) - lo) / (hi - lo), 0, 1)


# ---------------------------------------------------------------------------
# 1. CT multi-planar view
# ---------------------------------------------------------------------------

def plot_ct_slices(img_arr: np.ndarray,
                   spacing: tuple = (1.0, 1.0, 1.0),
                   wl: float = 40, ww: float = 400,
                   title: str = "CT — axial / coronal / sagittal") -> plt.Figure:
    """
    Show the central axial, coronal, and sagittal slices of a CT volume.

    Parameters
    ----------
    img_arr : (Z, Y, X) numpy array (Hounsfield units)
    spacing : (z_mm, y_mm, x_mm) voxel spacing
    wl, ww  : CT window level and width (default: abdomen soft-tissue)
    """
    windowed = _window(img_arr, wl, ww)
    z0, y0, x0 = _mid(img_arr, 0), _mid(img_arr, 1), _mid(img_arr, 2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # axial
    axes[0].imshow(windowed[z0], cmap="gray", origin="upper",
                   aspect=spacing[1] / spacing[2])
    axes[0].set_title(f"Axial (z={z0})")
    axes[0].axis("off")

    # coronal  (Z × X plane, flip so superior is up)
    axes[1].imshow(windowed[:, y0, :], cmap="gray", origin="lower",
                   aspect=spacing[0] / spacing[2])
    axes[1].set_title(f"Coronal (y={y0})")
    axes[1].axis("off")

    # sagittal (Z × Y plane)
    axes[2].imshow(windowed[:, :, x0], cmap="gray", origin="lower",
                   aspect=spacing[0] / spacing[1])
    axes[2].set_title(f"Sagittal (x={x0})")
    axes[2].axis("off")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Mask overlay on CT slices
# ---------------------------------------------------------------------------

def plot_mask_overlay(img_arr: np.ndarray,
                      mask_arr: np.ndarray,
                      spacing: tuple = (1.0, 1.0, 1.0),
                      wl: float = 40, ww: float = 400,
                      mask_color: str = "lime",
                      alpha: float = 0.35,
                      title: str = "Colon mask overlay") -> plt.Figure:
    """
    Overlay a binary mask on CT slices.  Chooses the slice with the most
    mask voxels for each plane so the colon is always visible.
    """
    windowed = _window(img_arr, wl, ww)
    mask = (mask_arr > 0).astype(float)

    # pick slice with most mask voxels per plane
    z0 = int(np.argmax(mask.sum(axis=(1, 2))))
    y0 = int(np.argmax(mask.sum(axis=(0, 2))))
    x0 = int(np.argmax(mask.sum(axis=(0, 1))))

    def _rgba(gray_2d, mask_2d):
        h, w = gray_2d.shape
        rgba = np.zeros((h, w, 4))
        rgba[..., :3] = gray_2d[..., None]
        rgba[..., 3]  = 1.0
        # paint mask pixels
        c = plt.cm.colors.to_rgb(mask_color)
        for i in range(3):
            rgba[..., i] = np.where(mask_2d > 0,
                                    gray_2d * (1 - alpha) + c[i] * alpha,
                                    gray_2d)
        return rgba

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0].imshow(_rgba(windowed[z0], mask[z0]),
                   origin="upper", aspect=spacing[1] / spacing[2])
    axes[0].set_title(f"Axial (z={z0})")
    axes[0].axis("off")

    axes[1].imshow(_rgba(windowed[:, y0, :], mask[:, y0, :]),
                   origin="lower", aspect=spacing[0] / spacing[2])
    axes[1].set_title(f"Coronal (y={y0})")
    axes[1].axis("off")

    axes[2].imshow(_rgba(windowed[:, :, x0], mask[:, :, x0]),
                   origin="lower", aspect=spacing[0] / spacing[1])
    axes[2].set_title(f"Sagittal (x={x0})")
    axes[2].axis("off")

    patch = mpatches.Patch(color=mask_color, alpha=alpha, label="Colon mask")
    fig.legend(handles=[patch], loc="lower center", ncol=1, fontsize=11)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Centerline overlay on CT / mask slices
# ---------------------------------------------------------------------------

def plot_centerline_on_slices(img_arr: np.ndarray,
                               mask_arr: np.ndarray,
                               centerline: dict,
                               spacing: tuple = (1.0, 1.0, 1.0),
                               wl: float = 40, ww: float = 400,
                               title: str = "Centerline overlay") -> plt.Figure:
    """
    Draw the centerline path on axial, coronal, and sagittal MIP-like projections.
    Centerline coordinates are in voxel [Z, Y, X] order.
    """
    pts = np.array([p["coordinate"] for p in centerline["point"]])
    cz, cy, cx = pts[:, 0], pts[:, 1], pts[:, 2]

    windowed = _window(img_arr, wl, ww)
    mask = (mask_arr > 0).astype(float)

    # Maximum intensity projection of the mask as background
    mip_axial   = windowed.max(axis=0)                  # Y × X
    mip_coronal = windowed.max(axis=1)                  # Z × X
    mip_sagit   = windowed.max(axis=2)                  # Z × Y

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes[0].imshow(mip_axial, cmap="gray", origin="upper",
                   aspect=spacing[1] / spacing[2])
    axes[0].plot(cx, cy, "r-", linewidth=1.5, label="centerline")
    axes[0].scatter(cx[0], cy[0], c="lime",   s=40, zorder=5, label="start")
    axes[0].scatter(cx[-1], cy[-1], c="red",  s=40, zorder=5, label="end")
    axes[0].set_title("Axial MIP + centerline")
    axes[0].axis("off")

    axes[1].imshow(mip_coronal, cmap="gray", origin="lower",
                   aspect=spacing[0] / spacing[2])
    axes[1].plot(cx, cz, "r-", linewidth=1.5)
    axes[1].scatter(cx[0], cz[0], c="lime",  s=40, zorder=5)
    axes[1].scatter(cx[-1], cz[-1], c="red", s=40, zorder=5)
    axes[1].set_title("Coronal MIP + centerline")
    axes[1].axis("off")

    axes[2].imshow(mip_sagit, cmap="gray", origin="lower",
                   aspect=spacing[0] / spacing[1])
    axes[2].plot(cy, cz, "r-", linewidth=1.5)
    axes[2].scatter(cy[0], cz[0], c="lime",  s=40, zorder=5)
    axes[2].scatter(cy[-1], cz[-1], c="red", s=40, zorder=5)
    axes[2].set_title("Sagittal MIP + centerline")
    axes[2].axis("off")

    axes[0].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. 3-D centerline + mask surface (downsampled)
# ---------------------------------------------------------------------------

def plot_centerline_3d(mask_arr: np.ndarray,
                       centerline: dict,
                       spacing: tuple = (1.0, 1.0, 1.0),
                       downsample: int = 4,
                       title: str = "3-D centerline") -> plt.Figure:
    """
    3-D scatter of the (downsampled) mask surface voxels and the centerline.
    spacing = (z_mm, y_mm, x_mm) used to scale axes to physical mm.
    """
    pts = np.array([p["coordinate"] for p in centerline["point"]])

    # surface voxels (boundary of mask, downsampled for speed)
    eroded = mask_arr > 0
    surface_coords = np.argwhere(eroded)[::downsample]

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    # mask surface
    ax.scatter(
        surface_coords[:, 2] * spacing[2],
        surface_coords[:, 1] * spacing[1],
        surface_coords[:, 0] * spacing[0],
        c="steelblue", alpha=0.03, s=1,
    )

    # centerline
    ax.plot(
        pts[:, 2] * spacing[2],
        pts[:, 1] * spacing[1],
        pts[:, 0] * spacing[0],
        "r-", linewidth=2, label="centerline",
    )
    ax.scatter(*[pts[0, i] * spacing[[2, 1, 0][i]] for i in range(3)],
               c="lime", s=60, zorder=5, label="start")
    ax.scatter(*[pts[-1, i] * spacing[[2, 1, 0][i]] for i in range(3)],
               c="red", s=60, zorder=5, label="end")

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title(title, fontweight="bold")
    ax.legend()
    fig.suptitle("Colon mask + extracted centerline (physical space)", fontsize=12)
    return fig


# ---------------------------------------------------------------------------
# 5. Straightened image + mask
# ---------------------------------------------------------------------------

def plot_straightened(straight_img: np.ndarray,
                      straight_seg: np.ndarray | None = None,
                      n_slices: int = 8,
                      title: str = "Straightened colon") -> plt.Figure:
    """
    Display evenly-spaced cross-sectional slices of the straightened volume.

    Parameters
    ----------
    straight_img : (L, H, W) — straightened CT intensity
    straight_seg : (L, H, W) — straightened binary mask (optional)
    n_slices     : how many slices to display
    """
    L = straight_img.shape[0]
    indices = np.linspace(0, L - 1, n_slices, dtype=int)

    rows = 2 if straight_seg is not None else 1
    fig, axes = plt.subplots(rows, n_slices,
                             figsize=(2.5 * n_slices, 3 * rows))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    if rows == 1:
        axes = axes[np.newaxis, :]

    for col, idx in enumerate(indices):
        sl = straight_img[idx].astype(float)
        sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8)
        axes[0, col].imshow(sl, cmap="gray", origin="upper")
        axes[0, col].set_title(f"z={idx}", fontsize=8)
        axes[0, col].axis("off")

        if straight_seg is not None:
            axes[1, col].imshow(straight_seg[idx], cmap="Blues",
                                vmin=0, vmax=1, origin="upper")
            axes[1, col].set_title(f"seg z={idx}", fontsize=8)
            axes[1, col].axis("off")

    axes[0, 0].set_ylabel("CT", fontsize=9)
    if straight_seg is not None:
        axes[1, 0].set_ylabel("Mask", fontsize=9)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Centerline radius profile
# ---------------------------------------------------------------------------

def plot_radius_profile(centerline: dict,
                        spacing: tuple = (1.0, 1.0, 1.0),
                        title: str = "Colon radius along centerline") -> plt.Figure:
    """
    Line plot of local tube radius vs arc-position along the centreline.
    """
    pts = np.array([p["coordinate"] for p in centerline["point"]], dtype=float)
    radii = np.array([p["width"] for p in centerline["point"]])

    diffs = np.diff(pts, axis=0)
    step_len = np.linalg.norm(diffs * np.array(spacing)[[0, 1, 2]], axis=1)
    arc_mm = np.concatenate([[0], np.cumsum(step_len)])

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.fill_between(arc_mm, 0, radii * spacing[1], alpha=0.3, color="steelblue")
    ax.plot(arc_mm, radii * spacing[1], color="steelblue", linewidth=1.5)
    ax.set_xlabel("Arc length (mm)")
    ax.set_ylabel("Radius (mm)")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 7. Full pipeline summary
# ---------------------------------------------------------------------------

def plot_pipeline_summary(img_arr: np.ndarray,
                          mask_arr: np.ndarray,
                          centerline: dict,
                          straight_img: np.ndarray,
                          straight_seg: np.ndarray,
                          spacing: tuple = (1.0, 1.0, 1.0),
                          wl: float = 40, ww: float = 400) -> plt.Figure:
    """
    Single figure summarising: original CT | mask overlay | straightened.
    """
    windowed = _window(img_arr, wl, ww)
    z0 = int(np.argmax((mask_arr > 0).sum(axis=(1, 2))))

    mid = straight_img.shape[0] // 2
    s_img = straight_img[mid].astype(float)
    s_img = (s_img - s_img.min()) / (s_img.max() - s_img.min() + 1e-8)

    pts = np.array([p["coordinate"] for p in centerline["point"]])

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle("SATO Colon Straightening Pipeline", fontsize=15, fontweight="bold")

    # panel 1: raw CT
    axes[0].imshow(windowed[z0], cmap="gray", origin="upper",
                   aspect=spacing[1] / spacing[2])
    axes[0].set_title("1. Raw CT (axial)", fontsize=11)
    axes[0].axis("off")

    # panel 2: mask overlay
    mask_slice = (mask_arr[z0] > 0).astype(float)
    axes[1].imshow(windowed[z0], cmap="gray", origin="upper",
                   aspect=spacing[1] / spacing[2])
    axes[1].imshow(np.ma.masked_where(mask_slice == 0, mask_slice),
                   cmap="Greens", alpha=0.5, origin="upper",
                   aspect=spacing[1] / spacing[2])
    axes[1].set_title("2. TotalSegmentator\ncolon mask", fontsize=11)
    axes[1].axis("off")

    # panel 3: centerline on axial MIP
    mip = windowed.max(axis=0)
    axes[2].imshow(mip, cmap="gray", origin="upper",
                   aspect=spacing[1] / spacing[2])
    axes[2].plot(pts[:, 2], pts[:, 1], "r-", linewidth=1.5)
    axes[2].scatter(pts[0, 2], pts[0, 1], c="lime", s=50, zorder=5)
    axes[2].scatter(pts[-1, 2], pts[-1, 1], c="red", s=50, zorder=5)
    axes[2].set_title("3. Extracted centerline", fontsize=11)
    axes[2].axis("off")

    # panel 4: central straightened slice
    axes[3].imshow(s_img, cmap="gray", origin="upper")
    axes[3].set_title("4. Straightened\ncross-section", fontsize=11)
    axes[3].axis("off")

    plt.tight_layout()
    return fig
