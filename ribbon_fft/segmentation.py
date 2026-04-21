"""CHM-based ribbon segmentation for ribbon forest analysis.

Implements the methodology from the "CHM-based ribbon segmentation" briefing:

1. Build CHM from raw DSM - DTM (do NOT use detrended DSM — that mixes
   topography with canopy).
2. Smooth the CHM with a small NaN-aware Gaussian (sigma = 4 px = 2 m at
   0.5 m/px) to suppress individual tree-crown noise while preserving
   ribbon boundaries.
3. Threshold at 1.5 m to produce a binary canopy/glade mask, then apply
   3x3 morphological opening (no closing — glades are real and should
   remain open).
4. For each east-west transect, extract contiguous runs of canopy,
   reject runs shorter than 10 m (20 px), and measure per-ribbon:
   leading_edge, trailing_edge, crest_x, crest_amplitude_max, crest_mean.
5. Pair consecutive ribbons along each transect and measure
   pure_glade_width (trailing-to-leading), crest_to_downwind_edge
   (crest-to-leading of next), and crest_to_crest.

Returns two data frames — one per ribbon, one per pair — suitable for
histograms, Spearman rank tests, and diagnostic plots.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import binary_opening, gaussian_filter


def smooth_chm_nan(chm, sigma=4.0):
    """Apply a NaN-aware 2-D Gaussian filter to the CHM.

    Uses the weighted-convolution trick: convolve the zero-filled data
    and the validity mask separately, then divide.  Pixels whose local
    Gaussian weight falls near zero (i.e. isolated in a sea of NaN) are
    returned as NaN.

    Parameters
    ----------
    chm : numpy.ndarray
        2-D canopy height model.  NaN where data is missing.
    sigma : float
        Gaussian sigma in pixels.  Default 4 px (2 m at 0.5 m/px).

    Returns
    -------
    smoothed : numpy.ndarray
        Smoothed CHM (NaN preserved where the Gaussian support is empty).
    """
    valid = np.isfinite(chm).astype(np.float64)
    filled = np.where(valid > 0, chm, 0.0).astype(np.float64)

    num = gaussian_filter(filled, sigma=sigma, mode="reflect")
    den = gaussian_filter(valid, sigma=sigma, mode="reflect")

    smoothed = np.full_like(num, np.nan)
    good = den > 1e-3
    smoothed[good] = num[good] / den[good]
    return smoothed


def build_ribbon_mask(chm_smooth, threshold=1.5, opening_size=3):
    """Threshold the smoothed CHM and apply morphological opening.

    Parameters
    ----------
    chm_smooth : numpy.ndarray
        Smoothed CHM in metres.
    threshold : float
        Canopy height threshold (m).  Pixels with height >= threshold
        are classified as canopy.
    opening_size : int
        Side length of the square structuring element for binary opening.
        Use 3 for a 3x3 kernel (removes isolated pixels / thin spurs).
        No closing is applied — glades are physically meaningful openings.

    Returns
    -------
    mask : numpy.ndarray of bool
        True = canopy (ribbon), False = glade or NaN.
    """
    canopy = np.where(np.isfinite(chm_smooth), chm_smooth >= threshold, False)
    if opening_size and opening_size > 1:
        struct = np.ones((opening_size, opening_size), dtype=bool)
        canopy = binary_opening(canopy, structure=struct)
    return canopy.astype(bool)


def extract_runs(mask_row):
    """Find contiguous True runs along a 1-D boolean array.

    Parameters
    ----------
    mask_row : numpy.ndarray of bool
        1-D canopy mask for a single transect.

    Returns
    -------
    starts : numpy.ndarray
        Leading-edge pixel index of each run (inclusive).
    ends : numpy.ndarray
        Trailing-edge pixel index of each run (inclusive).
    """
    m = mask_row.astype(np.int8)
    # Pad with zeros so edge runs are detected via diff.
    padded = np.concatenate(([0], m, [0]))
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0] - 1
    return starts, ends


def measure_ribbons_on_transect(chm_row, mask_row, dx=0.5,
                                min_length_m=10.0):
    """Extract ribbons on one transect, reject short runs, and measure.

    Parameters
    ----------
    chm_row : numpy.ndarray
        1-D smoothed CHM (m) for this transect.
    mask_row : numpy.ndarray of bool
        1-D binary canopy mask for this transect (after opening).
    dx : float
        Grid spacing in metres.
    min_length_m : float
        Minimum ribbon width along-transect to retain (m).

    Returns
    -------
    ribbons : list of dict
        One dict per retained ribbon with leading_edge_idx,
        trailing_edge_idx, leading_edge_m, trailing_edge_m, width_m,
        crest_idx, crest_x_m, crest_amplitude_max, crest_mean.
    n_rejected : int
        Number of runs that failed the length threshold.
    """
    starts, ends = extract_runs(mask_row)
    min_length_px = int(np.ceil(min_length_m / dx))

    ribbons = []
    n_rejected = 0
    for s, e in zip(starts, ends):
        run_length_px = e - s + 1
        if run_length_px < min_length_px:
            n_rejected += 1
            continue

        segment = chm_row[s:e + 1]
        # Some pixels inside the run may be NaN if smoothing couldn't fill
        # them; skip ribbons where we have no finite samples to measure.
        finite = np.isfinite(segment)
        if not finite.any():
            n_rejected += 1
            continue

        # Crest = tallest pixel within the ribbon.
        local_crest = int(np.nanargmax(segment))
        crest_idx = s + local_crest
        crest_amplitude_max = float(segment[local_crest])
        crest_mean = float(np.nanmean(segment))

        ribbons.append({
            "leading_edge_idx": int(s),
            "trailing_edge_idx": int(e),
            "leading_edge_m": s * dx,
            "trailing_edge_m": e * dx,
            "width_m": run_length_px * dx,
            "crest_idx": crest_idx,
            "crest_x_m": crest_idx * dx,
            "crest_amplitude_max": crest_amplitude_max,
            "crest_mean": crest_mean,
        })

    return ribbons, n_rejected


def measure_pairs(ribbons, dx=0.5):
    """Pair consecutive ribbons along one transect and measure.

    Pairing follows the along-transect order (west -> east).  For each
    ribbon i with a downwind neighbour i+1:

      pure_glade_width       = leading_{i+1} - trailing_i
      crest_to_downwind_edge = leading_{i+1} - crest_i
      crest_to_crest         = crest_{i+1}   - crest_i

    Parameters
    ----------
    ribbons : list of dict
        Output of measure_ribbons_on_transect (sorted west -> east).
    dx : float
        Grid spacing in metres (unused — distances already in metres).

    Returns
    -------
    pairs : list of dict
    """
    pairs = []
    for i in range(len(ribbons) - 1):
        a = ribbons[i]
        b = ribbons[i + 1]
        pure_glade_width = b["leading_edge_m"] - a["trailing_edge_m"]
        crest_to_downwind_edge = b["leading_edge_m"] - a["crest_x_m"]
        crest_to_crest = b["crest_x_m"] - a["crest_x_m"]

        pairs.append({
            "crest_x_m": a["crest_x_m"],
            "crest_amplitude_max": a["crest_amplitude_max"],
            "crest_mean": a["crest_mean"],
            "downwind_crest_x_m": b["crest_x_m"],
            "downwind_leading_edge_m": b["leading_edge_m"],
            "pure_glade_width": pure_glade_width,
            "crest_to_downwind_edge": crest_to_downwind_edge,
            "crest_to_crest": crest_to_crest,
        })
    return pairs


def run_segmentation_analysis(chm, dx=0.5, sigma_pixels=4.0,
                              chm_threshold=1.5, opening_size=3,
                              min_ribbon_length_m=10.0,
                              n_transects=100, buffer_rows=50, seed=42):
    """Run the full CHM-based ribbon segmentation pipeline.

    Parameters
    ----------
    chm : numpy.ndarray
        Canopy height model (m).  May contain NaN.
    dx : float
        Grid spacing in metres.
    sigma_pixels : float
        Gaussian smoothing sigma in pixels (default 4 -> 2 m at 0.5 m).
    chm_threshold : float
        Canopy/glade threshold in metres (default 1.5 m).
    opening_size : int
        Morphological opening structuring-element size in pixels.
    min_ribbon_length_m : float
        Minimum along-transect ribbon width (m).
    n_transects : int
        Number of random east-west transects to sample.
    buffer_rows : int
        Rows to skip at the top and bottom of the scene.
    seed : int
        Random seed for transect selection.

    Returns
    -------
    result : dict with keys
        chm_smooth : smoothed CHM array
        mask : binary ribbon mask (bool)
        ribbons_df : pandas.DataFrame, one row per ribbon
        pairs_df   : pandas.DataFrame, one row per ribbon pair
        transect_rows : np.ndarray of row indices used
        stats : dict of summary statistics (counts, negative CHM, etc.)
    """
    nrows, ncols = chm.shape

    # --- Validation counters on the raw CHM ---
    negative_chm_count = int(np.sum((chm < 0) & np.isfinite(chm)))
    total_finite = int(np.sum(np.isfinite(chm)))

    # --- Smooth + threshold + open ---
    chm_smooth = smooth_chm_nan(chm, sigma=sigma_pixels)
    mask = build_ribbon_mask(chm_smooth, threshold=chm_threshold,
                             opening_size=opening_size)

    # --- Pick transects ---
    rng = np.random.default_rng(seed)
    buffer_rows = min(buffer_rows, max(nrows // 4 - 1, 0))
    lo = buffer_rows
    hi = nrows - buffer_rows
    valid_rows = np.arange(lo, hi)
    n_take = min(n_transects, len(valid_rows))
    if n_take <= 0:
        transect_rows = np.array([], dtype=int)
    else:
        transect_rows = rng.choice(valid_rows, size=n_take, replace=False)
        transect_rows.sort()

    # --- Per-transect extraction ---
    all_ribbons = []
    all_pairs = []
    total_rejected = 0
    ribbons_per_transect = []

    for tid, row_idx in enumerate(transect_rows):
        chm_row = chm_smooth[row_idx, :]
        mask_row = mask[row_idx, :]

        ribbons, n_rej = measure_ribbons_on_transect(
            chm_row, mask_row, dx=dx, min_length_m=min_ribbon_length_m,
        )
        total_rejected += n_rej
        ribbons_per_transect.append(len(ribbons))

        for rank, r in enumerate(ribbons):
            r["transect_id"] = int(tid)
            r["row_idx"] = int(row_idx)
            r["y_m"] = row_idx * dx
            r["ribbon_rank"] = int(rank)
        all_ribbons.extend(ribbons)

        pairs = measure_pairs(ribbons, dx=dx)
        for rank, p in enumerate(pairs):
            p["transect_id"] = int(tid)
            p["row_idx"] = int(row_idx)
            p["y_m"] = row_idx * dx
            p["pair_rank"] = int(rank)
        all_pairs.extend(pairs)

    ribbons_df = pd.DataFrame(all_ribbons)
    pairs_df = pd.DataFrame(all_pairs)

    ribbons_per_transect = np.array(ribbons_per_transect, dtype=int)
    stats = {
        "n_transects": int(len(transect_rows)),
        "n_ribbons": int(len(ribbons_df)),
        "n_pairs": int(len(pairs_df)),
        "n_rejected_short_runs": int(total_rejected),
        "ribbons_per_transect_mean": (
            float(ribbons_per_transect.mean()) if len(ribbons_per_transect) else float("nan")
        ),
        "ribbons_per_transect_median": (
            float(np.median(ribbons_per_transect)) if len(ribbons_per_transect) else float("nan")
        ),
        "negative_chm_pixels": negative_chm_count,
        "finite_chm_pixels": total_finite,
        "negative_chm_fraction": (
            negative_chm_count / total_finite if total_finite > 0 else float("nan")
        ),
        "canopy_fraction": float(mask.sum() / mask.size),
    }

    return {
        "chm_smooth": chm_smooth,
        "mask": mask,
        "ribbons_df": ribbons_df,
        "pairs_df": pairs_df,
        "transect_rows": transect_rows,
        "stats": stats,
    }
