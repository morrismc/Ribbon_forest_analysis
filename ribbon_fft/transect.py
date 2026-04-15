"""Transect-based peak/trough detection for ribbon forest analysis.

Generates east-west transects across the detrended DSM, applies Gaussian
smoothing to suppress tree-crown noise, detects ribbon crests and glade
troughs via scipy.signal.find_peaks, and pairs each crest with its
downwind (eastward) trough to measure amplitude and spacing.

Follows the methodology outlined in the transect peak detection briefing,
informed by the 65.9 m dominant spectral peak from the 2D FFT analysis.
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def smooth_transect(profile, sigma=5):
    """Gaussian-smooth a 1-D transect, handling NaN gaps.

    Parameters
    ----------
    profile : numpy.ndarray
        1-D elevation profile.
    sigma : float
        Gaussian smoothing sigma in pixels.

    Returns
    -------
    smoothed : numpy.ndarray or None
        Smoothed profile, or None if too few valid pixels.
    """
    valid = ~np.isnan(profile)
    if valid.sum() < 100:
        return None
    smoothed = profile.copy()
    smoothed[valid] = gaussian_filter1d(profile[valid], sigma=sigma)
    return smoothed


def detect_peaks_troughs(profile, min_distance=80, min_prominence=1.0):
    """Detect ribbon crests (peaks) and glade minima (troughs).

    Parameters
    ----------
    profile : numpy.ndarray
        1-D smoothed elevation profile.
    min_distance : int
        Minimum distance between peaks in pixels.
    min_prominence : float
        Minimum prominence of peaks in data units (m).

    Returns
    -------
    peaks : numpy.ndarray
        Indices of detected peaks.
    troughs : numpy.ndarray
        Indices of detected troughs.
    """
    valid = ~np.isnan(profile)
    if valid.sum() < min_distance * 2:
        return None, None

    peaks, _ = find_peaks(profile, distance=min_distance, prominence=min_prominence)
    troughs, _ = find_peaks(-profile, distance=min_distance // 2)

    return peaks, troughs


def measure_crest_trough_pairs(profile, peaks, troughs, dx=0.5):
    """Pair each peak with its downwind (eastward) trough and measure.

    Parameters
    ----------
    profile : numpy.ndarray
        1-D smoothed elevation profile.
    peaks : numpy.ndarray
        Peak indices.
    troughs : numpy.ndarray
        Trough indices.
    dx : float
        Grid spacing in metres.

    Returns
    -------
    measurements : list of dict
        One dict per crest-trough pair with amplitude and spacing.
    """
    measurements = []

    for peak_idx in peaks:
        downstream_troughs = troughs[troughs > peak_idx]
        if len(downstream_troughs) == 0:
            continue

        next_trough_idx = downstream_troughs[0]
        upstream_troughs = troughs[troughs < peak_idx]

        crest_height = profile[peak_idx]
        trough_depth = profile[next_trough_idx]
        amplitude = crest_height - trough_depth

        if len(upstream_troughs) > 0:
            prev_trough_depth = profile[upstream_troughs[-1]]
            amplitude_mean = crest_height - np.mean([trough_depth, prev_trough_depth])
        else:
            amplitude_mean = amplitude

        trough_distance_m = (next_trough_idx - peak_idx) * dx

        downstream_peaks = peaks[peaks > peak_idx]
        crest_to_crest_m = (downstream_peaks[0] - peak_idx) * dx if len(downstream_peaks) > 0 else np.nan

        measurements.append({
            "peak_idx": peak_idx,
            "peak_x_m": peak_idx * dx,
            "peak_elevation": crest_height,
            "trough_idx": next_trough_idx,
            "trough_elevation": trough_depth,
            "amplitude_downwind": amplitude,
            "amplitude_mean_adj": amplitude_mean,
            "crest_to_trough_m": trough_distance_m,
            "crest_to_crest_m": crest_to_crest_m,
        })

    return measurements


def run_transect_analysis(dsm_detrended, dx=0.5, n_transects=100,
                          sigma_pixels=5, min_distance_pixels=80,
                          min_prominence=1.0, buffer_rows=50, seed=42):
    """Run the full transect peak detection pipeline.

    Parameters
    ----------
    dsm_detrended : numpy.ndarray
        2-D detrended DSM.
    dx : float
        Grid spacing in metres.
    n_transects : int
        Number of random east-west transects.
    sigma_pixels : float
        Gaussian smoothing sigma in pixels.
    min_distance_pixels : int
        Minimum distance between peaks in pixels.
    min_prominence : float
        Minimum peak prominence in metres.
    buffer_rows : int
        Row buffer from edges.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    df : pandas.DataFrame
        All crest-trough pair measurements.
    transect_rows : numpy.ndarray
        Row indices of the transects used.
    """
    nrows, ncols = dsm_detrended.shape

    rng = np.random.default_rng(seed)
    valid_rows = np.arange(buffer_rows, nrows - buffer_rows)
    n_transects = min(n_transects, len(valid_rows))
    transect_rows = rng.choice(valid_rows, size=n_transects, replace=False)
    transect_rows.sort()

    all_measurements = []

    for i, row_idx in enumerate(transect_rows):
        profile = dsm_detrended[row_idx, :].copy()
        smoothed = smooth_transect(profile, sigma=sigma_pixels)
        if smoothed is None:
            continue

        peaks, troughs = detect_peaks_troughs(
            smoothed, min_distance=min_distance_pixels,
            min_prominence=min_prominence,
        )
        if peaks is None or len(peaks) < 2:
            continue

        pairs = measure_crest_trough_pairs(smoothed, peaks, troughs, dx=dx)
        for pair in pairs:
            pair["transect_id"] = i
            pair["row_idx"] = int(row_idx)
            pair["y_m"] = row_idx * dx

        all_measurements.extend(pairs)

    df = pd.DataFrame(all_measurements)
    return df, transect_rows
