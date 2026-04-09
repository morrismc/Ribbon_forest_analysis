"""Sliding-window spectral power mapping.

Computes spectral power within a target frequency band at every pixel
using a sliding window, following the approach of Booth et al. (2009)
fft_powsum.m.
"""

import numpy as np
from .spectral import compute_2d_power_spectrum, radial_average, dominant_frequency


def sliding_window_power(dsm, dx, window_size=257, step=None,
                         freq_band=(0.01, 0.033)):
    """Map spectral power in a frequency band across the DSM using a sliding window.

    Parameters
    ----------
    dsm : numpy.ndarray
        2-D detrended DSM.
    dx : float
        Grid spacing (m).
    window_size : int
        Side length of the square analysis window (pixels).
    step : int, optional
        Stride between window centres. Defaults to window_size // 4.
    freq_band : tuple of float
        (f_min, f_max) frequency band of interest (cycles/m).

    Returns
    -------
    power_map : numpy.ndarray
        2-D array (coarser grid) of integrated power in the target band.
    wavelength_map : numpy.ndarray
        2-D array of dominant wavelength within the band at each window.
    row_centres : numpy.ndarray
        Row indices of window centres in the original DSM grid.
    col_centres : numpy.ndarray
        Column indices of window centres in the original DSM grid.
    """
    if step is None:
        step = window_size // 4

    nrows, ncols = dsm.shape
    half = window_size // 2

    # Window centre positions
    row_starts = np.arange(half, nrows - half, step)
    col_starts = np.arange(half, ncols - half, step)

    power_map = np.full((len(row_starts), len(col_starts)), np.nan)
    wavelength_map = np.full_like(power_map, np.nan)

    for i, r in enumerate(row_starts):
        for j, c in enumerate(col_starts):
            patch = dsm[r - half:r + half + 1, c - half:c + half + 1]

            # Skip patches with too many NaNs
            if np.isnan(patch).sum() > 0.1 * patch.size:
                continue

            power_2d, freq_x, freq_y = compute_2d_power_spectrum(
                patch, dx, apply_window=True, zero_pad=True
            )
            freq_r, power_r = radial_average(power_2d, freq_x, freq_y)

            # Integrate power in the frequency band
            in_band = (freq_r >= freq_band[0]) & (freq_r <= freq_band[1])
            if in_band.any():
                power_map[i, j] = np.sum(power_r[in_band])

                # Dominant wavelength within the band
                peak_f, peak_wl, _ = dominant_frequency(
                    freq_r, power_r, freq_range=freq_band
                )
                wavelength_map[i, j] = peak_wl

    return power_map, wavelength_map, row_starts, col_starts
