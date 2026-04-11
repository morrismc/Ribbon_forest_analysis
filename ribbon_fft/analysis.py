"""Amplitude extraction and spacing-amplitude correlation analysis."""

import numpy as np
from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter


def extract_amplitude(detrended, kernel_size=51):
    """Extract local ribbon amplitude (peak-to-trough range) from a detrended DSM.

    Uses a sliding window to compute the difference between local max and min,
    giving the peak-to-trough amplitude of the vegetation signal.

    Parameters
    ----------
    detrended : numpy.ndarray
        2-D detrended DSM.
    kernel_size : int
        Side length of the local analysis window (pixels).

    Returns
    -------
    amplitude : numpy.ndarray
        2-D array of local peak-to-trough amplitude.
    """
    # Replace NaN with 0 for the filters
    data = np.nan_to_num(detrended, nan=0.0)

    local_max = maximum_filter(data, size=kernel_size)
    local_min = minimum_filter(data, size=kernel_size)
    amplitude = local_max - local_min

    # Mask where original was NaN
    amplitude[np.isnan(detrended)] = np.nan

    return amplitude


def amplitude_spacing_correlation(amplitude_map, wavelength_map,
                                  row_centres, col_centres, dsm_shape):
    """Correlate local ribbon amplitude with local dominant wavelength (spacing).

    Parameters
    ----------
    amplitude_map : numpy.ndarray
        Full-resolution amplitude map from extract_amplitude().
    wavelength_map : numpy.ndarray
        Coarse-grid dominant wavelength from sliding_window_power().
    row_centres, col_centres : numpy.ndarray
        Row/column indices of the wavelength_map grid in the full DSM.
    dsm_shape : tuple of int
        Shape of the full DSM for bounds checking.

    Returns
    -------
    amplitudes : numpy.ndarray
        1-D array of mean amplitude at each window centre.
    wavelengths : numpy.ndarray
        1-D array of dominant wavelength at each window centre.
    r_value : float
        Pearson correlation coefficient.
    p_value : float
        Two-sided p-value for the correlation.
    """
    from scipy.stats import pearsonr

    amplitudes = []
    wavelengths = []

    # Sample the full-resolution amplitude map at the wavelength grid centres
    half_kernel = 25  # average amplitude over a 51x51 neighbourhood

    for i, r in enumerate(row_centres):
        for j, c in enumerate(col_centres):
            wl = wavelength_map[i, j]
            if np.isnan(wl) or wl <= 0:
                continue

            r0 = max(r - half_kernel, 0)
            r1 = min(r + half_kernel + 1, dsm_shape[0])
            c0 = max(c - half_kernel, 0)
            c1 = min(c + half_kernel + 1, dsm_shape[1])

            local_amp = amplitude_map[r0:r1, c0:c1]
            mean_amp = np.nanmean(local_amp)

            if np.isfinite(mean_amp):
                amplitudes.append(mean_amp)
                wavelengths.append(wl)

    amplitudes = np.array(amplitudes)
    wavelengths = np.array(wavelengths)

    if len(amplitudes) > 2:
        r_value, p_value = pearsonr(wavelengths, amplitudes)
    else:
        r_value, p_value = np.nan, np.nan

    return amplitudes, wavelengths, r_value, p_value
