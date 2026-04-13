"""2D Continuous Wavelet Transform analysis for ribbon forest spacing.

Uses a 2D Mexican Hat (Ricker) wavelet convolved at multiple spatial
scales to produce per-pixel maps of dominant wavelength and amplitude.
This gives much finer spatial localisation of the spacing-amplitude
relationship than the sliding-window FFT approach.

Kernel normalisation follows the continuous CWT convention of Booth et al.
(2009, Eq. 3) and Kumar & Foufoula-Georgiou (1994): a 1/s prefactor ensures
that wavelet coefficients are directly comparable across scales.  The
discrete convolution is scaled by dx^2 to approximate the continuous
double integral.

For directional analysis, an anisotropic (elongated) Morlet can be used
instead — this is left as a future extension.
"""

import numpy as np
from scipy.signal import fftconvolve


def mexican_hat_2d(size, scale):
    """Generate a 2D Mexican Hat (Ricker) wavelet kernel.

    Parameters
    ----------
    size : int
        Kernel side length in pixels (should be odd).
    scale : float
        Wavelet scale parameter sigma (in pixels).

    Returns
    -------
    kernel : numpy.ndarray
        2-D wavelet kernel with 1/s CWT normalisation (Booth et al. 2009,
        conv2_mexh.m line 79).
    """
    half = size // 2
    y, x = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    r2 = x ** 2 + y ** 2
    s2 = scale ** 2

    # Standard CWT kernel with 1/s prefactor (Booth et al. Eq. 3).
    # This ensures coefficients represent comparable physical amplitudes
    # across scales — without this, small scales are over-weighted.
    kernel = (1.0 / scale) * (2.0 - r2 / s2) * np.exp(-r2 / (2.0 * s2))

    return kernel


def cwt_2d(data, dx, scales_m=None, n_scales=20):
    """Compute a 2D continuous wavelet transform at multiple scales.

    Parameters
    ----------
    data : numpy.ndarray
        2-D detrended DSM.
    dx : float
        Grid spacing in map units (m).
    scales_m : array-like of float, optional
        Wavelet scales in map units (m). If None, logarithmically spaced
        from 10 m to 150 m.
    n_scales : int
        Number of scales if scales_m is not provided.

    Returns
    -------
    coefficients : numpy.ndarray
        3-D array of shape (n_scales, nrows, ncols) — wavelet
        coefficients at each scale and pixel.
    scales_m : numpy.ndarray
        1-D array of scales in map units.
    """
    if scales_m is None:
        scales_m = np.geomspace(10.0, 150.0, n_scales)
    else:
        scales_m = np.asarray(scales_m, dtype=float)

    # Replace NaN with 0 for convolution
    data_clean = np.nan_to_num(data, nan=0.0)

    nrows, ncols = data_clean.shape
    coefficients = np.zeros((len(scales_m), nrows, ncols), dtype=np.float64)

    for i, s_m in enumerate(scales_m):
        # Convert scale from metres to pixels
        s_px = s_m / dx

        # Kernel size: 6 sigma should capture the wavelet extent
        ksize = int(np.ceil(6 * s_px))
        if ksize % 2 == 0:
            ksize += 1
        ksize = max(ksize, 3)

        kernel = mexican_hat_2d(ksize, s_px)
        # FFT-based convolution is much faster for large kernels.
        # Multiply by dx^2 to approximate the continuous double integral
        # (each pixel covers dx * dx area).  See Booth conv2_mexh.m line 83.
        conv = fftconvolve(data_clean, kernel, mode="same")
        coefficients[i] = conv * (dx ** 2)

    return coefficients, scales_m


def cwt_dominant_scale(coefficients, scales_m):
    """Extract per-pixel dominant scale (wavelength) and amplitude from CWT.

    The dominant scale at each pixel is the one with the highest absolute
    wavelet coefficient.  The amplitude is the coefficient value at that
    scale, which is proportional to the local signal strength at that
    wavelength.

    For a Mexican Hat wavelet, the characteristic wavelength detected at
    scale s is approximately lambda = 2*pi*s / sqrt(2.5) ~ 4.0 * s.

    Parameters
    ----------
    coefficients : numpy.ndarray
        3-D CWT output (n_scales, nrows, ncols).
    scales_m : numpy.ndarray
        1-D array of scales in map units.

    Returns
    -------
    dominant_wavelength : numpy.ndarray
        2-D map of dominant wavelength (m) at each pixel.
    amplitude : numpy.ndarray
        2-D map of wavelet amplitude (absolute coefficient) at the
        dominant scale.
    dominant_scale_idx : numpy.ndarray
        2-D map of the index into scales_m at each pixel.
    """
    # Mexican Hat: characteristic wavelength ≈ 2*pi*s / sqrt(2.5)
    SCALE_TO_WAVELENGTH = 2.0 * np.pi / np.sqrt(2.5)

    abs_coeff = np.abs(coefficients)
    dominant_scale_idx = np.argmax(abs_coeff, axis=0)

    nrows, ncols = dominant_scale_idx.shape
    amplitude = np.zeros((nrows, ncols))
    dominant_wavelength = np.zeros((nrows, ncols))

    for i in range(len(scales_m)):
        mask = dominant_scale_idx == i
        amplitude[mask] = abs_coeff[i][mask]
        dominant_wavelength[mask] = scales_m[i] * SCALE_TO_WAVELENGTH

    return dominant_wavelength, amplitude, dominant_scale_idx


def cwt_amplitude_spacing(dominant_wavelength, amplitude, wavelength_range=None):
    """Extract paired amplitude-spacing samples for correlation analysis.

    Parameters
    ----------
    dominant_wavelength : numpy.ndarray
        2-D map of dominant wavelength (m).
    amplitude : numpy.ndarray
        2-D map of wavelet amplitude.
    wavelength_range : tuple of float, optional
        (min_wl, max_wl) — only include pixels within this range.
        Defaults to (20, 120) m.

    Returns
    -------
    wavelengths : numpy.ndarray
        1-D array of wavelengths at sampled pixels.
    amplitudes : numpy.ndarray
        1-D array of amplitudes at sampled pixels.
    r_value : float
        Pearson correlation coefficient.
    p_value : float
        Two-sided p-value.
    """
    from scipy.stats import pearsonr

    if wavelength_range is None:
        wavelength_range = (20.0, 120.0)

    valid = (
        np.isfinite(dominant_wavelength)
        & np.isfinite(amplitude)
        & (dominant_wavelength >= wavelength_range[0])
        & (dominant_wavelength <= wavelength_range[1])
        & (amplitude > 0)
    )

    wl = dominant_wavelength[valid].ravel()
    amp = amplitude[valid].ravel()

    # Subsample if there are too many points for the scatter plot
    if len(wl) > 10000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(wl), 10000, replace=False)
        wl_sample = wl[idx]
        amp_sample = amp[idx]
    else:
        wl_sample = wl
        amp_sample = amp

    if len(wl_sample) > 2:
        r_value, p_value = pearsonr(wl_sample, amp_sample)
    else:
        r_value, p_value = np.nan, np.nan

    return wl_sample, amp_sample, r_value, p_value
