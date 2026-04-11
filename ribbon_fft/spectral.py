"""2D FFT spectral analysis: windowing, power spectrum, and radial averaging.

Follows the approach of Perron et al. (2008) and the Python port by
Purinton (DEM-FFT).
"""

import numpy as np
from scipy.signal.windows import hann


def hann2d(nrows, ncols):
    """Create a 2-D Hann (raised cosine) window via outer product.

    Parameters
    ----------
    nrows, ncols : int
        Window dimensions.

    Returns
    -------
    window : numpy.ndarray
        2-D Hann window, shape (nrows, ncols).
    """
    return np.outer(hann(nrows), hann(ncols))


def next_power_of_2(n):
    """Return the smallest power of 2 >= n."""
    return 1 << (int(n) - 1).bit_length()


def compute_2d_power_spectrum(data, dx, apply_window=True, zero_pad=True):
    """Compute the 2-D power spectral density of a detrended surface.

    Parameters
    ----------
    data : numpy.ndarray
        2-D array (detrended DSM or patch). NaNs will be replaced with 0.
    dx : float
        Grid spacing in map units (m).
    apply_window : bool
        If True, apply a 2-D Hann window before the FFT.
    zero_pad : bool
        If True, zero-pad to the next power of 2 in each dimension.

    Returns
    -------
    power : numpy.ndarray
        2-D power spectrum (shifted so zero frequency is centred).
    freq_x : numpy.ndarray
        1-D array of spatial frequencies along x (cycles / map unit).
    freq_y : numpy.ndarray
        1-D array of spatial frequencies along y (cycles / map unit).
    """
    data = np.nan_to_num(data, nan=0.0)
    nrows, ncols = data.shape

    # Apply 2-D Hann window
    if apply_window:
        window = hann2d(nrows, ncols)
        # Normalise so windowing doesn't change total power
        # Correction factor = mean(window^2) — preserves power spectral density
        window_correction = np.sqrt(np.mean(window ** 2))
        data = data * window / window_correction

    # Zero-pad
    if zero_pad:
        nfft_r = next_power_of_2(nrows)
        nfft_c = next_power_of_2(ncols)
    else:
        nfft_r = nrows
        nfft_c = ncols

    # 2-D FFT
    fft2 = np.fft.fft2(data, s=(nfft_r, nfft_c))
    fft2_shifted = np.fft.fftshift(fft2)

    # Power spectral density (periodogram)
    # Normalise by number of points and grid spacing to get PSD in physical units
    power = (np.abs(fft2_shifted) ** 2) * (dx ** 2) / (nfft_r * nfft_c)

    # Frequency axes
    freq_y = np.fft.fftshift(np.fft.fftfreq(nfft_r, d=dx))
    freq_x = np.fft.fftshift(np.fft.fftfreq(nfft_c, d=dx))

    return power, freq_x, freq_y


def radial_average(power, freq_x, freq_y, n_bins=None):
    """Radially average a 2-D power spectrum to produce a 1-D spectrum.

    Parameters
    ----------
    power : numpy.ndarray
        2-D power spectrum (zero-frequency centred).
    freq_x, freq_y : numpy.ndarray
        1-D frequency axes.
    n_bins : int, optional
        Number of radial bins. Defaults to half the smaller FFT dimension.

    Returns
    -------
    freq_r : numpy.ndarray
        Radial frequency bin centres.
    power_r : numpy.ndarray
        Radially averaged power at each frequency.
    """
    # Build 2-D frequency-magnitude grid
    fx, fy = np.meshgrid(freq_x, freq_y)
    freq_mag = np.sqrt(fx ** 2 + fy ** 2)

    # Exclude the DC component (zero frequency)
    max_freq = freq_mag.max()
    if n_bins is None:
        n_bins = min(power.shape) // 2

    bin_edges = np.linspace(0, max_freq, n_bins + 1)
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    power_r = np.zeros(n_bins)
    for i in range(n_bins):
        in_bin = (freq_mag >= bin_edges[i]) & (freq_mag < bin_edges[i + 1])
        if in_bin.any():
            power_r[i] = np.mean(power[in_bin])

    return bin_centres, power_r


def dominant_frequency(freq_r, power_r, freq_range=None):
    """Find the dominant (peak-power) frequency in a 1-D radial spectrum.

    Parameters
    ----------
    freq_r : numpy.ndarray
        Radial frequency bin centres.
    power_r : numpy.ndarray
        Radially averaged power.
    freq_range : tuple of float, optional
        (f_min, f_max) — restrict the search to this frequency band.

    Returns
    -------
    peak_freq : float
        Frequency of the spectral peak (cycles / map unit).
    peak_wavelength : float
        Corresponding wavelength (map units).
    peak_power : float
        Power at the peak.
    """
    if freq_range is not None:
        mask = (freq_r >= freq_range[0]) & (freq_r <= freq_range[1])
        idx_offset = np.argmax(power_r[mask])
        idx = np.where(mask)[0][idx_offset]
    else:
        # Skip DC (index 0)
        idx = np.argmax(power_r[1:]) + 1

    peak_freq = freq_r[idx]
    peak_wavelength = 1.0 / peak_freq if peak_freq > 0 else np.inf
    peak_power = power_r[idx]

    return peak_freq, peak_wavelength, peak_power
