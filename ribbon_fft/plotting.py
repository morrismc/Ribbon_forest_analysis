"""Visualization functions for ribbon forest spectral analysis."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot_dsm(dsm, transform=None, title="Digital Surface Model", cmap="terrain",
             ax=None, save_path=None):
    """Plot the DSM as a 2-D image.

    Parameters
    ----------
    dsm : numpy.ndarray
        2-D elevation array.
    transform : rasterio.Affine, optional
        Geotransform for axis labelling in map coordinates.
    title : str
        Plot title.
    cmap : str
        Matplotlib colormap name.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure is created.
    save_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    else:
        fig = ax.figure

    if transform is not None:
        # Compute extent in map coordinates
        nrows, ncols = dsm.shape
        left = transform.c
        top = transform.f
        right = left + ncols * transform.a
        bottom = top + nrows * transform.e
        extent = [left, right, bottom, top]
        im = ax.imshow(dsm, cmap=cmap, extent=extent)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
    else:
        im = ax.imshow(dsm, cmap=cmap)
        ax.set_xlabel("Column (pixels)")
        ax.set_ylabel("Row (pixels)")

    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Elevation (m)", shrink=0.8)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_hillshade(dsm, dx=1.0, azimuth=315, altitude=45, title="Hillshade",
                   transform=None, ax=None, save_path=None):
    """Compute and plot a simple hillshade from the DSM.

    Parameters
    ----------
    dsm : numpy.ndarray
        2-D elevation array.
    dx : float
        Grid spacing.
    azimuth : float
        Sun azimuth in degrees (0=N, clockwise).
    altitude : float
        Sun altitude in degrees above horizon.
    title : str
        Plot title.
    transform : rasterio.Affine, optional
        Geotransform for axis labelling.
    ax : matplotlib.axes.Axes, optional
    save_path : str, optional

    Returns
    -------
    fig, ax
    """
    # Replace NaN for gradient computation
    data = np.nan_to_num(dsm, nan=0.0)

    # Compute slope components
    dy_arr, dx_arr = np.gradient(data, dx)

    # Convert angles to radians
    az_rad = np.radians(360 - azimuth + 90)  # convert to math convention
    alt_rad = np.radians(altitude)

    # Hillshade formula
    shade = (
        np.sin(alt_rad)
        + np.cos(alt_rad) * np.cos(az_rad) * dx_arr
        + np.cos(alt_rad) * np.sin(az_rad) * dy_arr
    )
    shade = np.clip(shade, 0, 1)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    else:
        fig = ax.figure

    if transform is not None:
        nrows, ncols = dsm.shape
        left = transform.c
        top = transform.f
        right = left + ncols * transform.a
        bottom = top + nrows * transform.e
        extent = [left, right, bottom, top]
        ax.imshow(shade, cmap="gray", extent=extent)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
    else:
        ax.imshow(shade, cmap="gray")

    ax.set_title(title)

    if save_path:
        fig = ax.figure
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_detrended(detrended, transform=None, title="Detrended DSM",
                   ax=None, save_path=None):
    """Plot the detrended DSM with a diverging colormap centred on zero.

    Parameters
    ----------
    detrended : numpy.ndarray
    transform : rasterio.Affine, optional
    title : str
    ax : matplotlib.axes.Axes, optional
    save_path : str, optional

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    else:
        fig = ax.figure

    vmax = np.nanpercentile(np.abs(detrended), 99)
    vmin = -vmax

    if transform is not None:
        nrows, ncols = detrended.shape
        left = transform.c
        top = transform.f
        right = left + ncols * transform.a
        bottom = top + nrows * transform.e
        extent = [left, right, bottom, top]
        im = ax.imshow(detrended, cmap="RdBu_r", vmin=vmin, vmax=vmax, extent=extent)
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
    else:
        im = ax.imshow(detrended, cmap="RdBu_r", vmin=vmin, vmax=vmax)

    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Detrended elevation (m)", shrink=0.8)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_power_spectrum_1d(freq_r, power_r, dx=None, title="Radially Averaged Power Spectrum",
                           ax=None, save_path=None, annotate_peaks=True):
    """Plot 1-D radially averaged power spectrum on log-log axes.

    Parameters
    ----------
    freq_r : numpy.ndarray
        Radial frequency bin centres (cycles/m).
    power_r : numpy.ndarray
        Radially averaged power.
    dx : float, optional
        Grid spacing (for annotating Nyquist frequency).
    title : str
    ax : matplotlib.axes.Axes, optional
    save_path : str, optional
    annotate_peaks : bool
        If True, mark the dominant peak and annotate its wavelength.

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig = ax.figure

    # Filter out zero-frequency and zero-power bins
    valid = (freq_r > 0) & (power_r > 0)
    f = freq_r[valid]
    p = power_r[valid]

    ax.loglog(f, p, "k-", linewidth=0.8)

    if annotate_peaks and len(f) > 0:
        # Find the peak (skip very low frequencies)
        min_freq = f[1] if len(f) > 1 else 0
        search_mask = f > min_freq * 2
        if search_mask.any():
            peak_idx = np.argmax(p[search_mask])
            peak_f = f[search_mask][peak_idx]
            peak_p = p[search_mask][peak_idx]
            peak_wl = 1.0 / peak_f

            ax.axvline(peak_f, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
            ax.annotate(
                f"$\\lambda$ = {peak_wl:.1f} m",
                xy=(peak_f, peak_p),
                xytext=(peak_f * 2, peak_p * 2),
                arrowprops=dict(arrowstyle="->", color="red"),
                fontsize=10, color="red",
            )

    # Mark expected ribbon spacing range
    ax.axvspan(0.01, 0.033, alpha=0.1, color="green", label="Expected ribbon band (30\u2013100 m)")

    if dx is not None:
        nyquist = 1.0 / (2.0 * dx)
        ax.axvline(nyquist, color="gray", linestyle=":", alpha=0.5)
        ax.text(nyquist * 0.9, ax.get_ylim()[0] * 10, f"Nyquist ({nyquist:.1f})",
                rotation=90, va="bottom", fontsize=8, color="gray")

    ax.set_xlabel("Spatial frequency (cycles/m)")
    ax.set_ylabel("Power spectral density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_power_spectrum_2d(power, freq_x, freq_y,
                           title="2D Power Spectrum", ax=None, save_path=None):
    """Plot the 2-D power spectrum as an image.

    Parameters
    ----------
    power : numpy.ndarray
        2-D power spectrum (zero-frequency centred).
    freq_x, freq_y : numpy.ndarray
        Frequency axes.
    title : str
    ax : matplotlib.axes.Axes, optional
    save_path : str, optional

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    else:
        fig = ax.figure

    # Mask zero values for log scale
    power_plot = power.copy()
    power_plot[power_plot <= 0] = np.nan

    extent = [freq_x[0], freq_x[-1], freq_y[-1], freq_y[0]]

    im = ax.imshow(
        power_plot,
        norm=LogNorm(vmin=np.nanpercentile(power_plot, 5),
                     vmax=np.nanpercentile(power_plot, 99)),
        extent=extent,
        cmap="inferno",
        aspect="equal",
    )

    ax.set_xlabel("Frequency x (cycles/m)")
    ax.set_ylabel("Frequency y (cycles/m)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Power", shrink=0.8)

    # Crosshairs at zero
    ax.axhline(0, color="white", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="white", linewidth=0.5, alpha=0.5)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def plot_amplitude_vs_spacing(amplitudes, wavelengths, r_value, p_value,
                              ax=None, save_path=None):
    """Scatter plot of ribbon amplitude vs. dominant spacing wavelength.

    Parameters
    ----------
    amplitudes : numpy.ndarray
        Mean amplitude at each sample point.
    wavelengths : numpy.ndarray
        Dominant wavelength at each sample point.
    r_value : float
        Pearson r.
    p_value : float
        P-value.
    ax : matplotlib.axes.Axes, optional
    save_path : str, optional

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = ax.figure

    ax.scatter(wavelengths, amplitudes, s=15, alpha=0.5, edgecolors="none")

    # Fit line
    if len(wavelengths) > 2 and np.isfinite(r_value):
        z = np.polyfit(wavelengths, amplitudes, 1)
        x_fit = np.linspace(np.nanmin(wavelengths), np.nanmax(wavelengths), 100)
        ax.plot(x_fit, np.polyval(z, x_fit), "r-", linewidth=1.5,
                label=f"r = {r_value:.3f}, p = {p_value:.2e}")
        ax.legend(fontsize=10)

    ax.set_xlabel("Dominant wavelength (m)")
    ax.set_ylabel("Mean ribbon amplitude (m)")
    ax.set_title("Ribbon Amplitude vs. Spacing")
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig, ax


def create_summary_figure(dsm, detrended, power_2d, freq_x, freq_y,
                          freq_r, power_r, dx, transform=None,
                          save_path=None):
    """Create a 4-panel summary figure (Phase 1 outputs).

    Panels:
    1. Hillshade of original DSM
    2. Detrended DSM
    3. 2D power spectrum
    4. 1D radially averaged power spectrum

    Parameters
    ----------
    dsm : numpy.ndarray
        Original DSM.
    detrended : numpy.ndarray
        Detrended DSM.
    power_2d : numpy.ndarray
        2-D power spectrum.
    freq_x, freq_y : numpy.ndarray
        Frequency axes.
    freq_r : numpy.ndarray
        Radial frequencies.
    power_r : numpy.ndarray
        Radial power.
    dx : float
        Grid spacing.
    transform : rasterio.Affine, optional
    save_path : str, optional

    Returns
    -------
    fig
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    plot_hillshade(dsm, dx=dx, transform=transform, ax=axes[0, 0],
                   title="Hillshade of DSM")
    plot_detrended(detrended, transform=transform, ax=axes[0, 1])
    plot_power_spectrum_2d(power_2d, freq_x, freq_y, ax=axes[1, 0])
    plot_power_spectrum_1d(freq_r, power_r, dx=dx, ax=axes[1, 1])

    fig.suptitle("Ribbon Forest Spectral Analysis \u2014 Snowy Range, WY", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig
