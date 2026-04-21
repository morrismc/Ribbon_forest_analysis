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

    # Mask zero/negative values for log scale
    power_plot = power.copy()
    power_plot[power_plot <= 0] = np.nan

    extent = [freq_x[0], freq_x[-1], freq_y[-1], freq_y[0]]

    # Compute percentiles on strictly positive values to keep LogNorm valid
    positive = power_plot[np.isfinite(power_plot) & (power_plot > 0)]
    if len(positive) > 0:
        vmin = np.percentile(positive, 5)
        vmax = np.percentile(positive, 99)
        vmin = max(vmin, positive.min())  # guarantee vmin > 0
    else:
        vmin, vmax = 1e-10, 1.0

    im = ax.imshow(
        power_plot,
        norm=LogNorm(vmin=vmin, vmax=vmax),
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


def plot_transect_summary(dsm_detrended, df, transect_rows, dx=0.5,
                          fft_peak_wavelength=65.9, save_path=None):
    """Create a 6-panel transect analysis summary figure.

    Panels:
    1. Example transect with raw/smoothed profiles and detected peaks/troughs
    2. Scatter: amplitude vs. crest-to-trough distance (H1 test)
    3. Histogram: crest-to-trough distances
    4. Histogram: crest amplitudes
    5. Histogram: crest-to-crest spacing
    6. Map of peak/trough locations on the detrended DSM

    Parameters
    ----------
    dsm_detrended : numpy.ndarray
        2-D detrended DSM.
    df : pandas.DataFrame
        Crest-trough measurements from run_transect_analysis().
    transect_rows : numpy.ndarray
        Row indices of the transects.
    dx : float
        Grid spacing.
    fft_peak_wavelength : float
        Dominant wavelength from FFT for reference lines.
    save_path : str, optional

    Returns
    -------
    fig
    """
    from .transect import smooth_transect, detect_peaks_troughs

    nrows, ncols = dsm_detrended.shape
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Panel 1: Example transect
    ax = axes[0, 0]
    example_row = transect_rows[len(transect_rows) // 2]
    raw = dsm_detrended[example_row, :]
    sm = smooth_transect(raw)
    x_m = np.arange(len(raw)) * dx
    ax.plot(x_m, raw, "gray", alpha=0.4, lw=0.5, label="Raw")
    if sm is not None:
        ax.plot(x_m, sm, "k", lw=1, label="Smoothed")
        p, t = detect_peaks_troughs(sm)
        if p is not None and len(p) > 0:
            ax.plot(p * dx, sm[p], "rv", ms=8, label="Crests")
        if t is not None and len(t) > 0:
            ax.plot(t * dx, sm[t], "b^", ms=6, label="Troughs")
    ax.set_xlabel("East\u2013West distance (m)")
    ax.set_ylabel("Detrended elevation (m)")
    ax.set_title(f"Example transect (row {example_row})")
    ax.legend(fontsize=8)

    # Panel 2: Scatter — H1 test
    ax = axes[0, 1]
    if len(df) > 0:
        sc = ax.scatter(
            df["crest_to_trough_m"], df["amplitude_downwind"],
            c=df["y_m"], cmap="viridis", alpha=0.5, s=20, edgecolors="none",
        )
        fig.colorbar(sc, ax=ax, label="N\u2013S position (m)")
    ax.set_xlabel("Crest to downwind trough (m)")
    ax.set_ylabel("Crest amplitude (m)")
    ax.set_title("H\u2081 test: Amplitude vs. trough distance")

    # Panel 3: Histogram of crest-to-trough distances
    ax = axes[0, 2]
    if len(df) > 0:
        ax.hist(df["crest_to_trough_m"], bins=30, color="steelblue",
                edgecolor="white", alpha=0.8)
    ax.axvline(fft_peak_wavelength, color="red", ls="--", lw=1.5,
               label=f"FFT peak ({fft_peak_wavelength:.1f} m)")
    ax.set_xlabel("Crest to trough distance (m)")
    ax.set_ylabel("Count")
    ax.set_title("Trough distance distribution")
    ax.legend()

    # Panel 4: Histogram of amplitudes
    ax = axes[1, 0]
    if len(df) > 0:
        ax.hist(df["amplitude_downwind"], bins=30, color="coral",
                edgecolor="white", alpha=0.8)
    ax.set_xlabel("Crest amplitude (m)")
    ax.set_ylabel("Count")
    ax.set_title("Amplitude distribution")

    # Panel 5: Crest-to-crest spacing
    ax = axes[1, 1]
    if len(df) > 0:
        crest_spacing = df["crest_to_crest_m"].dropna()
        if len(crest_spacing) > 0:
            ax.hist(crest_spacing, bins=30, color="seagreen",
                    edgecolor="white", alpha=0.8)
    ax.axvline(fft_peak_wavelength, color="red", ls="--", lw=1.5,
               label=f"FFT peak ({fft_peak_wavelength:.1f} m)")
    ax.set_xlabel("Crest-to-crest spacing (m)")
    ax.set_ylabel("Count")
    ax.set_title("Crest spacing distribution")
    ax.legend()

    # Panel 6: Map of peak/trough locations
    ax = axes[1, 2]
    vmax = np.nanpercentile(np.abs(dsm_detrended), 99)
    ax.imshow(
        dsm_detrended, cmap="RdBu_r", aspect="equal",
        extent=[0, ncols * dx, nrows * dx, 0],
        vmin=-vmax, vmax=vmax, alpha=0.7,
    )
    if len(df) > 0:
        ax.scatter(df["peak_x_m"], df["y_m"], c="red", s=3,
                   label="Crests", alpha=0.5)
        trough_x = df["trough_idx"] * dx
        ax.scatter(trough_x, df["y_m"], c="blue", s=3,
                   label="Troughs", alpha=0.5)
    ax.set_xlabel("East\u2013West (m)")
    ax.set_ylabel("North\u2013South (m)")
    ax.set_title("Peak/trough locations on DSM")
    ax.legend(fontsize=8)

    fig.suptitle("Ribbon Forest Transect Analysis", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig


def plot_segmentation_summary(chm, chm_smooth, mask, ribbons_df, pairs_df,
                              transect_rows, dx=0.5,
                              chm_threshold=1.5,
                              min_ribbon_length_m=10.0,
                              fft_peak_wavelength=65.9,
                              save_path=None):
    """7-panel diagnostic figure for CHM-based ribbon segmentation.

    Panels:
        1. Example transect — smoothed CHM with retained ribbons shaded
           green, rejected runs orange, crests marked, edges marked.
        2. Histogram of ribbon widths.
        3. Histogram of crest amplitudes (max canopy height per ribbon).
        4. H1 scatter: crest_amplitude_max vs. pure_glade_width.
        5. H1 scatter: crest_amplitude_max vs. crest_to_downwind_edge.
        6. Histogram of crest-to-crest spacing, with FFT peak marked.
        7. Spatial map: CHM background, ribbon mask overlay.

    Parameters
    ----------
    chm : numpy.ndarray
        Raw canopy height model (for the spatial panel).
    chm_smooth : numpy.ndarray
        Smoothed CHM (used for the example transect).
    mask : numpy.ndarray of bool
        Binary ribbon mask.
    ribbons_df, pairs_df : pandas.DataFrame
    transect_rows : numpy.ndarray
    dx : float
    chm_threshold : float
    min_ribbon_length_m : float
    fft_peak_wavelength : float
    save_path : str, optional

    Returns
    -------
    fig
    """
    from .segmentation import extract_runs

    nrows, ncols = mask.shape
    min_length_px = int(np.ceil(min_ribbon_length_m / dx))

    fig = plt.figure(figsize=(18, 13))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # --- Panel 1: Example transect ---
    ax1 = fig.add_subplot(gs[0, :])
    if len(transect_rows) > 0:
        example_row = int(transect_rows[len(transect_rows) // 2])
    else:
        example_row = nrows // 2
    x_m = np.arange(ncols) * dx
    ax1.plot(x_m, chm_smooth[example_row, :], "k", lw=0.8,
             label="Smoothed CHM (m)")
    ax1.axhline(chm_threshold, color="gray", ls="--", lw=0.8,
                label=f"Threshold = {chm_threshold} m")

    starts, ends = extract_runs(mask[example_row, :])
    kept_label_done = False
    rejected_label_done = False
    for s, e in zip(starts, ends):
        x0 = s * dx
        x1 = (e + 1) * dx
        run_px = e - s + 1
        if run_px >= min_length_px:
            ax1.axvspan(
                x0, x1, color="green", alpha=0.25,
                label="Ribbon (kept)" if not kept_label_done else None,
            )
            kept_label_done = True
        else:
            ax1.axvspan(
                x0, x1, color="orange", alpha=0.3,
                label="Rejected run" if not rejected_label_done else None,
            )
            rejected_label_done = True

    row_ribbons = ribbons_df[ribbons_df["row_idx"] == example_row] \
        if len(ribbons_df) else ribbons_df
    if len(row_ribbons) > 0:
        ax1.plot(row_ribbons["crest_x_m"], row_ribbons["crest_amplitude_max"],
                 "rv", ms=8, label="Crests")
        ax1.plot(row_ribbons["leading_edge_m"],
                 np.full(len(row_ribbons), chm_threshold),
                 "g|", ms=14, mew=2, label="Leading edge")
        ax1.plot(row_ribbons["trailing_edge_m"],
                 np.full(len(row_ribbons), chm_threshold),
                 "b|", ms=14, mew=2, label="Trailing edge")
    ax1.set_xlabel("East–West distance (m)")
    ax1.set_ylabel("Canopy height (m)")
    ax1.set_title(f"Example transect (row {example_row})")
    ax1.legend(fontsize=8, ncol=4, loc="upper right")

    # --- Panel 2: Ribbon width histogram ---
    ax2 = fig.add_subplot(gs[1, 0])
    if len(ribbons_df) > 0:
        ax2.hist(ribbons_df["width_m"], bins=30, color="seagreen",
                 edgecolor="white", alpha=0.85)
    ax2.axvline(min_ribbon_length_m, color="red", ls="--", lw=1.2,
                label=f"Min = {min_ribbon_length_m:.0f} m")
    ax2.set_xlabel("Ribbon width (m)")
    ax2.set_ylabel("Count")
    ax2.set_title("Ribbon width distribution")
    ax2.legend(fontsize=8)

    # --- Panel 3: Amplitude histogram ---
    ax3 = fig.add_subplot(gs[1, 1])
    if len(ribbons_df) > 0:
        ax3.hist(ribbons_df["crest_amplitude_max"], bins=30, color="coral",
                 edgecolor="white", alpha=0.85)
    ax3.set_xlabel("Max canopy height per ribbon (m)")
    ax3.set_ylabel("Count")
    ax3.set_title("Crest amplitude distribution")

    # --- Panel 4: H1 scatter — amplitude vs. pure glade width ---
    ax4 = fig.add_subplot(gs[1, 2])
    if len(pairs_df) > 0:
        valid = pairs_df.dropna(
            subset=["crest_amplitude_max", "pure_glade_width"]
        )
        valid = valid[valid["pure_glade_width"] > 0]
        if len(valid) > 0:
            sc = ax4.scatter(
                valid["pure_glade_width"], valid["crest_amplitude_max"],
                c=valid["y_m"], cmap="viridis", s=18, alpha=0.55,
                edgecolors="none",
            )
            fig.colorbar(sc, ax=ax4, label="N–S (m)", shrink=0.85)
            try:
                from scipy.stats import spearmanr
                rho, pval = spearmanr(
                    valid["pure_glade_width"], valid["crest_amplitude_max"]
                )
                ax4.text(
                    0.04, 0.96,
                    f"Spearman ρ = {rho:.3f}\np = {pval:.2e}\nN = {len(valid)}",
                    transform=ax4.transAxes, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                    fontsize=9,
                )
            except Exception:
                pass
    ax4.set_xlabel("Pure glade width (m)")
    ax4.set_ylabel("Crest amplitude (m)")
    ax4.set_title("H₁: amplitude vs. glade width")

    # --- Panel 5: H1 scatter — amplitude vs. crest_to_downwind_edge ---
    ax5 = fig.add_subplot(gs[2, 0])
    if len(pairs_df) > 0:
        valid = pairs_df.dropna(
            subset=["crest_amplitude_max", "crest_to_downwind_edge"]
        )
        valid = valid[valid["crest_to_downwind_edge"] > 0]
        if len(valid) > 0:
            sc = ax5.scatter(
                valid["crest_to_downwind_edge"], valid["crest_amplitude_max"],
                c=valid["y_m"], cmap="viridis", s=18, alpha=0.55,
                edgecolors="none",
            )
            fig.colorbar(sc, ax=ax5, label="N–S (m)", shrink=0.85)
            try:
                from scipy.stats import spearmanr
                rho, pval = spearmanr(
                    valid["crest_to_downwind_edge"],
                    valid["crest_amplitude_max"],
                )
                ax5.text(
                    0.04, 0.96,
                    f"Spearman ρ = {rho:.3f}\np = {pval:.2e}\nN = {len(valid)}",
                    transform=ax5.transAxes, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
                    fontsize=9,
                )
            except Exception:
                pass
    ax5.set_xlabel("Crest to downwind edge (m)")
    ax5.set_ylabel("Crest amplitude (m)")
    ax5.set_title("H₁: amplitude vs. crest-to-edge")

    # --- Panel 6: Crest-to-crest spacing ---
    ax6 = fig.add_subplot(gs[2, 1])
    if len(pairs_df) > 0:
        c2c = pairs_df["crest_to_crest"].dropna()
        c2c = c2c[c2c > 0]
        if len(c2c) > 0:
            ax6.hist(c2c, bins=30, color="steelblue",
                     edgecolor="white", alpha=0.85)
            med = float(np.median(c2c))
            ax6.axvline(med, color="black", ls="-", lw=1.2,
                        label=f"Median = {med:.1f} m")
    ax6.axvline(fft_peak_wavelength, color="red", ls="--", lw=1.5,
                label=f"FFT peak = {fft_peak_wavelength:.1f} m")
    ax6.set_xlabel("Crest-to-crest spacing (m)")
    ax6.set_ylabel("Count")
    ax6.set_title("Crest-to-crest spacing")
    ax6.legend(fontsize=8)

    # --- Panel 7: Spatial map — CHM with ribbon mask overlay ---
    ax7 = fig.add_subplot(gs[2, 2])
    vmax = np.nanpercentile(chm, 98) if np.any(np.isfinite(chm)) else 1.0
    ax7.imshow(
        chm, cmap="YlGn", vmin=0, vmax=vmax,
        extent=[0, ncols * dx, nrows * dx, 0],
    )
    # Red overlay where mask is True
    overlay = np.zeros((*mask.shape, 4), dtype=float)
    overlay[mask, 0] = 1.0      # R
    overlay[mask, 3] = 0.35     # alpha
    ax7.imshow(overlay, extent=[0, ncols * dx, nrows * dx, 0])
    ax7.set_xlabel("East–West (m)")
    ax7.set_ylabel("North–South (m)")
    ax7.set_title("CHM with ribbon mask (red)")

    fig.suptitle("CHM-based Ribbon Segmentation", fontsize=14, y=0.995)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig
