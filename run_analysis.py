"""Ribbon forest spectral analysis — interactive runner for VSCode.

HOW TO RUN THIS FILE IN VSCODE
------------------------------
1. Install the Python extension (Microsoft) and open this folder in VSCode.
2. Install dependencies once:
       pip install -r requirements.txt
3. Set DSM_PATH below to point at your GeoTIFF.
4. EITHER:
   - Press F5 to run the whole script (uses the launch config in .vscode/launch.json), OR
   - Click "Run Cell" above any `# %%` cell to run it interactively — plots
     will open in the VSCode Interactive Window where you can zoom, pan, and
     inspect variables. Click "Run All Cells" to do the whole pipeline.

CELLS
-----
Each `# %%` block below is a cell. In order:
    1. Setup and configuration
    2. Load DSM
    3. Detrend
    4. 2D FFT and power spectra
    5. Plots
    6. (optional) Phase 2 — sliding-window spatial mapping
"""

# %% [markdown]
# # Ribbon Forest Spectral Analysis
# Snowy Range (Medicine Bow Mountains), Wyoming.
# Adapts Booth et al. (2009) / Perron et al. (2008) spectral methods to
# detect quasi-periodic vegetation patterns in a LiDAR DSM.

# %% Cell 1 — Setup and configuration -----------------------------------------
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from ribbon_fft.load_dsm import load_dsm, fill_nodata_nearest, save_geotiff
from ribbon_fft.detrend import detrend_dsm
from ribbon_fft.spectral import (
    compute_2d_power_spectrum,
    radial_average,
    dominant_frequency,
)
from ribbon_fft.plotting import (
    plot_hillshade,
    plot_detrended,
    plot_power_spectrum_1d,
    plot_power_spectrum_2d,
    create_summary_figure,
)

# ---- EDIT THIS PATH TO POINT AT YOUR DSM ------------------------------------
# Windows example (use a raw string with r"..." OR forward slashes):
# DSM_PATH = Path(r"C:\Users\mmorriss\Desktop\Side_projects\Ribbon_forests\GIS\Rasters\Southern_field_site_snowies.tif")
DSM_PATH = Path(r"C:\Users\mmorriss\Desktop\Side_projects\Ribbon_forests\GIS\Rasters\Southern_field_site_snowies_DSM_04-15-2026.tif")

# ---- DEM (bare earth, for canopy height model) --------------------------------
# Set to None if you don't have a DEM yet; the CHM cell will be skipped.
DEM_PATH = Path(r"C:\Users\mmorriss\Desktop\Side_projects\Ribbon_forests\GIS\Rasters\Southern_site_dem.tif")

# Output directory (relative to this file; falls back to cwd in interactive mode)
try:
    OUTPUT_DIR = Path(__file__).parent / "outputs"
except NameError:
    OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Analysis options
# Set to None to analyse the full scene; set to an int (e.g. 512) to analyse
# a central square patch of that size in pixels. The full scene at 0.5 m/px
# is large — start with 1024 or 2048 to keep things snappy, then scale up.
WINDOW_SIZE = 1024

# Detrending order: 1 = planar, 2 = quadratic, 3 = cubic.
# Use order >= 2 when the study area spans significant hillslope curvature.
DETREND_ORDER = 2

# Expected ribbon spacing band: 30–100 m → 0.01–0.033 cycles/m
FREQ_BAND = (0.01, 0.033)

print(f"DSM path   : {DSM_PATH}")
print(f"Exists     : {DSM_PATH.exists()}")
print(f"Window     : {WINDOW_SIZE} px (None = full scene)")
print(f"Output dir : {OUTPUT_DIR}")

if not DSM_PATH.exists():
    raise FileNotFoundError(
        f"DSM not found at {DSM_PATH}. Edit DSM_PATH at the top of this file."
    )

# %% Cell 2 — Load DSM --------------------------------------------------------
print("Loading DSM...")
dsm, transform, crs, dx = load_dsm(str(DSM_PATH))
print(f"  Shape     : {dsm.shape}")
print(f"  Resolution: {dx} m/pixel")
print(f"  CRS       : {crs}")
print(f"  Masked px : {dsm.mask.sum() if np.ma.is_masked(dsm) else 0}")

# Fill NoData
dsm_filled = fill_nodata_nearest(dsm)

# Optionally crop to a central patch for speed
if WINDOW_SIZE is not None:
    nr, nc = dsm_filled.shape
    r0 = (nr - WINDOW_SIZE) // 2
    c0 = (nc - WINDOW_SIZE) // 2
    dsm_filled = dsm_filled[r0:r0 + WINDOW_SIZE, c0:c0 + WINDOW_SIZE]
    from rasterio.transform import Affine
    transform = transform * Affine.translation(c0, r0)
    print(f"  Cropped to: {dsm_filled.shape} (central patch)")

# %% Cell 3 — Detrend (remove best-fit surface) -------------------------------
print(f"Detrending (order={DETREND_ORDER} polynomial surface removal)...")
detrended, surface, coeffs = detrend_dsm(dsm_filled, order=DETREND_ORDER)
print(f"  Detrended range: {np.nanmin(detrended):.2f} to {np.nanmax(detrended):.2f} m")

# Save detrended DSM (keeps CRS/transform)
save_geotiff(
    str(OUTPUT_DIR / "detrended_dsm.tif"),
    detrended.astype(np.float32),
    transform,
    crs,
)
print(f"  Saved: {OUTPUT_DIR / 'detrended_dsm.tif'}")

# %% Cell 4 — 2D FFT and radially averaged power spectrum ---------------------
print("Computing 2D power spectrum...")
power_2d, freq_x, freq_y = compute_2d_power_spectrum(
    detrended, dx, apply_window=True, zero_pad=True
)

print("Radially averaging...")
freq_r, power_r = radial_average(power_2d, freq_x, freq_y)

# Find dominant frequency in the expected ribbon-spacing band
peak_f, peak_wl, peak_p = dominant_frequency(freq_r, power_r, freq_range=FREQ_BAND)
print(f"  Peak in ribbon band : f = {peak_f:.4f} cycles/m, lambda = {peak_wl:.1f} m")

# Global peak (outside DC)
peak_f_g, peak_wl_g, _ = dominant_frequency(freq_r, power_r)
print(f"  Global spectral peak: f = {peak_f_g:.4f} cycles/m, lambda = {peak_wl_g:.1f} m")

# %% Cell 5 — Plots (these show inline in the VSCode Interactive Window) ------
print("Plotting...")

# 1. Hillshade of the DSM region analysed
fig, _ = plot_hillshade(dsm_filled, dx=dx, transform=transform)
fig.savefig(OUTPUT_DIR / "hillshade.png", dpi=150, bbox_inches="tight")
plt.show()

# 2. Detrended DSM (diverging colormap centred on zero)
fig, _ = plot_detrended(detrended, transform=transform)
fig.savefig(OUTPUT_DIR / "detrended.png", dpi=150, bbox_inches="tight")
plt.show()

# 3. 2D power spectrum (shows directional structure)
fig, _ = plot_power_spectrum_2d(power_2d, freq_x, freq_y)
fig.savefig(OUTPUT_DIR / "power_spectrum_2d.png", dpi=150, bbox_inches="tight")
plt.show()

# 4. 1D radially averaged power spectrum
fig, _ = plot_power_spectrum_1d(freq_r, power_r, dx=dx)
fig.savefig(OUTPUT_DIR / "power_spectrum_1d.png", dpi=150, bbox_inches="tight")
plt.show()

# 5. 4-panel summary
summary_fig = create_summary_figure(
    dsm_filled, detrended, power_2d, freq_x, freq_y,
    freq_r, power_r, dx, transform=transform,
    save_path=str(OUTPUT_DIR / "summary.png"),
)
plt.show()

print(f"\nPhase 1 complete. Outputs in: {OUTPUT_DIR}")

# %% Cell 6 — (Optional) Phase 2: sliding-window spatial mapping --------------
# This is slower — it runs a 2D FFT in a sliding window across the whole
# cropped DSM. Start with a modest region or a large step for speed.
RUN_PHASE2 = False

if RUN_PHASE2:
    from ribbon_fft.spatial_map import sliding_window_power
    from ribbon_fft.analysis import extract_amplitude, amplitude_spacing_correlation
    from ribbon_fft.plotting import plot_amplitude_vs_spacing

    print("Phase 2: sliding-window spectral mapping...")
    SW_WINDOW = 257   # pixels
    SW_STEP   = 128   # pixels
    print(f"  window = {SW_WINDOW} px, step = {SW_STEP} px")

    power_map, wavelength_map, row_c, col_c = sliding_window_power(
        detrended, dx,
        window_size=SW_WINDOW,
        step=SW_STEP,
        freq_band=FREQ_BAND,
    )

    amplitude_map = extract_amplitude(detrended, kernel_size=51)

    amps, wls, r_val, p_val = amplitude_spacing_correlation(
        amplitude_map, wavelength_map, row_c, col_c, detrended.shape
    )
    print(f"  Pearson r = {r_val:.4f}, p = {p_val:.2e}, N = {len(amps)}")

    fig, _ = plot_amplitude_vs_spacing(amps, wls, r_val, p_val)
    fig.savefig(OUTPUT_DIR / "amplitude_vs_spacing.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Dominant wavelength map
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(wavelength_map, cmap="viridis")
    fig.colorbar(im, ax=ax, label="Dominant wavelength (m)")
    ax.set_title("Local dominant ribbon spacing")
    fig.savefig(OUTPUT_DIR / "wavelength_map.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Phase 2 complete. Outputs in: {OUTPUT_DIR}")

# %% Cell 7 — 2D Continuous Wavelet Transform (amplitude–spacing analysis) -----
# The CWT localises both scale (wavelength) and position simultaneously,
# giving per-pixel dominant wavelength and amplitude — much finer than the
# sliding-window FFT.  This is the preferred method for the amplitude-spacing
# correlation question.
RUN_CWT = True

if RUN_CWT:
    from ribbon_fft.wavelet import cwt_2d, cwt_dominant_scale, cwt_amplitude_spacing
    from ribbon_fft.plotting import plot_amplitude_vs_spacing

    # Wavelet scales to probe (in metres).  Logarithmically spaced from
    # 5 m to 40 m in scale, which maps to ~20–160 m in wavelength via the
    # Mexican Hat's scale-to-wavelength factor (~4x).
    CWT_SCALES = np.geomspace(5.0, 40.0, 25)

    print("2D CWT analysis...")
    print(f"  Scales: {CWT_SCALES[0]:.1f} – {CWT_SCALES[-1]:.1f} m "
          f"({len(CWT_SCALES)} scales)")

    coefficients, scales_used = cwt_2d(detrended, dx, scales_m=CWT_SCALES)
    dom_wavelength, cwt_amplitude, _ = cwt_dominant_scale(coefficients, scales_used)

    print(f"  Dominant wavelength range: "
          f"{np.nanmin(dom_wavelength):.1f} – {np.nanmax(dom_wavelength):.1f} m")
    print(f"  Median dominant wavelength: {np.nanmedian(dom_wavelength):.1f} m")

    # Amplitude vs. spacing correlation (CWT-based)
    wl_s, amp_s, r_cwt, p_cwt = cwt_amplitude_spacing(
        dom_wavelength, cwt_amplitude, wavelength_range=(20, 120)
    )
    print(f"  CWT Pearson r = {r_cwt:.4f}, p = {p_cwt:.2e}, N = {len(wl_s)}")

    # --- Plots ---
    # Dominant wavelength map
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(dom_wavelength, cmap="viridis", vmin=20, vmax=120)
    fig.colorbar(im, ax=ax, label="Dominant wavelength (m)")
    ax.set_title("CWT Dominant Ribbon Wavelength")
    fig.savefig(OUTPUT_DIR / "cwt_wavelength_map.png", dpi=150, bbox_inches="tight")
    plt.show()

    # CWT amplitude map
    fig, ax = plt.subplots(figsize=(10, 8))
    vmax = np.nanpercentile(cwt_amplitude, 98)
    im = ax.imshow(cwt_amplitude, cmap="magma", vmin=0, vmax=vmax)
    fig.colorbar(im, ax=ax, label="Wavelet amplitude")
    ax.set_title("CWT Ribbon Amplitude")
    fig.savefig(OUTPUT_DIR / "cwt_amplitude_map.png", dpi=150, bbox_inches="tight")
    plt.show()

    # CWT-based amplitude vs. spacing scatter
    fig, _ = plot_amplitude_vs_spacing(amp_s, wl_s, r_cwt, p_cwt)
    ax = fig.axes[0]
    ax.set_xlabel("Dominant wavelength (m)")
    ax.set_ylabel("Wavelet amplitude")
    ax.set_title("CWT: Ribbon Amplitude vs. Spacing")
    fig.savefig(OUTPUT_DIR / "cwt_amplitude_vs_spacing.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"CWT analysis complete. Outputs in: {OUTPUT_DIR}")

# %% Cell 8 — Transect peak/trough analysis -----------------------------------
# 100 random east-west transects, Gaussian-smoothed, with peak/trough detection.
# This tests H1: do taller ribbons have wider downwind glades?
RUN_TRANSECTS = True

if RUN_TRANSECTS:
    from ribbon_fft.transect import run_transect_analysis
    from ribbon_fft.plotting import plot_transect_summary

    print("Transect peak/trough analysis...")
    SIGMA_PIXELS = 5       # Gaussian sigma in pixels (~2.5 m smoothing)
    MIN_DIST_PX  = 80      # minimum peak-to-peak distance (~40 m)
    MIN_PROM     = 1.0     # minimum peak prominence (m)
    N_TRANSECTS  = 100

    df_transects, transect_rows = run_transect_analysis(
        detrended, dx=dx,
        n_transects=N_TRANSECTS,
        sigma_pixels=SIGMA_PIXELS,
        min_distance_pixels=MIN_DIST_PX,
        min_prominence=MIN_PROM,
    )
    print(f"  Detected {len(df_transects)} crest-trough pairs across "
          f"{len(transect_rows)} transects")

    if len(df_transects) > 0:
        print(f"  Median amplitude: {df_transects['amplitude_downwind'].median():.1f} m")
        print(f"  Median crest-to-trough: {df_transects['crest_to_trough_m'].median():.1f} m")
        print(f"  Median crest-to-crest: {df_transects['crest_to_crest_m'].median():.1f} m")

        # Correlation
        from scipy.stats import pearsonr
        valid = df_transects.dropna(subset=["amplitude_downwind", "crest_to_trough_m"])
        if len(valid) > 2:
            r, p = pearsonr(valid["crest_to_trough_m"], valid["amplitude_downwind"])
            print(f"  Pearson r (amplitude vs trough dist) = {r:.4f}, p = {p:.2e}")

        # 6-panel summary figure
        fft_peak = peak_wl if "peak_wl" in dir() else 65.9
        fig = plot_transect_summary(
            detrended, df_transects, transect_rows, dx=dx,
            fft_peak_wavelength=fft_peak,
            save_path=str(OUTPUT_DIR / "transect_analysis.png"),
        )
        plt.show()

        # Save CSV
        csv_path = OUTPUT_DIR / "ribbon_crest_trough_measurements.csv"
        df_transects.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

    print(f"Transect analysis complete. Outputs in: {OUTPUT_DIR}")

# %% Cell 9 — Canopy Height Model (CHM = DSM - DEM) ---------------------------
# Requires the DEM file. Set DEM_PATH at the top of this script.
RUN_CHM = True

if RUN_CHM and DEM_PATH is not None and DEM_PATH.exists():
    from ribbon_fft.load_dsm import load_and_align_dem, compute_chm

    print("Computing Canopy Height Model...")
    print(f"  DEM path: {DEM_PATH}")

    # Load DEM and resample to match DSM grid
    dem_aligned = load_and_align_dem(str(DEM_PATH), str(DSM_PATH))
    print(f"  DEM shape (aligned): {dem_aligned.shape}")
    print(f"  DSM shape: {dsm.shape}")

    # If we cropped the DSM earlier, crop the DEM to match
    dsm_full = fill_nodata_nearest(dsm)
    if WINDOW_SIZE is not None:
        nr_full, nc_full = dsm_full.shape
        r0 = (nr_full - WINDOW_SIZE) // 2
        c0 = (nc_full - WINDOW_SIZE) // 2
        dem_crop = dem_aligned[r0:r0 + WINDOW_SIZE, c0:c0 + WINDOW_SIZE]
        dsm_crop = dsm_full[r0:r0 + WINDOW_SIZE, c0:c0 + WINDOW_SIZE]
    else:
        dem_crop = dem_aligned
        dsm_crop = dsm_full

    chm = compute_chm(dsm_crop, dem_crop)
    print(f"  CHM range: {np.nanmin(chm):.1f} – {np.nanmax(chm):.1f} m")
    print(f"  Median canopy height: {np.nanmedian(chm):.1f} m")

    # Save CHM as GeoTIFF
    save_geotiff(
        str(OUTPUT_DIR / "canopy_height_model.tif"),
        chm.astype(np.float32),
        transform, crs,
    )

    # Plot CHM
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(chm, cmap="YlGn", vmin=0, vmax=np.nanpercentile(chm, 98))
    fig.colorbar(im, ax=ax, label="Canopy height (m)")
    ax.set_title("Canopy Height Model (DSM \u2013 DEM)")
    fig.savefig(OUTPUT_DIR / "canopy_height_model.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"CHM complete. Outputs in: {OUTPUT_DIR}")
elif RUN_CHM:
    print(f"Skipping CHM: DEM not found at {DEM_PATH}")
