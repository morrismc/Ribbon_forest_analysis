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
DSM_PATH = Path(r"C:\Users\mmorriss\Desktop\Side_projects\Ribbon_forests\GIS\Rasters\Southern_field_site_snowies.tif")

# Output directory (relative to this file)
OUTPUT_DIR = Path(__file__).parent / "outputs" if "__file__" in globals() else Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Analysis options
# Set to None to analyse the full scene; set to an int (e.g. 512) to analyse
# a central square patch of that size in pixels. The full scene at 0.5 m/px
# is large — start with 1024 or 2048 to keep things snappy, then scale up.
WINDOW_SIZE = 1024

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

# %% Cell 3 — Detrend (remove best-fit plane) ---------------------------------
print("Detrending (least-squares plane removal)...")
detrended, plane, coeffs = detrend_dsm(dsm_filled)
print(f"  Plane: z = {coeffs[0]:.6f}*x + {coeffs[1]:.6f}*y + {coeffs[2]:.2f}")
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
