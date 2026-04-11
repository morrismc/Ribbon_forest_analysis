#!/usr/bin/env python3
"""CLI entry point for ribbon forest spectral analysis.

Usage
-----
Phase 1 (full-scene analysis):
    python -m ribbon_fft.main <dsm.tif> [--output-dir outputs] [--window-size 512]

Phase 2 (sliding-window spatial mapping — future):
    python -m ribbon_fft.main <dsm.tif> --spatial-map [--step 64] [--freq-band 0.01 0.033]
"""

import argparse
import os
import sys

import numpy as np

from .load_dsm import load_dsm, fill_nodata_nearest, save_geotiff
from .detrend import detrend_dsm
from .spectral import compute_2d_power_spectrum, radial_average, dominant_frequency
from .plotting import (
    plot_hillshade,
    plot_detrended,
    plot_power_spectrum_1d,
    plot_power_spectrum_2d,
    create_summary_figure,
)


def run_phase1(dsm_path, output_dir="outputs", window_size=None, patch=None):
    """Run Phase 1 analysis: load, detrend, full-scene FFT, plot spectra.

    Parameters
    ----------
    dsm_path : str
        Path to input GeoTIFF DSM.
    output_dir : str
        Directory for output files.
    window_size : int, optional
        If provided, analyse a central patch of this size instead of the full scene.
    patch : tuple of int, optional
        (row_start, col_start, size) to analyse a specific patch.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Load ---
    print(f"Loading DSM: {dsm_path}")
    dsm, transform, crs, dx = load_dsm(dsm_path)
    print(f"  Shape: {dsm.shape}, Resolution: {dx} m, CRS: {crs}")

    # Fill NoData for analysis
    print("Filling NoData pixels (nearest-neighbour)...")
    dsm_filled = fill_nodata_nearest(dsm)

    # --- Optional patch extraction ---
    if patch is not None:
        r0, c0, sz = patch
        print(f"  Extracting patch at ({r0}, {c0}), size {sz}x{sz}")
        dsm_filled = dsm_filled[r0:r0 + sz, c0:c0 + sz]
        # Adjust transform for the patch
        from rasterio.transform import Affine
        transform = transform * Affine.translation(c0, r0)
    elif window_size is not None:
        # Centre patch
        nr, nc = dsm_filled.shape
        r0 = (nr - window_size) // 2
        c0 = (nc - window_size) // 2
        print(f"  Extracting central patch: size {window_size}x{window_size}")
        dsm_filled = dsm_filled[r0:r0 + window_size, c0:c0 + window_size]
        from rasterio.transform import Affine
        transform = transform * Affine.translation(c0, r0)

    print(f"  Analysis region: {dsm_filled.shape}")

    # --- Detrend ---
    print("Detrending (removing best-fit plane)...")
    detrended, plane, coeffs = detrend_dsm(dsm_filled)
    print(f"  Plane coefficients: a={coeffs[0]:.6f}, b={coeffs[1]:.6f}, c={coeffs[2]:.2f}")

    # Save detrended DSM as GeoTIFF
    save_geotiff(
        os.path.join(output_dir, "detrended_dsm.tif"),
        detrended.astype(np.float32),
        transform, crs,
    )

    # --- 2D FFT ---
    print("Computing 2D power spectrum...")
    power_2d, freq_x, freq_y = compute_2d_power_spectrum(
        detrended, dx, apply_window=True, zero_pad=True
    )

    # --- Radial average ---
    print("Radially averaging power spectrum...")
    freq_r, power_r = radial_average(power_2d, freq_x, freq_y)

    # --- Find dominant frequency ---
    # Search in the expected ribbon-spacing band (30-100 m → 0.01-0.033 cycles/m)
    peak_f, peak_wl, peak_p = dominant_frequency(freq_r, power_r, freq_range=(0.01, 0.033))
    print(f"  Dominant frequency in ribbon band: {peak_f:.4f} cycles/m")
    print(f"  Dominant wavelength: {peak_wl:.1f} m")

    # Also find global peak (outside DC)
    peak_f_global, peak_wl_global, _ = dominant_frequency(freq_r, power_r)
    print(f"  Global spectral peak: {peak_wl_global:.1f} m ({peak_f_global:.4f} cycles/m)")

    # --- Plots ---
    print("Generating plots...")

    # Individual plots
    fig, _ = plot_hillshade(dsm_filled, dx=dx, transform=transform)
    fig.savefig(os.path.join(output_dir, "hillshade.png"), dpi=150, bbox_inches="tight")

    fig, _ = plot_detrended(detrended, transform=transform)
    fig.savefig(os.path.join(output_dir, "detrended.png"), dpi=150, bbox_inches="tight")

    fig, _ = plot_power_spectrum_1d(freq_r, power_r, dx=dx)
    fig.savefig(os.path.join(output_dir, "power_spectrum_1d.png"), dpi=150, bbox_inches="tight")

    fig, _ = plot_power_spectrum_2d(power_2d, freq_x, freq_y)
    fig.savefig(os.path.join(output_dir, "power_spectrum_2d.png"), dpi=150, bbox_inches="tight")

    # Summary figure
    create_summary_figure(
        dsm_filled, detrended, power_2d, freq_x, freq_y,
        freq_r, power_r, dx, transform=transform,
        save_path=os.path.join(output_dir, "summary.png"),
    )

    import matplotlib.pyplot as plt
    plt.close("all")

    print(f"\nPhase 1 complete. Outputs saved to: {output_dir}/")
    print("Files:")
    for f in sorted(os.listdir(output_dir)):
        print(f"  {f}")

    return {
        "detrended": detrended,
        "power_2d": power_2d,
        "freq_x": freq_x,
        "freq_y": freq_y,
        "freq_r": freq_r,
        "power_r": power_r,
        "peak_wavelength": peak_wl,
        "peak_frequency": peak_f,
        "dx": dx,
        "transform": transform,
        "crs": crs,
    }


def run_phase2(dsm_path, output_dir="outputs", window_size=257, step=64,
               freq_band=(0.01, 0.033)):
    """Run Phase 2: sliding-window spectral mapping and amplitude analysis.

    Parameters
    ----------
    dsm_path : str
        Path to input GeoTIFF DSM.
    output_dir : str
        Output directory.
    window_size : int
        Sliding window size (pixels).
    step : int
        Step between windows (pixels).
    freq_band : tuple
        (f_min, f_max) frequency band.
    """
    from .spatial_map import sliding_window_power
    from .analysis import extract_amplitude, amplitude_spacing_correlation
    from .plotting import plot_amplitude_vs_spacing

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading DSM: {dsm_path}")
    dsm, transform, crs, dx = load_dsm(dsm_path)
    dsm_filled = fill_nodata_nearest(dsm)

    print("Detrending...")
    detrended, _, _ = detrend_dsm(dsm_filled)

    print(f"Sliding-window spectral analysis (window={window_size}, step={step})...")
    print(f"  Frequency band: {freq_band[0]:.4f} - {freq_band[1]:.4f} cycles/m")
    print(f"  Wavelength band: {1/freq_band[1]:.0f} - {1/freq_band[0]:.0f} m")

    power_map, wavelength_map, row_centres, col_centres = sliding_window_power(
        detrended, dx, window_size=window_size, step=step, freq_band=freq_band
    )

    print("Extracting ribbon amplitudes...")
    amplitude_map = extract_amplitude(detrended)

    print("Computing amplitude-spacing correlation...")
    amps, wls, r_val, p_val = amplitude_spacing_correlation(
        amplitude_map, wavelength_map, row_centres, col_centres, detrended.shape
    )

    print(f"  Pearson r = {r_val:.4f}, p = {p_val:.2e}")
    print(f"  N samples = {len(amps)}")

    # Plot
    fig, _ = plot_amplitude_vs_spacing(amps, wls, r_val, p_val)
    fig.savefig(os.path.join(output_dir, "amplitude_vs_spacing.png"),
                dpi=150, bbox_inches="tight")

    import matplotlib.pyplot as plt
    plt.close("all")

    print(f"\nPhase 2 complete. Outputs saved to: {output_dir}/")

    return {
        "power_map": power_map,
        "wavelength_map": wavelength_map,
        "amplitude_map": amplitude_map,
        "amplitudes": amps,
        "wavelengths": wls,
        "r_value": r_val,
        "p_value": p_val,
    }


def main():
    # Force non-interactive backend only when run as a CLI script
    import matplotlib
    matplotlib.use("Agg")

    parser = argparse.ArgumentParser(
        description="Ribbon forest spectral analysis using 2D DFT of LiDAR DSM.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Phase 1: full-scene spectral analysis
  python -m ribbon_fft.main data/Southern_field_site_snowies.tif

  # Phase 1 on a 512x512 central patch (faster)
  python -m ribbon_fft.main data/Southern_field_site_snowies.tif --window-size 512

  # Phase 1 on a specific patch
  python -m ribbon_fft.main data/Southern_field_site_snowies.tif --patch 1000 500 512

  # Phase 2: sliding-window mapping + amplitude analysis
  python -m ribbon_fft.main data/Southern_field_site_snowies.tif --spatial-map
        """,
    )
    parser.add_argument("dsm", help="Path to input GeoTIFF DSM")
    parser.add_argument("--output-dir", "-o", default="outputs",
                        help="Output directory (default: outputs)")
    parser.add_argument("--window-size", "-w", type=int, default=None,
                        help="Analyse a central patch of this size (pixels)")
    parser.add_argument("--patch", nargs=3, type=int, metavar=("ROW", "COL", "SIZE"),
                        help="Analyse a specific patch: row_start col_start size")

    # Phase 2 options
    parser.add_argument("--spatial-map", action="store_true",
                        help="Run Phase 2 sliding-window spatial mapping")
    parser.add_argument("--step", type=int, default=64,
                        help="Sliding window step size in pixels (default: 64)")
    parser.add_argument("--sw-window", type=int, default=257,
                        help="Sliding window size for spatial mapping (default: 257)")
    parser.add_argument("--freq-band", nargs=2, type=float, default=[0.01, 0.033],
                        metavar=("FMIN", "FMAX"),
                        help="Frequency band of interest in cycles/m (default: 0.01 0.033)")

    args = parser.parse_args()

    # Phase 1 always runs
    patch_arg = tuple(args.patch) if args.patch else None
    results = run_phase1(
        args.dsm,
        output_dir=args.output_dir,
        window_size=args.window_size,
        patch=patch_arg,
    )

    # Phase 2 if requested
    if args.spatial_map:
        print("\n" + "=" * 60)
        print("Phase 2: Sliding-window spectral mapping")
        print("=" * 60 + "\n")
        run_phase2(
            args.dsm,
            output_dir=args.output_dir,
            window_size=args.sw_window,
            step=args.step,
            freq_band=tuple(args.freq_band),
        )


if __name__ == "__main__":
    main()
