# Ribbon Forest Spectral Analysis

Spectral analysis of **ribbon forests** in the Snowy Range (Medicine Bow Mountains), Wyoming, using **2D discrete Fourier transforms (DFT)** on a LiDAR-derived digital surface model (DSM). The core question this tool is built to answer:

> **Is there a relationship between the height (amplitude) of forest ribbons and their spacing to adjacent ribbons?**

The method adapts Booth et al. (2009) — originally designed for automated landslide mapping — to detect quasi-periodic vegetation patterns instead of hummocky landslide topography, following the 2D FFT framework of Perron et al. (2008).

---

## Table of contents

1. [Scientific context](#scientific-context)
2. [Dependencies](#dependencies)
3. [Project structure](#project-structure)
4. [Workflow](#workflow)
5. [Input data](#input-data)
6. [Running the analysis](#running-the-analysis)
7. [Outputs](#outputs)
8. [Results figures](#results-figures)
9. [Key parameters](#key-parameters)
10. [References](#references)

---

## Scientific context

Ribbon forests are roughly linear strips of subalpine forest separated by treeless "snow glades." Several competing hypotheses explain their origin and maintenance, and each predicts a different **spectral signature**:

| Hypothesis | Mechanism | Spectral prediction |
|---|---|---|
| **Wind–snow feedback** (Billings, 1969) | Ribbons act as snow fences; drift depth/width controls spacing | Relatively uniform spacing → **sharp spectral peak** |
| **Geomorphic / lithologic control** (Butler et al., 2003) | Ribbons sit on bedrock ridges from glacial scouring; glades occupy troughs | Spacing reflects bedrock structure → peak may be broad or absent |
| **Hybrid topography + snow** (Calder et al., 2014; Bekker & Malanson) | Topographic rises provide establishment sites; wind–snow feedback maintains the pattern | Both components present — peak plus broader background |
| **Fire-mediated origin** (Buckner, 1977) | Crown fires clear large areas; new ribbons establish in drift zones | Spacing may reflect fire history — more irregular |

### What spectral analysis can tell us

- **Regular spacing** (strong narrow peak) → favours wind–snow feedback or regular geomorphic control
- **Variable spacing** (broad or multiple peaks) → favours geological heterogeneity or fire disturbance
- **2D directional anisotropy** → reveals ribbon orientation relative to wind / geology
- **Amplitude–spacing correlation** → novel contribution; could distinguish snow-depth-limited spacing from geomorphically fixed spacing

Hiemstra et al.'s semivariogram analysis at Libby Flats gave a snow-depth autocorrelation range of **~68 m**, which is a useful benchmark for the expected ribbon spacing.

---

## Dependencies

Core Python packages (see `requirements.txt`):

- `numpy` — array math, FFT
- `scipy` — Hann window, statistics, image filters
- `rasterio` — GeoTIFF I/O with CRS / transform preservation
- `matplotlib` — plotting

Install in one step:

```bash
pip install -r requirements.txt
```

Python 3.9+ is recommended. No compiled / GPU dependencies are required.

**Optional (for VSCode interactive workflow):**

- VSCode [Python extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- VSCode [Jupyter extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

Both are recommended automatically when you open the folder in VSCode (via `.vscode/extensions.json`).

---

## Project structure

```
Ribbon_forest_analysis/
├── README.md                   ← you are here
├── requirements.txt            ← Python dependencies
├── run_analysis.py             ← VSCode-friendly interactive runner (# %% cells)
├── .vscode/
│   ├── launch.json             ← F5 run configurations
│   ├── settings.json           ← workspace settings
│   └── extensions.json         ← recommended extensions
└── ribbon_fft/                 ← analysis package
    ├── __init__.py
    ├── load_dsm.py             ← GeoTIFF I/O, NoData handling
    ├── detrend.py              ← least-squares plane fitting & removal
    ├── spectral.py             ← 2D FFT, windowing, radial averaging
    ├── spatial_map.py          ← sliding-window spectral power mapping
    ├── analysis.py             ← amplitude extraction & spacing correlation
    ├── plotting.py             ← visualization functions
    └── main.py                 ← CLI entry point
```

### Module responsibilities

| Module | What it does |
|---|---|
| `load_dsm.py` | Reads a GeoTIFF into a masked array; fills NoData with nearest-neighbour interpolation; writes outputs back as GeoTIFFs with CRS preserved. |
| `detrend.py` | Fits and subtracts a least-squares plane `z = ax + by + c` to remove the regional topographic gradient, isolating local relief (the vegetation signal). Replicates the MATLAB `detrend_tp.m` / `lsplane.m` from Perron's `2DSpecTools`. |
| `spectral.py` | Applies a 2D Hann window, zero-pads to the next power of 2, computes the 2D power spectral density via `numpy.fft.fft2`, and radially averages it to get a 1D power-vs-frequency curve. Also finds the dominant frequency / wavelength. |
| `spatial_map.py` | Runs a sliding window across the detrended DSM, computing the spectral power integrated over a target frequency band at each window centre (à la Booth et al.'s `fft_powsum.m`). Produces 2D maps of local spectral power and dominant wavelength. |
| `analysis.py` | Extracts local peak-to-trough amplitude of the vegetation signal and computes the Pearson correlation between local amplitude and local dominant wavelength — the core question this tool is built to answer. |
| `plotting.py` | All visualization: hillshade, detrended DSM, 1D/2D power spectra (log-log), amplitude-vs-spacing scatter, and a 4-panel summary figure. |
| `main.py` | CLI entry point with Phase 1 (full-scene) and Phase 2 (sliding-window) modes. |

---

## Workflow

### Phase 1 — characteristic ribbon spacing (current scope)

```
  GeoTIFF DSM
      │
      ▼
  Load & mask NoData  ──────── load_dsm.load_dsm / fill_nodata_nearest
      │
      ▼
  Detrend (remove best-fit plane) ── detrend.detrend_dsm
      │
      ▼
  2D Hann window + zero-pad + FFT ── spectral.compute_2d_power_spectrum
      │
      ├──▶ 2D power spectrum (directional structure)
      │
      ▼
  Radial average ──────────── spectral.radial_average
      │
      ├──▶ 1D power spectrum (log-log plot)
      │
      ▼
  Find peak in 0.01–0.033 cycles/m band ── spectral.dominant_frequency
      │
      └──▶ Characteristic ribbon wavelength
```

### Phase 2 — spatial variation and amplitude–spacing correlation

```
  Detrended DSM
      │
      ├──▶ sliding window FFT ── spatial_map.sliding_window_power
      │       │
      │       ├──▶ 2D map of spectral power in ribbon band
      │       └──▶ 2D map of dominant wavelength
      │
      ├──▶ local amplitude ─── analysis.extract_amplitude
      │       │
      │       └──▶ 2D map of peak-to-trough amplitude
      │
      └──▶ correlate amplitude ↔ wavelength at matching points
              └──▶ Pearson r, p-value, scatter plot
```

### Future phases (not yet implemented)

- Directional filtering to isolate ribbons perpendicular to the prevailing wind
- 2D continuous wavelet transform (CWT) for scale-localised analysis
- Comparison with independent snow-depth / wind direction datasets

---

## Input data

- **File**: `Southern_field_site_snowies.tif`
- **Type**: Digital Surface Model (DSM) — **includes tree canopy**, not bare earth
- **Resolution**: 0.5 m × 0.5 m
- **Source**: LiDAR point cloud
- **Format**: GeoTIFF
- **Location**: Snowy Range, Wyoming (Medicine Bow Mountains)
- **Default path** (edit in `run_analysis.py`):
  ```
  C:\Users\mmorriss\Desktop\Side_projects\Ribbon_forests\GIS\Rasters\Southern_field_site_snowies.tif
  ```

The DSM (rather than a bare-earth DEM) is intentional: the vegetation signal **is** the ribbon forest, and the ~10–20 m peak-to-trough canopy relief is the thing we want to measure.

---

## Running the analysis

### Option A — VSCode (recommended)

1. Open the `Ribbon_forest_analysis` folder in VSCode.
2. Accept the recommended extensions when prompted (Python + Jupyter).
3. Install dependencies once:
   ```
   pip install -r requirements.txt
   ```
4. Open `run_analysis.py` and confirm `DSM_PATH` points to your file.
5. Either:
   - **Press F5** → runs the whole script (uses the "Run ribbon forest analysis" launch config).
   - **Click "Run Cell" / "Run All Cells"** above any `# %%` block → runs interactively; plots display inline in the Interactive Window; variables persist for inspection in the Variables pane.

The script is organised as six cells:

| Cell | Purpose |
|---|---|
| 1 | Imports, config (edit `DSM_PATH`, `WINDOW_SIZE`, `FREQ_BAND` here) |
| 2 | Load DSM, fill NoData, optionally crop to a central patch |
| 3 | Detrend (subtract best-fit plane) |
| 4 | Compute 2D FFT & radially averaged 1D power spectrum |
| 5 | Generate all Phase 1 plots |
| 6 | Optional Phase 2 — sliding-window spatial mapping (toggle `RUN_PHASE2 = True`) |

### Option B — Command line

```bash
# Phase 1 — full scene
python -m ribbon_fft.main path/to/Southern_field_site_snowies.tif -o outputs

# Phase 1 — faster, central 1024x1024 patch
python -m ribbon_fft.main path/to/dsm.tif --window-size 1024

# Phase 1 — specific patch at (row, col), size
python -m ribbon_fft.main path/to/dsm.tif --patch 2000 1500 1024

# Phase 1 + Phase 2 — full pipeline with sliding-window mapping
python -m ribbon_fft.main path/to/dsm.tif --spatial-map --sw-window 257 --step 64
```

---

## Outputs

All outputs are written to `outputs/` (configurable). Phase 1 produces:

| File | Description |
|---|---|
| `detrended_dsm.tif` | GeoTIFF of the detrended DSM (CRS & transform preserved) |
| `hillshade.png` | Hillshade of the analysed region — spatial reference |
| `detrended.png` | Detrended DSM with diverging colormap centred on zero |
| `power_spectrum_1d.png` | Log-log 1D radially averaged power spectrum with the expected ribbon band shaded |
| `power_spectrum_2d.png` | 2D power spectrum showing directional structure |
| `summary.png` | 4-panel summary figure (all of the above in one image) |

Phase 2 additionally produces:

| File | Description |
|---|---|
| `amplitude_vs_spacing.png` | Scatter plot of ribbon amplitude vs. dominant wavelength with Pearson r / p |
| `wavelength_map.png` | 2D map of locally dominant ribbon wavelength |

---

## Results figures

> Results figures will be added here once the analysis is run on the real DSM. Placeholders below indicate the intended layout — drop the generated PNGs into a `docs/figures/` directory and update the paths.

### Fig. 1 — Study area and DSM

<!-- ![Hillshade of the southern Snowy Range field site DSM](docs/figures/hillshade.png) -->

*Hillshade of the Snowy Range field site showing the ribbon forest pattern.*

### Fig. 2 — Detrended DSM

<!-- ![Detrended DSM](docs/figures/detrended.png) -->

*DSM after removal of the best-fit regional plane. Positive values (red) are local highs — forest ribbons. Negative values (blue) are glades.*

### Fig. 3 — 1D radially averaged power spectrum

<!-- ![1D power spectrum](docs/figures/power_spectrum_1d.png) -->

*Log-log plot of power vs. spatial frequency. The shaded green band marks the expected ribbon-spacing range (0.01–0.033 cycles/m, corresponding to 30–100 m wavelengths). Any spectral peak within this band is a candidate characteristic spacing.*

### Fig. 4 — 2D power spectrum

<!-- ![2D power spectrum](docs/figures/power_spectrum_2d.png) -->

*2D periodogram showing directional anisotropy. Strong elongation perpendicular to a wavevector indicates ribbons aligned along that direction — compare with known prevailing winter wind direction.*

### Fig. 5 — Phase 2: amplitude vs. spacing

<!-- ![Amplitude vs spacing](docs/figures/amplitude_vs_spacing.png) -->

*Scatter plot of local ribbon peak-to-trough amplitude vs. locally dominant wavelength, with Pearson correlation. A positive slope would support the hypothesis that taller ribbons have wider lee-side drifts and therefore larger spacings — the wind–snow feedback prediction.*

---

## Key parameters

| Parameter | Default | Notes |
|---|---|---|
| Grid spacing `dx` | 0.5 m | From DSM resolution |
| Nyquist frequency | 1.0 cycles/m | = 1/(2·dx) |
| Expected ribbon spacing | 30–100 m | Based on Hiemstra et al. (~68 m) and Billings observations |
| Target frequency band | 0.01–0.033 cycles/m | = 1/spacing |
| Phase 1 window size | full scene or user-chosen patch | Powers of 2 are FFT-efficient (e.g. 1024, 2048) |
| Phase 2 sliding window | 257 px (~128 m) | Must capture ≥2–3 ribbon wavelengths |
| Phase 2 step | 64 px (~32 m) | Quarter-window stride is a reasonable default |
| Detrending | least-squares plane | Linear trend assumed |

All of these can be edited at the top of `run_analysis.py` or passed as CLI arguments to `ribbon_fft.main`.

---

## References

1. **Booth, A.M., Roering, J.J., & Perron, J.T. (2009).** Automated landslide mapping using spectral analysis and high-resolution topographic data. *Geomorphology*, 109, 132–147.
2. **Perron, J.T., Kirchner, J.W., & Dietrich, W.E. (2008).** Spectral signatures of characteristic spatial scales and nonfractal structure in landscapes. *JGR: Earth Surface*, 113, F04003.
3. **Billings, W.D. (1969).** Vegetational pattern near alpine timberline as affected by fire–snowdrift interactions. *Vegetatio*, 19, 192–207.
4. **Butler, D.R., Malanson, G.P., Bekker, M.F., & Resler, L.M. (2003).** Lithologic, structural, and geomorphic controls on ribbon forest patterns. *Geomorphology*, 55, 203–217.
5. **Buckner, D.L. (1977).** Ribbon forest development and maintenance in the central Rocky Mountains of Colorado. *PhD dissertation*, University of Colorado.
6. **Calder, W.J., Horn, K.J., & Shuman, B.N. (2014).** High-elevation fire regimes in subalpine ribbon forests. *Rocky Mountain Geology*, 49(1), 75–90.
7. **Hiemstra, C.A., Liston, G.E., & Reiners, W.A.** Snow redistribution by wind and interactions with vegetation at upper treeline in the Medicine Bow Mountains, Wyoming.
8. **Purinton, B. & Bookhagen, B. (2017).** Validation of digital elevation models (DEMs) and comparison of geomorphic metrics on the southern Central Andean Plateau. *Earth Surface Dynamics*, 5, 211–237.

### Reference implementations

- [bpurinton/DEM-FFT](https://github.com/bpurinton/DEM-FFT) — Python port of Perron's MATLAB `2DSpecTools` (used as the reference for FFT convention choices here).
- Perron's `2DSpecTools` and Booth et al.'s `ALMtools` — the original MATLAB packages.
