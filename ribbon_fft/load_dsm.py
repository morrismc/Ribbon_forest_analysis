"""GeoTIFF I/O, NoData handling, and basic DSM utilities."""

import numpy as np
import rasterio


def load_dsm(filepath):
    """Load a GeoTIFF DSM and return the array, transform, CRS, and nodata value.

    Parameters
    ----------
    filepath : str or Path
        Path to the GeoTIFF file.

    Returns
    -------
    dsm : numpy.ma.MaskedArray
        2-D masked array with NoData pixels masked.
    transform : rasterio.Affine
        Affine geotransform.
    crs : rasterio.crs.CRS
        Coordinate reference system.
    resolution : float
        Grid spacing in map units (assumes square pixels).
    """
    with rasterio.open(filepath) as src:
        dsm = src.read(1)
        nodata = src.nodata
        transform = src.transform
        crs = src.crs
        resolution = src.res[0]  # assumes square pixels

    # Build mask for nodata and non-finite values
    mask = ~np.isfinite(dsm)
    if nodata is not None:
        mask |= dsm == nodata

    dsm = np.ma.MaskedArray(dsm, mask=mask)

    return dsm, transform, crs, resolution


def extract_patch(dsm, row, col, size):
    """Extract a square patch from the DSM centred on (row, col).

    Parameters
    ----------
    dsm : numpy.ndarray or numpy.ma.MaskedArray
        2-D elevation array.
    row, col : int
        Centre pixel indices.
    size : int
        Side length of the square patch (pixels).

    Returns
    -------
    patch : numpy.ndarray
        Extracted patch (may contain masked values).
    """
    half = size // 2
    r0 = max(row - half, 0)
    r1 = min(row + half + 1, dsm.shape[0])
    c0 = max(col - half, 0)
    c1 = min(col + half + 1, dsm.shape[1])
    return dsm[r0:r1, c0:c1]


def fill_nodata_nearest(dsm):
    """Fill masked (NoData) pixels with nearest-neighbour interpolation.

    Parameters
    ----------
    dsm : numpy.ma.MaskedArray
        DSM with masked NoData pixels.

    Returns
    -------
    filled : numpy.ndarray
        DSM with NoData pixels filled.
    """
    from scipy.ndimage import distance_transform_edt

    data = dsm.data.copy()

    # dsm.mask can be np.ma.nomask (a scalar False) when no pixels are masked
    if not isinstance(dsm.mask, np.ndarray) or not dsm.mask.any():
        return data
    mask = dsm.mask

    # distance_transform_edt returns indices of nearest valid pixel
    _, nearest_idx = distance_transform_edt(mask, return_distances=True, return_indices=True)
    filled = data[tuple(nearest_idx)]
    return filled


def save_geotiff(filepath, data, transform, crs, nodata=None):
    """Write a 2-D array to a GeoTIFF preserving geospatial metadata.

    Parameters
    ----------
    filepath : str or Path
        Output path.
    data : numpy.ndarray
        2-D array to write.
    transform : rasterio.Affine
        Affine geotransform.
    crs : rasterio.crs.CRS
        Coordinate reference system.
    nodata : float, optional
        NoData value to embed in the file.
    """
    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)


def load_and_align_dem(dem_path, dsm_path):
    """Load a DEM and resample/align it to match a DSM's grid.

    The DEM may have different resolution and extent than the DSM.
    This function reprojects/resamples the DEM to match the DSM's
    transform, CRS, and shape exactly.

    Parameters
    ----------
    dem_path : str or Path
        Path to the DEM GeoTIFF (bare earth).
    dsm_path : str or Path
        Path to the DSM GeoTIFF (with canopy) — used as the target grid.

    Returns
    -------
    dem_aligned : numpy.ndarray
        DEM resampled to match the DSM grid.
    """
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    with rasterio.open(dsm_path) as dsm_src:
        dst_transform = dsm_src.transform
        dst_crs = dsm_src.crs
        dst_shape = (dsm_src.height, dsm_src.width)

    with rasterio.open(dem_path) as dem_src:
        dem_data = dem_src.read(1).astype(np.float64)
        dem_nodata = dem_src.nodata

        # Replace nodata with NaN before reprojection
        if dem_nodata is not None:
            dem_data[dem_data == dem_nodata] = np.nan
        dem_data[~np.isfinite(dem_data)] = np.nan

        # Allocate output array
        dem_aligned = np.full(dst_shape, np.nan, dtype=np.float64)

        reproject(
            source=dem_data,
            destination=dem_aligned,
            src_transform=dem_src.transform,
            src_crs=dem_src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=np.nan,
            dst_nodata=np.nan,
        )

    return dem_aligned


def compute_chm(dsm_filled, dem_aligned):
    """Compute a canopy height model: CHM = DSM - DEM.

    Parameters
    ----------
    dsm_filled : numpy.ndarray
        DSM with NoData filled (includes canopy).
    dem_aligned : numpy.ndarray
        DEM aligned to the DSM grid (bare earth).

    Returns
    -------
    chm : numpy.ndarray
        Canopy height model. Negative values are clipped to 0.
    """
    chm = dsm_filled - dem_aligned
    # Clip negative values (DEM slightly above DSM due to noise/interpolation)
    chm = np.where(np.isfinite(chm), np.maximum(chm, 0.0), np.nan)
    return chm
