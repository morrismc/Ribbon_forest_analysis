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
    mask = dsm.mask if dsm.mask is not np.bool_(False) else np.zeros_like(data, dtype=bool)

    if not mask.any():
        return data

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
