"""Least-squares plane fitting and removal for DSM detrending.

Replicates the approach from Perron's 2DSpecTools (detrend_tp.m / lsplane.m):
fit z = ax + by + c via least-squares on the valid (non-masked) pixels, then
subtract the fitted plane to isolate local relief (vegetation signal).
"""

import numpy as np


def fit_plane(dsm, mask=None):
    """Fit a least-squares plane z = ax + by + c to the DSM.

    Parameters
    ----------
    dsm : numpy.ndarray
        2-D elevation array.
    mask : numpy.ndarray of bool, optional
        True where pixels are invalid/masked. If None, all pixels are used.

    Returns
    -------
    coeffs : tuple of float
        (a, b, c) plane coefficients.
    plane : numpy.ndarray
        2-D array of the fitted plane evaluated at every pixel.
    """
    nrows, ncols = dsm.shape

    # Build coordinate grids (row = y, col = x)
    rows, cols = np.mgrid[0:nrows, 0:ncols]

    if mask is not None and mask.any():
        valid = ~mask
        x = cols[valid].ravel()
        y = rows[valid].ravel()
        z = dsm[valid].ravel()
    else:
        x = cols.ravel()
        y = rows.ravel()
        z = dsm.ravel()

    # Design matrix: [x, y, 1]
    A = np.column_stack([x, y, np.ones_like(x)])

    # Solve via least-squares (uses SVD internally)
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    a, b, c = coeffs

    # Evaluate plane on full grid
    plane = a * cols + b * rows + c

    return (a, b, c), plane


def detrend_dsm(dsm):
    """Remove a best-fit plane from the DSM.

    Parameters
    ----------
    dsm : numpy.ndarray or numpy.ma.MaskedArray
        2-D elevation array.

    Returns
    -------
    detrended : numpy.ndarray
        DSM with the planar trend removed.
    plane : numpy.ndarray
        The fitted plane that was subtracted.
    coeffs : tuple
        (a, b, c) plane coefficients.
    """
    if isinstance(dsm, np.ma.MaskedArray):
        data = dsm.filled(np.nan)
        if isinstance(dsm.mask, np.ndarray):
            mask = dsm.mask | ~np.isfinite(data)
        else:
            mask = ~np.isfinite(data)
    else:
        data = dsm.copy()
        mask = ~np.isfinite(data)

    coeffs, plane = fit_plane(data, mask=mask)
    detrended = data - plane

    # Re-apply NaN where original was masked
    if np.any(mask):
        detrended[mask] = np.nan

    return detrended, plane, coeffs
