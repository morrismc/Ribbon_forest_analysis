"""Least-squares surface fitting and removal for DSM detrending.

Supports both planar (order=1) and higher-order polynomial detrending.
The planar fit replicates the approach from Perron's 2DSpecTools
(detrend_tp.m / lsplane.m).  Higher-order fits (order=2, 3) handle
hillslope curvature that a single plane cannot capture.
"""

import numpy as np


def _prepare_data(dsm):
    """Extract a plain ndarray and a boolean mask from a DSM input.

    Parameters
    ----------
    dsm : numpy.ndarray or numpy.ma.MaskedArray

    Returns
    -------
    data : numpy.ndarray
    mask : numpy.ndarray of bool
    """
    if isinstance(dsm, np.ma.MaskedArray):
        data = dsm.filled(np.nan)
        if isinstance(dsm.mask, np.ndarray):
            mask = dsm.mask | ~np.isfinite(data)
        else:
            mask = ~np.isfinite(data)
    else:
        data = np.asarray(dsm, dtype=float).copy()
        mask = ~np.isfinite(data)
    return data, mask


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


def fit_polynomial_surface(dsm, mask=None, order=2):
    """Fit a 2-D polynomial surface of a given order to the DSM.

    order=1 is equivalent to fit_plane().
    order=2 fits z = a*x^2 + b*y^2 + c*xy + d*x + e*y + f  (6 terms).
    order=3 adds cubic terms (10 terms total).

    Parameters
    ----------
    dsm : numpy.ndarray
        2-D elevation array.
    mask : numpy.ndarray of bool, optional
        True where pixels are invalid/masked.
    order : int
        Polynomial order (1, 2, or 3).

    Returns
    -------
    surface : numpy.ndarray
        2-D array of the fitted surface evaluated at every pixel.
    coeffs : numpy.ndarray
        Fitted polynomial coefficients.
    """
    nrows, ncols = dsm.shape
    rows, cols = np.mgrid[0:nrows, 0:ncols]

    if mask is not None and np.any(mask):
        valid = ~mask
        x = cols[valid].ravel().astype(float)
        y = rows[valid].ravel().astype(float)
        z = dsm[valid].ravel().astype(float)
    else:
        x = cols.ravel().astype(float)
        y = rows.ravel().astype(float)
        z = dsm.ravel().astype(float)

    # Normalise coordinates to [0, 1] for numerical stability
    x_scale = max(ncols - 1, 1)
    y_scale = max(nrows - 1, 1)
    xn = x / x_scale
    yn = y / y_scale

    # Build design matrix with all terms up to the given order
    terms = []
    for total_deg in range(order + 1):
        for iy in range(total_deg + 1):
            ix = total_deg - iy
            terms.append(xn ** ix * yn ** iy)
    A = np.column_stack(terms)

    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

    # Evaluate on the full grid
    xn_full = cols.astype(float) / x_scale
    yn_full = rows.astype(float) / y_scale
    surface = np.zeros_like(dsm, dtype=float)
    idx = 0
    for total_deg in range(order + 1):
        for iy in range(total_deg + 1):
            ix = total_deg - iy
            surface += coeffs[idx] * xn_full ** ix * yn_full ** iy
            idx += 1

    return surface, coeffs


def detrend_dsm(dsm, order=1):
    """Remove a best-fit surface from the DSM.

    Parameters
    ----------
    dsm : numpy.ndarray or numpy.ma.MaskedArray
        2-D elevation array.
    order : int
        Polynomial order for the trend surface.  1 = planar (default),
        2 = quadratic, 3 = cubic.  Use order >= 2 when the study area
        spans significant hillslope curvature.

    Returns
    -------
    detrended : numpy.ndarray
        DSM with the trend surface removed.
    surface : numpy.ndarray
        The fitted surface that was subtracted.
    coeffs : tuple or numpy.ndarray
        Fitted coefficients.
    """
    data, mask = _prepare_data(dsm)

    if order == 1:
        coeffs, surface = fit_plane(data, mask=mask)
    else:
        surface, coeffs = fit_polynomial_surface(data, mask=mask, order=order)

    detrended = data - surface

    # Re-apply NaN where original was masked
    if np.any(mask):
        detrended[mask] = np.nan

    return detrended, surface, coeffs
