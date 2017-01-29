"""Processing and signal analysis.
"""
# TODO: fornberg, exact number of points
# TODO: singh     higher different order and k
# TODO: cwt
# TODO: peak_find == cwt + interp + idxmax
# TODO: allow reductions with new_dim=None
# TODO: only optionally handle nan
# TODO: check which methods should be parsing nan

import functools

import numpy as np
from scipy import interpolate, signal
import xarray as xr
from xarray.core.computation import apply_ufunc
import numba

from .utils import argwhere


def nan_wrap_const_length(fn):
    """Take a function that accepts two vector arguments y and x and wrap it
    such that only non-nan sections are supplied to it. Assume output vector is
    the same length as input.
    """
    def nanified(fx, x, *args, **kwargs):
        notnull = np.isfinite(fx)
        if np.all(notnull):
            # All data present
            return fn(fx, x, *args, **kwargs)
        elif np.any(notnull):
            # Some missing data,
            res = np.tile(np.nan, fx.shape)
            res[notnull] = fn(fx[notnull], x[notnull], *args, **kwargs)
            return res
        else:
            # No valid data, just return nans
            return np.tile(np.nan, fx.shape)

    return nanified


def xr_1d_apply(func, xobj, dim, new_dim=False, leave_nan=False):
    """Take a vector function and wrap it so that it can be applied along
    xarray dimensions.
    """
    if leave_nan:
        # Assume function can handle nan values itself.
        nnfunc = func
    else:  # modify function to only act on nonnull data
        def nnfunc(xin):
            """Nan wrap a function, but accepting changes to the coordinates.
            """
            isnull = ~np.isfinite(xin)
            # Just use function normally if no missing data
            if not np.any(isnull):
                return func(xin)

            # If all null, match new output dimension with np.nan
            elif np.all(isnull):
                if new_dim is None:
                    return np.nan
                if new_dim is False:
                    return np.tile(np.nan, xin.size)
                else:
                    return np.tile(np.nan, new_dim.size)

            # Partially null: apply function only on not null data
            else:
                nonnull = ~isnull
                part_res = func(xin[nonnull])
                if new_dim is False:
                    # just match old shape
                    res = np.empty(xin.shape, dtype=part_res.dtype)
                    res[isnull] = np.nan
                    res[nonnull] = part_res
                    return res
                else:  # can't know
                    return part_res

    new_xobj = xobj.copy(deep=True)
    # convert to dataset temporarily
    if isinstance(xobj, xr.DataArray):
        new_xobj = new_xobj.to_dataset(name='__temp_name__')

    # if the dimension is changing, create a temporary one
    if not (new_dim is None or new_dim is False):
        new_xobj.coords['__temp_dim__'] = new_dim

    # calculate fn and insert with new coord
    for v in new_xobj.data_vars:
        var = new_xobj[v]
        if dim in var.dims:
            old_dims = var.dims
            axis = argwhere(old_dims, dim)
            if new_dim is False:
                new_dims = old_dims
            elif new_dim is None:
                new_dims = tuple(d for d in old_dims if d != dim)
            else:
                new_dims = tuple((d if d != dim else '__temp_dim__')
                                 for d in old_dims)
            new_xobj[v] = (new_dims, np.apply_along_axis(
                nnfunc, axis, var.data))

    if not (new_dim is None or new_dim is False):
        new_xobj = new_xobj.drop(dim)
        new_xobj = new_xobj.rename({'__temp_dim__': dim})

    # convert back to dataarray if originally given, strip temp name
    if isinstance(xobj, xr.DataArray):
        # new_xobj = new_xobj.to_array()
        # new_xobj.name = xobj.name
        # new_xobj = new_xobj.rename({'__temp_name__': xobj.name})
        new_xobj = new_xobj['__temp_name__']
        new_xobj.name = xobj.name

    return new_xobj


# --------------------------------------------------------------------------- #
#                    fornberg's finite difference algortihm                   #
# --------------------------------------------------------------------------- #

@numba.jit(["float64(float64[:],float64[:],float64,int64)"],
           nopython=True)  # pragma: no cover
def finite_difference_fornberg(fx, x, z, order):
    """Fornberg finite difference method for single poitn `z`.
    """
    c1 = 1.0
    c4 = x[0] - z
    n = len(x) - 1
    c = np.zeros((len(x), order + 1))
    c[0, 0] = 1.0

    for i in range(1, n + 1):
        mn = min(i, order)
        c2 = 1.0
        c5 = c4
        c4 = x[i] - z

        for j in range(0, i):
            c3 = x[i] - x[j]
            c2 *= c3

            if j == i - 1:

                for k in range(mn, 0, -1):
                    c[i, k] = c1 * (k * c[i - 1, k - 1] -
                                    c5 * c[i - 1, k]) / c2
                c[i, 0] = -c1 * c5 * c[i - 1, 0] / c2

            for k in range(mn, 0, -1):
                c[j, k] = (c4 * c[j, k] - k * c[j, k - 1]) / c3

            c[j, 0] = c4 * c[j, 0] / c3

        c1 = c2

    return np.dot(c[:, order], fx)


@numba.jit(nopython=True)  # pragma: no cover
def finite_diff_array(fx, x, ix, order, window):
    """Fornberg finite difference method for array of points `ix`.
    """
    out = np.empty(len(ix))
    fx = fx.astype(np.float64)

    w = window
    if w < 0:  # use whole window
        for i, z in enumerate(ix):
            out[i] = finite_difference_fornberg(fx, x, z, order)
    else:
        forward_limit = (x[0] + w / 2)
        foward_win = x[0] + w
        backward_limit = (x[-1] - w / 2)
        backward_win = x[-1] - w

        for i, z in enumerate(ix):
            if z < forward_limit:  # use forward diff
                bm = np.less(x, foward_win)
            elif z > backward_limit:  # backward diff
                bm = np.greater(x, backward_win)
            else:  # central diff
                bm = np.less(np.abs(x - z), w)
            wx = x[bm]
            wfx = fx[bm]
            out[i] = finite_difference_fornberg(wfx, wx, z, order)
    return out


def wfdiff(fx, x, ix, order, mode='points', window=5, return_func=False):
    """Find (d^k fx)/(dx^k) at points ix, using a windowed finite difference.
    This is only appropirate for very nicely sampled/analytic data.

    Uses algorithm found in:
        Calculation of Weights in Finite Difference Formulas
        Bengt Fornberg
        SIAM Rev., 40(3), 685–691
        http://dx.doi.org/10.1137/S0036144596322507

    Parameters
    ----------
        fx : array
            Function values at grid values.
        x : array
            Grid values, same legnth as `fx`, assumed sorted.
        ix : array or int
            If array, values at which to evalute finite difference else number
            of points to linearly space in the range and use.
        order : int
            Order of derivate, 0 yields an interpolation.
        mode : {'points', 'relative', 'absolute'}, optional
            Used in conjuction with window.
            If 'points', the window size is set such that the average number
            of points in each window is given b `window`.
            If 'relative', the window size is set such that its size relative
            to the total range is given by `window`.
            If 'absolute', the window size is geven explicitly by `window`.
        window : int or float, optional
            Depends on `mode`,
                - 'points', target number of points to use for each window.
                - 'relative' relative size of window compared to full range.
                - 'absolute' The absolute window size.
        return_func : bool, optional
            If True, return the single argument function wfdiff(fx).

    Returns
    -------
        ifx : array
            Interpolated kth-derivative of data.
    """
    if mode in {'p', 'pts', 'points'}:
        abs_win = (x[-1] - x[0]) * window / len(x)
    elif mode in {'r', 'rel', 'relative'}:
        abs_win = (x[-1] - x[0]) * window
    elif mode in {'a', 'abs', 'absolute'}:
        abs_win = window
    else:
        raise ValueError("mode: {} not valid".format(mode))

    if isinstance(ix, int):
        ix = np.linspace(x[0], x[-1], ix)
    elif ix.dtype != float:
        ix = ix.astype(float)
    if x.dtype != float:
        x = x.astype(float)

    if return_func:
        return functools.partial(finite_diff_array, x=x, ix=ix,
                                 order=order, window=abs_win)
    else:
        return finite_diff_array(fx, x, ix, order, abs_win)


def xr_wfdiff(xobj, dim, ix=100, order=1, mode='points', window=5):
    """Find (d^k fx)/(dx^k) at points ix, using a windowed finite difference.
    This is only appropirate for very nicely sampled/analytic data.

    Uses algorithm found in:
        Calculation of Weights in Finite Difference Formulas
        Bengt Fornberg
        SIAM Rev., 40(3), 685–691
        http://dx.doi.org/10.1137/S0036144596322507

    Paramters
    ---------
        xobj : xarray.DataArray or xarray.Dataset
            Object to find windowed finite difference for.
        dim : str
            Dimension to find windowed finite difference along.
        ix : array or int
            If array, values at which to evalute finite difference else number
            of points to linearly space in the range and use.
        order : int
            Order of derivate, 0 yields an interpolation.
        mode : {'points', 'relative', 'absolute'}
            Used in conjuction with window.
            If 'points', the window size is set such that the average number
            of points in each window is given b `window`.
            If 'relative', the window size is set such that its size relative
            to the total range is given by `window`.
            If 'absolute', the window size is geven explicitly by `window`.
        window : int or float
            Depends on `mode`,
                - 'points', target number of points to use for each window.
                - 'relative' relative size of window compared to full range.
                - 'absolute' The absolute window size.

    Returns
    -------
        new_xobj : xarray.DataArray or xarray.Dataset
            Object now with windowed finite difference along `dim`.
    """
    # original grid
    x = xobj[dim].data
    # generate interpolation grid if not given as array, and set as coords
    if isinstance(ix, int):
        ix = np.linspace(x[0], x[-1], ix)
    # make re-useable single arg function
    diff_fn = wfdiff(None, x=x, ix=ix, order=order, mode=mode,
                     window=window, return_func=True)
    return xr_1d_apply(diff_fn, xobj, dim, new_dim=ix)


xr.Dataset.fdiff = xr_wfdiff
xr.DataArray.fdiff = xr_wfdiff


# --------------------------------------------------------------------------- #
#                         Simple averaged scaled diff                         #
# --------------------------------------------------------------------------- #

@numba.jit(nopython=True)
def simple_average_diff(fx, x, k=1):
    n = len(x)
    dfx = np.empty(n - k)
    for i in range(n - k):
        dfx[i] = (fx[i + 1] - fx[i]) / (x[i + 1] - x[i])
        for j in range(i + 1, i + k):
            dfx[i] += (fx[j + 1] - fx[j]) / (x[j + 1] - x[j])
        dfx[i] /= k
    return dfx


def xr_sdiff(xobj, dim, k=1):
    x = xobj[dim].data
    n = len(x)
    nx = sum(x[ki:n - k + ki] for ki in range(k + 1)) / (k + 1)
    func = functools.partial(simple_average_diff, x=x, k=k)
    return xr_1d_apply(func, xobj, dim, new_dim=nx)


xr.Dataset.sdiff = xr_sdiff
xr.DataArray.sdiff = xr_sdiff


# --------------------------------------------------------------------------- #
#                      Unevenly spaced finite difference                      #
# ---------------------------------------------------------------------------

@nan_wrap_const_length
@numba.jit(nopython=True)
def usdiff(fx, x):
    """
    Singh, Ashok K., and B. S. Bhadauria. "Finite difference formulae for
    unequal sub-intervals using lagrange’s interpolation formula."
    International Journal of Mathematics and Analysis 3.17 (2009): 815-827.
    """
    n = len(x)
    dfx = np.empty(n)

    # Forward difference for first point
    h1 = x[1] - x[0]
    h2 = x[2] - x[1]
    dfx[0] = (- fx[0] * (2 * h1 + h2) / (h1 * (h1 + h2)) +
              fx[1] * (h1 + h2) / (h1 * h2) -
              fx[2] * h1 / ((h1 + h2) * h2))

    # Central difference for middle points
    for i in range(1, n - 1):
        h1 = x[i] - x[i - 1]
        h2 = x[i + 1] - x[i]
        dfx[i] = (- fx[i - 1] * h2 / (h1 * (h1 + h2)) -
                  fx[i] * (h1 - h2) / (h1 * h2) +
                  fx[i + 1] * h1 / (h2 * (h1 + h2)))

    # Backwards difference for last point
    h1 = x[n - 2] - x[n - 3]
    h2 = x[n - 1] - x[n - 2]
    dfx[n - 1] = (fx[n - 3] * h2 / (h1 * (h1 + h2)) -
                  fx[n - 2] * (h1 + h2) / (h1 * h2) +
                  fx[n - 1] * (h1 + 2 * h2) / (h2 * (h1 + h2)))

    return dfx


def xr_usdiff(xobj, dim):
    """Uneven-third-order finite difference derivative.
    """
    func = functools.partial(usdiff, x=xobj[dim].values)
    return xr_1d_apply(func, xobj, dim, new_dim=False, leave_nan=True)


xr.Dataset.usdiff = xr_usdiff
xr.DataArray.usdiff = xr_usdiff


@nan_wrap_const_length
@numba.jit(nopython=True)
def usdiff_err(efx, x):
    """
    Propagate unvertainties using uneven finite difference formula.
    """
    n = len(x)
    edfx = np.empty(n)

    # Forward difference for first point
    h1 = x[1] - x[0]
    h2 = x[2] - x[1]
    edfx[0] = ((efx[0] * (2 * h1 + h2) / (h1 * (h1 + h2)))**2 +
               (efx[1] * (h1 + h2) / (h1 * h2))**2 +
               (efx[2] * h1 / ((h1 + h2) * h2))**2)**0.5

    # Central difference for middle points
    for i in range(1, n - 1):
        h1 = x[i] - x[i - 1]
        h2 = x[i + 1] - x[i]
        edfx[i] = ((efx[i - 1] * h2 / (h1 * (h1 + h2)))**2 +
                   (efx[i] * (h1 - h2) / (h1 * h2))**2 +
                   (efx[i + 1] * h1 / (h2 * (h1 + h2)))**2)**0.5

    # Backwards difference for last point
    h1 = x[n - 2] - x[n - 3]
    h2 = x[n - 1] - x[n - 2]
    edfx[n - 1] = ((efx[n - 3] * h2 / (h1 * (h1 + h2)))**2 +
                   (efx[n - 2] * (h1 + h2) / (h1 * h2))**2 +
                   (efx[n - 1] * (h1 + 2 * h2) / (h2 * (h1 + h2)))**2)**0.5

    return edfx


def xr_usdiff_err(xobj, dim):
    """Uneven-third-order finite difference derivative.
    """
    func = functools.partial(usdiff_err, x=xobj[dim].values)
    return xr_1d_apply(func, xobj, dim, new_dim=False, leave_nan=True)


xr.Dataset.usdiff_err = xr_usdiff_err
xr.DataArray.usdiff_err = xr_usdiff_err


# --------------------------------------------------------------------------- #
#                            spline interpolation                             #
# --------------------------------------------------------------------------- #


def array_interp1d(fx, x, ix, kind='cubic', return_func=False, **kwargs):

    if isinstance(ix, int):
        ix = np.linspace(x[0], x[-1], ix)

    def fx_interp(fx):
        ifn = interpolate.interp1d(x, fx, kind=kind, **kwargs)
        return ifn(ix)

    if return_func:
        return fx_interp
    return fx_interp(fx)


def xr_interp1d(xobj, dim, ix=100, kind='cubic', **kwargs):
    # original grid
    x = xobj[dim].data
    # generate interpolation grid if not given as array, and set as coords
    if isinstance(ix, int):
        ix = np.linspace(x[0], x[-1], ix)
    # make re-useable single arg function
    fn = array_interp1d(None, x=x, ix=ix, kind=kind,
                        return_func=True, **kwargs)
    return xr_1d_apply(fn, xobj, dim, new_dim=ix, leave_nan=False)


xr.Dataset.interp = xr_interp1d
xr.DataArray.interp = xr_interp1d


# --------------------------------------------------------------------------- #
#                            pchip interpolation                              #
# --------------------------------------------------------------------------- #

def array_pchip(fx, x, ix, return_func=False):

    if isinstance(ix, int):
        ix = np.linspace(x[0], x[-1], ix)

    def fx_interp(fx):
        ifn = interpolate.PchipInterpolator(x, fx)
        return ifn(ix)

    if return_func:
        return fx_interp
    return fx_interp(fx)


def xr_pchip(xobj, dim, ix=100):
    # original grid
    x = xobj[dim].data
    # generate interpolation grid if not given as array, and set as coords
    if isinstance(ix, int):
        ix = np.linspace(x[0], x[-1], ix)
    # make re-useable single arg function
    fn = array_interp1d(None, x=x, ix=ix, return_func=True)
    return xr_1d_apply(fn, xobj, dim, new_dim=ix, leave_nan=False)


xr.Dataset.pchip = xr_pchip
xr.DataArray.pchip = xr_pchip


# --------------------------------------------------------------------------- #
#                           scipy signal filtering                            #
# --------------------------------------------------------------------------- #

def xr_filter_wiener(xobj, dim, *args, **kwargs):
    func = functools.partial(signal.wiener, *args, **kwargs)
    return xr_1d_apply(func, xobj, dim, new_dim=False)


xr.DataArray.wiener = xr_filter_wiener
xr.Dataset.wiener = xr_filter_wiener


def xr_filtfilt(xobj, dim, filt='butter', *args, **kwargs):
    filter_func = getattr(signal, filt)
    b, a = filter_func(*args, **kwargs)

    def func(x):
        return signal.filtfilt(b, a, x, method='gust')

    return xr_1d_apply(func, xobj, dim, new_dim=False, leave_nan=False)


xr.DataArray.filtfilt = xr_filtfilt
xr.Dataset.filtfilt = xr_filtfilt


# --------------------------------------------------------------------------- #
#                               idxmax idxmin                                 #
# --------------------------------------------------------------------------- #

def gufunc_idxmax(x, y, **kwargs):
    indx = np.argmax(x, **kwargs)
    res = np.take(y, indx)
    return res


def xr_idxmax(obj, dim):
    sig = ([(dim,), (dim,)], [()])
    kwargs = {'axis': -1}
    return apply_ufunc(gufunc_idxmin, obj, obj[dim],
                       signature=sig, kwargs=kwargs)


xr.DataArray.idxmax = xr_idxmax
xr.Dataset.idxmax = xr_idxmax


def gufunc_idxmin(x, y, **kwargs):
    indx = np.argmin(x, **kwargs)
    res = np.take(y, indx)
    return res


def xr_idxmin(obj, dim):
    sig = ([(dim,), (dim,)], [()])
    kwargs = {'axis': -1}
    return apply_ufunc(gufunc_idxmin, obj, obj[dim],
                       signature=sig, kwargs=kwargs)


xr.DataArray.idxmin = xr_idxmin
xr.Dataset.idxmin = xr_idxmin
