"""Processing and signal analysis.
"""

import functools

import numpy as np
from scipy import interpolate, signal
import xarray as xr
from xarray.core.computation import apply_ufunc
from numba import njit, guvectorize, double, int_
from scipy.interpolate import LSQUnivariateSpline


_NUMBA_CACHE_DEFAULT = False


class LazyCompile(object):
    """Class that does 'compiles' a function only when called.
    """

    def __init__(self, fn, compiler, compiler_args, compiler_kwargs):
        self.fn = fn
        self.compiler = compiler
        self.compiled = False
        self.compiler_args = compiler_args
        self.compiler_kwargs = compiler_kwargs

    def compile_fn(self):
        self._fn = self.compiler(*self.compiler_args,
                                 **self.compiler_kwargs)(self.fn)
        self.compiled = True

    def __call__(self, *args, **kwargs):
        if not self.compiled:
            self.compile_fn()
        return self._fn(*args, **kwargs)


def lazy_guvectorize(*gufunc_args, **gufunc_kwargs):
    """Function to wrap LazyCompile around functions.
    """
    def actual_fn_wrapper(fn):
        lazy_fn = LazyCompile(fn, guvectorize, gufunc_args, gufunc_kwargs)

        @functools.wraps(fn)
        def cached_function(*args, **kwargs):
            return lazy_fn(*args, **kwargs)

        return cached_function

    return actual_fn_wrapper


@njit
def preprocess_nan_func(x, y, out):  # pragma: no cover
    """Pre-process data for a 1d function that doesn't accept nan-values.
    """
    # strip out nan
    mask = np.isfinite(x) & np.isfinite(y)
    num_nan = np.sum(~mask)

    if num_nan == x.size:
        out[:] = np.nan
        return
    elif num_nan != 0:
        x = x[mask]
        y = y[mask]

    return x, y, num_nan, mask


@njit
def postprocess_nan_func(yf, num_nan, mask, out):  # pragma: no cover
    """Post-process data for a 1d function that doesn't accept nan-values.
    """
    if num_nan != 0:
        out[~mask] = np.nan
        out[mask] = yf
    else:
        out[:] = yf


def preprocess_interp1d_nan_func(x, y, out):
    """Pre-process data for a 1d function that doesn't accept nan-values and
    needs evenly spaced data.
    """
    # strip out nan
    mask = np.isfinite(x) & np.isfinite(y)
    num_nan = np.sum(~mask)

    if num_nan == x.size:
        out[:] = np.nan
        return
    elif num_nan != 0:
        x = x[mask]
        y = y[mask]

    # interpolate to evenly spaced grid
    x_even = np.linspace(x.min(), x.max(), x.size)
    y_even = np.interp(x_even, x, y)
    return x, y, x_even, y_even, num_nan, mask


def postprocess_interp1d_nan_func(x, x_even, yf_even, num_nan, mask, out):
    """Post-process data for a 1d function that doesn't accept nan-values and
    needs evenly spaced data.
    """
    # re-interpolate to original spacing
    yf = np.interp(x, x_even, yf_even)

    if num_nan != 0:
        out[~mask] = np.nan
        out[mask] = yf
    else:
        out[:] = yf


# --------------------------------------------------------------------------- #
#                    fornberg's finite difference algortihm                   #
# --------------------------------------------------------------------------- #

@njit
def diff_fornberg(fx, x, z, order):  # pragma: no cover
    """Fornberg finite difference method for single point `z`.
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


@lazy_guvectorize([
    (int_[:], int_[:], double[:], int_[:], double[:], double[:]),
    (double[:], int_[:], double[:], int_[:], double[:], double[:]),
    (int_[:], double[:], double[:], int_[:], double[:], double[:]),
    (double[:], double[:], double[:], int_[:], double[:], double[:]),
], '(n),(n),(m),(),()->(m)', cache=_NUMBA_CACHE_DEFAULT, nopython=True)
def finite_diff_array(fx, x, ix, order, window, out=None):  # pragma: no cover
    """Fornberg finite difference method for array of points `ix`.
    """
    fx = fx.astype(np.float64)

    w = window[0]
    if w < 0:  # use whole window
        for i, z in enumerate(ix):
            out[i] = diff_fornberg(fx, x, z, order[0])
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
            out[i] = diff_fornberg(wfx, wx, z, order[0])


def _diff_fornberg_broadcast(x, fx, ix, order, mode='points',
                             window=5, axis=-1):
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
    if axis != -1:
        fx = fx.swapaxes(axis, -1)

    if mode in {'p', 'pts', 'points'}:
        abs_win = (x[-1] - x[0]) * window / len(x)
    elif mode in {'r', 'rel', 'relative'}:
        abs_win = (x[-1] - x[0]) * window
    elif mode in {'a', 'abs', 'absolute'}:
        abs_win = float(window)
    else:
        raise ValueError("mode: {} not valid".format(mode))

    if ix is None:
        ix = x.astype(float)
    elif isinstance(ix, int):
        ix = np.linspace(x[0], x[-1], ix)
    elif ix.dtype != float:
        ix = ix.astype(float)

    return finite_diff_array(fx, x, ix, order, abs_win)


def xr_diff_fornberg(obj, dim, ix=100, order=1, mode='points', window=5):
    """Find ``(d^k fx)/(dx^k)`` at points ``ix``, using a windowed finite
    difference. This is only appropirate for very nicely sampled/analytic data.

    Uses algorithm found in:
        Calculation of Weights in Finite Difference Formulas
        Bengt Fornberg
        SIAM Rev., 40(3), 685–691
        http://dx.doi.org/10.1137/S0036144596322507

    Parameters
    ----------
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
        of points in each window is given by `window`.
        If 'relative', the window size is set such that its size relative
        to the total range is given by `window`.
        If 'absolute', the window size is geven explicitly by `window`.
    window : int or float
        Depends on ``mode``:

        - 'points', target number of points to use for each window.
        - 'relative' relative size of window compared to full range.
        - 'absolute' The absolute window size.

    Returns
    -------
    new_xobj : xarray.DataArray or xarray.Dataset
        Object now with windowed finite difference along `dim`.
    """
    input_core_dims = [(dim,), (dim,)]
    args = (obj[dim], obj)

    if ix is None:
        kwargs = {'ix': ix, 'axis': -1, 'order': order,
                  'mode': mode, 'window': window}

        output_core_dims = [(dim,)]
        return apply_ufunc(_diff_fornberg_broadcast, *args, kwargs=kwargs,
                           input_core_dims=input_core_dims,
                           output_core_dims=output_core_dims)

    if isinstance(ix, int):
        ix = np.linspace(float(obj[dim].min()), float(obj[dim].max()), ix)

    kwargs = {'ix': ix, 'axis': -1, 'order': order,
              'mode': mode, 'window': window}
    output_core_dims = [('__temp_dim__',)]

    result = apply_ufunc(_diff_fornberg_broadcast, *args, kwargs=kwargs,
                         input_core_dims=input_core_dims,
                         output_core_dims=output_core_dims)
    result['__temp_dim__'] = ix
    return result.rename({'__temp_dim__': dim})


# --------------------------------------------------------------------------- #
#                      Unevenly spaced finite difference                      #
# --------------------------------------------------------------------------- #

@lazy_guvectorize([
    (int_[:], int_[:], double[:]),
    (double[:], int_[:], double[:]),
    (int_[:], double[:], double[:]),
    (double[:], double[:], double[:]),
], "(n),(n)->(n)", cache=_NUMBA_CACHE_DEFAULT, nopython=True)
def diff_u(fx, x, out=None):  # pragma: no cover
    xynm = preprocess_nan_func(x, fx, out)
    if xynm is None:
        return
    x, fx, num_nan, mask = xynm
    yf = np.empty_like(fx)

    n = len(x)

    # Forward difference for first point
    h1 = x[1] - x[0]
    h2 = x[2] - x[1]
    yf[0] = (- fx[0] * (2 * h1 + h2) / (h1 * (h1 + h2)) +
             fx[1] * (h1 + h2) / (h1 * h2) -
             fx[2] * h1 / ((h1 + h2) * h2))

    # Central difference for middle points
    for i in range(1, n - 1):
        h1 = x[i] - x[i - 1]
        h2 = x[i + 1] - x[i]
        yf[i] = (- fx[i - 1] * h2 / (h1 * (h1 + h2)) -
                 fx[i] * (h1 - h2) / (h1 * h2) +
                 fx[i + 1] * h1 / (h2 * (h1 + h2)))

    # Backwards difference for last point
    h1 = x[n - 2] - x[n - 3]
    h2 = x[n - 1] - x[n - 2]
    yf[n - 1] = (fx[n - 3] * h2 / (h1 * (h1 + h2)) -
                 fx[n - 2] * (h1 + h2) / (h1 * h2) +
                 fx[n - 1] * (h1 + 2 * h2) / (h2 * (h1 + h2)))

    postprocess_nan_func(yf, num_nan, mask, out)


def _broadcast_diff_u(x, fx, axis=-1):
    if axis != -1:
        fx = fx.swapaxes(axis, -1)
    return diff_u(fx, x)


def xr_diff_u(obj, dim):
    """Uneven-third-order finite difference derivative [1].

    [1] Singh, Ashok K., and B. S. Bhadauria. "Finite difference formulae for
    unequal sub-intervals using lagrange’s interpolation formula."
    International Journal of Mathematics and Analysis 3.17 (2009): 815-827.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        The object to differentiate.
    dim : str
        The dimension to differentiate along.

    Returns
    -------
    new_xobj : xarray.DataArray or xarray.Dataset
    """
    kwargs = {'axis': -1}
    input_core_dims = [(dim,), (dim,)]
    output_core_dims = [(dim,)]
    args = (obj[dim], obj)
    return apply_ufunc(_broadcast_diff_u, *args,
                       input_core_dims=input_core_dims,
                       output_core_dims=output_core_dims,
                       kwargs=kwargs)


@lazy_guvectorize([
    (int_[:], int_[:], double[:]),
    (double[:], int_[:], double[:]),
    (int_[:], double[:], double[:]),
    (double[:], double[:], double[:]),
], "(n),(n)->(n)", cache=_NUMBA_CACHE_DEFAULT, nopython=True)
def diff_u_err(efx, x, out=None):  # pragma: no cover
    """Propagate uncertainties using uneven finite difference formula.
    """
    xynm = preprocess_nan_func(x, efx, out)
    if xynm is None:
        return
    x, efx, num_nan, mask = xynm
    yf = np.empty_like(efx)

    n = len(x)
    yf = np.empty_like(efx)

    # Forward difference for first point
    h1 = x[1] - x[0]
    h2 = x[2] - x[1]
    yf[0] = ((efx[0] * (2 * h1 + h2) / (h1 * (h1 + h2)))**2 +
             (efx[1] * (h1 + h2) / (h1 * h2))**2 +
             (efx[2] * h1 / ((h1 + h2) * h2))**2)**0.5

    # Central difference for middle points
    for i in range(1, n - 1):
        h1 = x[i] - x[i - 1]
        h2 = x[i + 1] - x[i]
        yf[i] = ((efx[i - 1] * h2 / (h1 * (h1 + h2)))**2 +
                 (efx[i] * (h1 - h2) / (h1 * h2))**2 +
                 (efx[i + 1] * h1 / (h2 * (h1 + h2)))**2)**0.5

    # Backwards difference for last point
    h1 = x[n - 2] - x[n - 3]
    h2 = x[n - 1] - x[n - 2]
    yf[n - 1] = ((efx[n - 3] * h2 / (h1 * (h1 + h2)))**2 +
                 (efx[n - 2] * (h1 + h2) / (h1 * h2))**2 +
                 (efx[n - 1] * (h1 + 2 * h2) / (h2 * (h1 + h2)))**2)**0.5

    postprocess_nan_func(yf, num_nan, mask, out)


def _broadcast_diff_u_err(x, fx, axis=-1):
    if axis != -1:
        fx = fx.swapaxes(axis, -1)
    return diff_u_err(fx, x)


def xr_diff_u_err(obj, dim):
    """Propagate error through uneven-third-order finite difference derivative.
    If you have calculated a derivative already using ``xr_diff_u``, and you
    have data about the uncertainty on the original data, this function
    propagates that error through to be an error on the derivative.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        The object to differentiate.
    dim : str
        The dimension to differentiate along.

    Returns
    -------
    new_xobj : xarray.DataArray or xarray.Dataset
    """
    kwargs = {'axis': -1}
    input_core_dims = [(dim,), (dim,)]
    output_core_dims = [(dim,)]
    args = (obj[dim], obj)
    return apply_ufunc(_broadcast_diff_u_err, *args,
                       input_core_dims=input_core_dims,
                       output_core_dims=output_core_dims,
                       kwargs=kwargs)


# --------------------------------------------------------------------------- #
#                                interpolation                                #
# --------------------------------------------------------------------------- #

_INTERP_INT2STR = {
    0: 'zero',
    1: 'slinear',
    2: 'quadratic',
    3: 'cubic',
}


@lazy_guvectorize([
    (int_[:], int_[:], int_[:], double[:]),
    (double[:], int_[:], int_[:], double[:]),
    (int_[:], double[:], int_[:], double[:]),
    (double[:], double[:], int_[:], double[:]),
], '(n),(n),()->(n)', cache=_NUMBA_CACHE_DEFAULT)
def _gufunc_interp(x, y, order, out=None):  # pragma: no cover
    xynm = preprocess_nan_func(x, y, out)
    if xynm is None:
        return
    x, y, num_nan, mask = xynm

    # interpolating function
    ifn = interpolate.interp1d(x, y, kind=_INTERP_INT2STR[order[0]],
                               bounds_error=False)
    yf = ifn(x)
    postprocess_nan_func(yf, num_nan, mask, out)


@lazy_guvectorize([
    (int_[:], int_[:], double[:], int_[:], double[:]),
    (double[:], int_[:], double[:], int_[:], double[:]),
    (int_[:], double[:], double[:], int_[:], double[:]),
    (double[:], double[:], double[:], int_[:], double[:])
], '(n),(n),(m),()->(m)', cache=_NUMBA_CACHE_DEFAULT)
def _gufunc_interp_upscale(x, y, ix, order, out=None):  # pragma: no cover
    xynm = preprocess_nan_func(x, y, out)
    if xynm is None:
        return
    x, y, _, _ = xynm

    # interpolating function
    ifn = interpolate.interp1d(x, y, kind=_INTERP_INT2STR[order[0]],
                               bounds_error=False)

    # no need to nan_func post process and upscaling into exact right size
    out[:] = ifn(ix)


def _broadcast_interp(x, y, ix=100, order=3, axis=-1):
    if axis != -1:
        y = y.swapaxes(axis, -1)

    if ix is None:
        return _gufunc_interp(x, y, order)

    if isinstance(ix, int):
        # automatic upscale
        ix = np.linspace(x.min(), x.max(), ix)

    return _gufunc_interp_upscale(x, y, ix, order)


def xr_interp(obj, dim, ix=100, order=3):
    """Interpolate along axis ``dim`` using :func:`scipy.interpolate.interp1d`.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        The object to interpolate.
    dim : str
        The axis to interpolate along.
    ix : int or array
        If int, interpolate to this many points spaced evenly along the range
        of the original data. If array, interpolate to those points directly.
    order : int
        Supplied to :func:`scipy.interpolate.interp1d` as the order of
        interpolation.

    Returns
    -------
    new_xobj : xarray.DataArray or xarray.Dataset

    See Also
    --------
    xr_pchip
    """

    input_core_dims = [(dim,), (dim,)]
    args = (obj[dim], obj)
    kwargs = {'ix': ix, 'axis': -1, 'order': order}

    if ix is None:
        output_core_dims = [(dim,)]
        return apply_ufunc(_broadcast_interp, *args, kwargs=kwargs,
                           input_core_dims=input_core_dims,
                           output_core_dims=output_core_dims)

    if isinstance(ix, int):
        ix = np.linspace(float(obj[dim].min()), float(obj[dim].max()), ix)

    kwargs['ix'] = ix
    output_core_dims = [('__temp_dim__',)]

    result = apply_ufunc(_broadcast_interp, *args, kwargs=kwargs,
                         input_core_dims=input_core_dims,
                         output_core_dims=output_core_dims)
    result['__temp_dim__'] = ix
    return result.rename({'__temp_dim__': dim})


# --------------------------------------------------------------------------- #
#                            pchip interpolation                              #
# --------------------------------------------------------------------------- #

@lazy_guvectorize([
    (int_[:], int_[:], double[:]),
    (double[:], int_[:], double[:]),
    (int_[:], double[:], double[:]),
    (double[:], double[:], double[:]),
], '(n),(n)->(n)', cache=_NUMBA_CACHE_DEFAULT)
def _gufunc_pchip(x, y, out=None):  # pragma: no cover
    xynm = preprocess_nan_func(x, y, out)
    if xynm is None:
        return
    x, y, num_nan, mask = xynm

    # interpolating function
    ifn = interpolate.PchipInterpolator(x, y, extrapolate=False)
    yf = ifn(x)
    postprocess_nan_func(yf, num_nan, mask, out)


@lazy_guvectorize([
    (int_[:], int_[:], double[:], double[:]),
    (double[:], int_[:], double[:], double[:]),
    (int_[:], double[:], double[:], double[:]),
    (double[:], double[:], double[:], double[:])
], '(n),(n),(m)->(m)', cache=_NUMBA_CACHE_DEFAULT)
def _gufunc_pchip_upscale(x, y, ix, out=None):  # pragma: no cover
    xynm = preprocess_nan_func(x, y, out)
    if xynm is None:
        return
    x, y, _, mask = xynm

    # interpolating function
    ifn = interpolate.PchipInterpolator(x, y, extrapolate=False)
    out[:] = ifn(ix)


def _broadcast_pchip(x, y, ix=100, axis=-1):
    if axis != -1:
        y = y.swapaxes(axis, -1)

    if ix is None:
        return _gufunc_pchip(x, y)

    if isinstance(ix, int):
        # automatic upscale
        ix = np.linspace(x.min(), x.max(), ix)

    return _gufunc_pchip_upscale(x, y, ix)


def xr_interp_pchip(obj, dim, ix=100):
    """Interpolate along axis ``dim`` using :func:`scipy.interpolate.pchip`.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        The object to interpolate.
    dim : str
        The axis to interpolate along.
    ix : int or array
        If int, interpolate to this many points spaced evenly along the range
        of the original data. If array, interpolate to those points directly.

    Returns
    -------
    new_xobj : xarray.DataArray or xarray.Dataset

    See Also
    --------
    xr_interp
    """

    input_core_dims = [(dim,), (dim,)]
    args = (obj[dim], obj)

    if ix is None:
        kwargs = {'ix': ix, 'axis': -1}
        output_core_dims = [(dim,)]
        return apply_ufunc(_broadcast_pchip, *args, kwargs=kwargs,
                           input_core_dims=input_core_dims,
                           output_core_dims=output_core_dims)

    if isinstance(ix, int):
        ix = np.linspace(float(obj[dim].min()), float(obj[dim].max()), ix)

    kwargs = {'ix': ix, 'axis': -1}
    output_core_dims = [('__temp_dim__',)]

    result = apply_ufunc(_broadcast_pchip, *args, kwargs=kwargs,
                         input_core_dims=input_core_dims,
                         output_core_dims=output_core_dims)
    result['__temp_dim__'] = ix
    return result.rename({'__temp_dim__': dim})


# --------------------------------------------------------------------------- #
#                           scipy signal filtering                            #
# --------------------------------------------------------------------------- #

@lazy_guvectorize([
    (int_[:], int_[:], int_[:], double[:], double[:]),
    (double[:], int_[:], int_[:], double[:], double[:]),
    (int_[:], double[:], int_[:], double[:], double[:]),
    (double[:], double[:], int_[:], double[:], double[:]),
], '(n),(n),(),()->(n)', cache=_NUMBA_CACHE_DEFAULT)
def _gufunc_filter_wiener(x, y, mysize, noise, out=None):  # pragma: no cover
    # Pre-process
    xynm = preprocess_interp1d_nan_func(x, y, out)
    if xynm is None:  # all nan
        return
    x, y, x_even, y_even, num_nan, mask = xynm

    # filter even data
    yf_even = signal.wiener(y_even, mysize=mysize, noise=noise)

    # Post-process
    postprocess_interp1d_nan_func(x, x_even, yf_even, num_nan, mask, out)


def _broadcast_filter_wiener(x, y, mysize=5, noise=1e-2, axis=-1):
    if axis != -1:
        y = y.swapaxes(axis, -1)
    return _gufunc_filter_wiener(x, y, mysize, noise)


def xr_filter_wiener(obj, dim, mysize=5, noise=1e-2):
    kwargs = {'mysize': mysize, 'noise': noise, 'axis': -1}
    input_core_dims = [(dim,), (dim,)]
    output_core_dims = [(dim,)]
    args = (obj[dim], obj)
    return apply_ufunc(_broadcast_filter_wiener, *args,
                       input_core_dims=input_core_dims,
                       output_core_dims=output_core_dims,
                       kwargs=kwargs)


@lazy_guvectorize([
    (int_[:], int_[:], int_[:], double[:], double[:]),
    (double[:], int_[:], int_[:], double[:], double[:]),
    (int_[:], double[:], int_[:], double[:], double[:]),
    (double[:], double[:], int_[:], double[:], double[:]),
], '(n),(n),(),()->(n)', cache=_NUMBA_CACHE_DEFAULT)
def _gufunc_filtfilt_butter(x, y, N, Wn, out=None):  # pragma: no cover
    # Pre-process
    xynm = preprocess_interp1d_nan_func(x, y, out)
    if xynm is None:  # all nan
        return
    x, y, x_even, y_even, num_nan, mask = xynm

    # filter function
    b, a = signal.butter(N=N[0], Wn=Wn[0])
    # filter even data
    yf_even = signal.filtfilt(b, a, y_even, method='gust')

    # Post-process
    postprocess_interp1d_nan_func(x, x_even, yf_even, num_nan, mask, out)


def _broadcast_filtfilt_butter(x, y, N=2, Wn=0.4, axis=-1):
    if axis != -1:
        y = y.swapaxes(axis, -1)
    return _gufunc_filtfilt_butter(x, y, N, Wn)


def xr_filtfilt_butter(obj, dim, N=2, Wn=0.4):
    """Filter (with forward and backward pass) data along ``dim`` using
    the butterworth design :py:func:`scipy.signal.butter`.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        The object to apply signal filtering to.
    dim : str
        The dimension to filter along.
    N : int, optional
        The order of the filter.
    Wn : scalar, optional
        Critical frequency.
    """
    kwargs = {'N': N, 'Wn': Wn, 'axis': -1}
    input_core_dims = [(dim,), (dim,)]
    output_core_dims = [(dim,)]
    args = (obj[dim], obj)
    return apply_ufunc(_broadcast_filtfilt_butter, *args,
                       input_core_dims=input_core_dims,
                       output_core_dims=output_core_dims,
                       kwargs=kwargs)


@lazy_guvectorize([
    (int_[:], int_[:], int_[:], double[:], double[:]),
    (double[:], int_[:], int_[:], double[:], double[:]),
    (int_[:], double[:], int_[:], double[:], double[:]),
    (double[:], double[:], int_[:], double[:], double[:]),
], '(n),(n),(),()->(n)', cache=_NUMBA_CACHE_DEFAULT)
def _gufunc_filtfilt_bessel(x, y, N, Wn, out=None):  # pragma: no cover
    # Pre-process
    xynm = preprocess_interp1d_nan_func(x, y, out)
    if xynm is None:  # all nan
        return
    x, y, x_even, y_even, num_nan, mask = xynm

    # filter function
    b, a = signal.bessel(N=N[0], Wn=Wn[0])
    # filter even data
    yf_even = signal.filtfilt(b, a, y_even, method='gust')

    # Post-process
    postprocess_interp1d_nan_func(x, x_even, yf_even, num_nan, mask, out)


def _broadcast_filtfilt_bessel(x, y, N=2, Wn=0.4, axis=-1):
    if axis != -1:
        y = y.swapaxes(axis, -1)
    return _gufunc_filtfilt_bessel(x, y, N, Wn)


def xr_filtfilt_bessel(obj, dim, N=2, Wn=0.4):
    """Filter (with forward and backward pass) data along ``dim`` using
    the bessel design :py:func:`scipy.signal.bessel`.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        The object to apply signal filtering to.
    dim : str
        The dimension to filter along.
    N : int, optional
        The order of the filter.
    Wn : scalar, optional
        Critical frequency.
    """
    kwargs = {'N': N, 'Wn': Wn, 'axis': -1}
    input_core_dims = [(dim,), (dim,)]
    output_core_dims = [(dim,)]
    args = (obj[dim], obj)
    return apply_ufunc(_broadcast_filtfilt_bessel, *args,
                       input_core_dims=input_core_dims,
                       output_core_dims=output_core_dims,
                       kwargs=kwargs)


# --------------------------------------------------------------------------- #
#                               idxmax idxmin                                 #
# --------------------------------------------------------------------------- #

def _index_from_1d_array(array, indices):
    return array.take(indices)


def gufunc_idxmax(x, y, axis=None):
    import dask.array as da
    indx = x.argmax(axis=axis)
    func = functools.partial(_index_from_1d_array, y)

    if isinstance(x, da.Array):
        return da.map_blocks(func, indx, dtype=indx.dtype)
    else:
        return func(indx)


def xr_idxmax(obj, dim):
    """Find the coordinate of the maximum along ``dim``.

    Parameters
    ----------
    obj : xarray.DataArray or xarray.Dataset
        Object to find coordnate maximum in.
    dim : str
        Dimension along which to find maximum

    Returns
    -------
    new_xobj : xarray.DataArray or xarray.Dataset
    """
    input_core_dims = [(dim,), (dim,)]
    kwargs = {'axis': -1}
    allna = obj.isnull().all(dim)
    return apply_ufunc(gufunc_idxmax, obj.fillna(-np.inf), obj[dim],
                       input_core_dims=input_core_dims, kwargs=kwargs,
                       dask='allowed').where(~allna)


xr.DataArray.idxmax = xr_idxmax
xr.Dataset.idxmax = xr_idxmax


def gufunc_idxmin(x, y, axis=None):
    import dask.array as da
    indx = x.argmin(axis=axis)
    func = functools.partial(_index_from_1d_array, y)

    if isinstance(x, da.Array):
        return da.map_blocks(func, indx, dtype=indx.dtype)
    else:
        return func(indx)


def xr_idxmin(obj, dim):
    """Find the coordinate of the minimum along ``dim``.

    Parameters
    ----------
    obj : xarray.DataArray or xarray.Dataset
        Object to find coordnate maximum in.
    dim : str
        Dimension along which to find maximum

    Returns
    -------
    new_xobj : xarray.DataArray or xarray.Dataset
    """
    input_core_dims = [(dim,), (dim,)]
    kwargs = {'axis': -1}
    allna = obj.isnull().all(dim)
    return apply_ufunc(gufunc_idxmin, obj.fillna(np.inf), obj[dim],
                       input_core_dims=input_core_dims, kwargs=kwargs,
                       dask='allowed').where(~allna)


xr.DataArray.idxmin = xr_idxmin
xr.Dataset.idxmin = xr_idxmin


# --------------------------------------------------------------------------- #
#                      Univariate spline interpolation                        #
# --------------------------------------------------------------------------- #

@lazy_guvectorize([
    (int_[:], int_[:], double[:], int_[:], double[:]),
    (double[:], int_[:], double[:], int_[:], double[:]),
    (int_[:], double[:], double[:], int_[:], double[:]),
    (double[:], double[:], double[:], int_[:], double[:]),
], '(n),(n),(n),()->(n)', cache=_NUMBA_CACHE_DEFAULT)
def _gufunc_unispline_err(x, y, err, num_knots, out=None):  # pragma: no cover
    xi = x.min()
    xf = x.max()
    t = np.linspace(xi, xf, num_knots)[1:-1]
    fn_interp = LSQUnivariateSpline(x, y, t=t, w=1 / err)
    out[:] = fn_interp(x)


@lazy_guvectorize([
    (int_[:], int_[:], int_[:], double[:]),
    (double[:], int_[:], int_[:], double[:]),
    (int_[:], double[:], int_[:], double[:]),
    (double[:], double[:], int_[:], double[:]),
], '(n),(n),()->(n)', cache=_NUMBA_CACHE_DEFAULT)
def _gufunc_unispline_noerr(x, y, num_knots, out=None):  # pragma: no cover
    xi = x.min()
    xf = x.max()
    t = np.linspace(xi, xf, num_knots)[1:-1]
    fn_interp = LSQUnivariateSpline(x, y, t=t)
    out[:] = fn_interp(x)


@lazy_guvectorize([
    (int_[:], int_[:], double[:], int_[:], double[:], double[:]),
    (double[:], int_[:], double[:], int_[:], double[:], double[:]),
    (int_[:], double[:], double[:], int_[:], double[:], double[:]),
    (double[:], double[:], double[:], int_[:], double[:], double[:]),
], '(n),(n),(n),(),(m),(m)', cache=_NUMBA_CACHE_DEFAULT)
def _gufunc_unispline_err_upscale(x, y, err, num_knots,
                                  ix, out=None):  # pragma: no cover
    xi = x.min()
    xf = x.max()
    t = np.linspace(xi, xf, num_knots)[1:-1]
    fn_interp = LSQUnivariateSpline(x, y, t=t, w=1 / err)
    out[:] = fn_interp(ix)


@lazy_guvectorize([
    (int_[:], int_[:], int_[:], double[:], double[:]),
    (double[:], int_[:], int_[:], double[:], double[:]),
    (int_[:], double[:], int_[:], double[:], double[:]),
    (double[:], double[:], int_[:], double[:], double[:]),
], '(n),(n),(),(m),(m)', cache=_NUMBA_CACHE_DEFAULT, forceobj=True)
def _gufunc_unispline_noerr_upscale(x, y, num_knots,
                                    ix, out=None):  # pragma: no cover
    xi = x.min()
    xf = x.max()
    t = np.linspace(xi, xf, num_knots)[1:-1]
    fn_interp = LSQUnivariateSpline(x, y, t=t)
    out[:] = fn_interp(ix)


def _broadcast_unispline(x, y, err=None, num_knots=11, ix=None, axis=-1):
    """Dispatch to the correct numba gufunc.
    """
    if axis != -1:
        y = y.swapaxes(axis, -1)

    if ix is None:
        if err is None:
            return _gufunc_unispline_noerr(x, y, num_knots)
        return _gufunc_unispline_err(x, y, err, num_knots)

    # prepare interpolaing array to pass in
    if isinstance(ix, int):
        ix = np.linspace(x.min(), x.max(), ix)

    # prepare output data to pass in
    if axis < 0:
        axis += y.ndim
    out_shape = y.shape[:-1] + ix.shape
    out = np.empty(out_shape, dtype=x.dtype)

    if err is None:
        _gufunc_unispline_noerr_upscale(x, y, num_knots, ix, out)
    else:
        _gufunc_unispline_err_upscale(x, y, err, num_knots, ix, out)
    return out


def xr_unispline(obj, dim, err=None, num_knots=11, ix=None):
    """Fit a univariate spline along a dimension using linearly spaced knots.

    Parameters
    ----------
    obj : Dataset or DataArray
        Object to fit spline to.
    dim : str
        Dimension to fit spline along.
    err : DataArray or str (optional)
        Error in variables, with with to weight spline fitting. If `err`
        is a string, use the corresponding variable found within `obj`.
    num_knots : int (optional)
        Number of linearly spaced interior knots to form spline with,
        defaults to 11.
    ix : array-like or int (optional)
        Which points to evaluate the newly fitted spline. If int, `ix` many
        points will be chosen linearly spaced across the datas range. If
        None, spline will be evaluated at original coordinates.

    Returns
    -------
    new_xobj : xarray.DataArray or xarray.Dataset

    See Also
    --------
    xr_polyfit
    """
    if isinstance(err, str):
        err = obj[err]

    if ix is None:
        kwargs = {'num_knots': num_knots, 'axis': -1}
        if err is None:
            input_core_dims = [(dim,), (dim,)]
            output_core_dims = [(dim,)]
            args = (obj[dim], obj)
        else:
            input_core_dims = [(dim,), (dim,), (dim,)]
            output_core_dims = [(dim,)]
            args = (obj[dim], obj, err)
        return apply_ufunc(_broadcast_unispline, *args,
                           input_core_dims=input_core_dims,
                           output_core_dims=output_core_dims,
                           kwargs=kwargs)
    else:
        if isinstance(ix, int):
            ix = np.linspace(float(obj[dim].min()), float(obj[dim].max()), ix)
        kwargs = {'num_knots': num_knots, 'axis': -1, 'ix': ix}

        if err is None:
            input_core_dims = [(dim,), (dim,)]
            output_core_dims = [('__temp_dim__',)]
            args = (obj[dim], obj)
        else:
            input_core_dims = [(dim,), (dim,), (dim,)]
            output_core_dims = [('__temp_dim__',)]
            args = (obj[dim], obj, err)
        result = apply_ufunc(_broadcast_unispline, *args,
                             input_core_dims=input_core_dims,
                             output_core_dims=output_core_dims,
                             kwargs=kwargs)
        result['__temp_dim__'] = ix
        return result.rename({'__temp_dim__': dim})


# --------------------------------------------------------------------------- #
#                            polynomial fitting                               #
# --------------------------------------------------------------------------- #

@lazy_guvectorize([
    (int_[:], int_[:], int_[:], double[:]),
    (double[:], int_[:], int_[:], double[:]),
    (int_[:], double[:], int_[:], double[:]),
    (double[:], double[:], int_[:], double[:]),
], '(n),(n),()->(n)', cache=_NUMBA_CACHE_DEFAULT, forceobj=True)
def _gufunc_polyfit(x, y, deg, out=None):  # pragma: no cover
    xynm = preprocess_nan_func(x, y, out)
    if xynm is None:
        return
    x, y, num_nan, mask = xynm

    # interpolating function
    c = np.polynomial.polynomial.polyfit(x, y, deg[0])
    yf = np.polynomial.polynomial.polyval(x, c)
    postprocess_nan_func(yf, num_nan, mask, out)


@lazy_guvectorize([
    (int_[:], int_[:], double[:], int_[:], double[:]),
    (double[:], int_[:], double[:], int_[:], double[:]),
    (int_[:], double[:], double[:], int_[:], double[:]),
    (double[:], double[:], double[:], int_[:], double[:])
], '(n),(n),(m),()->(m)', cache=_NUMBA_CACHE_DEFAULT)
def _gufunc_polyfit_upscale(x, y, ix, deg, out=None):  # pragma: no cover
    xynm = preprocess_nan_func(x, y, out)
    if xynm is None:
        return
    x, y, _, mask = xynm

    # interpolating function
    c = np.polynomial.polynomial.polyfit(x, y, deg[0])
    out[:] = np.polynomial.polynomial.polyval(ix, c)


@lazy_guvectorize([
    (int_[:], int_[:], int_[:], double[:]),
    (double[:], int_[:], int_[:], double[:]),
    (int_[:], double[:], int_[:], double[:]),
    (double[:], double[:], int_[:], double[:]),
], '(n),(n),()->(n)', cache=_NUMBA_CACHE_DEFAULT)
def _gufunc_chebfit(x, y, deg, out=None):  # pragma: no cover
    xynm = preprocess_nan_func(x, y, out)
    if xynm is None:
        return
    x, y, num_nan, mask = xynm

    # interpolating function
    c = np.polynomial.chebyshev.chebfit(x, y, deg[0])
    yf = np.polynomial.chebyshev.chebval(x, c)
    postprocess_nan_func(yf, num_nan, mask, out)


@lazy_guvectorize([
    (int_[:], int_[:], double[:], int_[:], double[:]),
    (double[:], int_[:], double[:], int_[:], double[:]),
    (int_[:], double[:], double[:], int_[:], double[:]),
    (double[:], double[:], double[:], int_[:], double[:])
], '(n),(n),(m),()->(m)', cache=_NUMBA_CACHE_DEFAULT)
def _gufunc_chebfit_upscale(x, y, ix, deg, out=None):  # pragma: no cover
    xynm = preprocess_nan_func(x, y, out)
    if xynm is None:
        return
    x, y, _, mask = xynm

    # interpolating function
    c = np.polynomial.chebyshev.chebfit(x, y, deg[0])
    out[:] = np.polynomial.chebyshev.chebval(ix, c)


@lazy_guvectorize([
    (int_[:], int_[:], int_[:], double[:]),
    (double[:], int_[:], int_[:], double[:]),
    (int_[:], double[:], int_[:], double[:]),
    (double[:], double[:], int_[:], double[:]),
], '(n),(n),()->(n)', cache=_NUMBA_CACHE_DEFAULT)
def _gufunc_legfit(x, y, deg, out=None):  # pragma: no cover
    xynm = preprocess_nan_func(x, y, out)
    if xynm is None:
        return
    x, y, num_nan, mask = xynm

    # interpolating function
    c = np.polynomial.legendre.legfit(x, y, deg[0])
    yf = np.polynomial.legendre.legval(x, c)
    postprocess_nan_func(yf, num_nan, mask, out)


@lazy_guvectorize([
    (int_[:], int_[:], double[:], int_[:], double[:]),
    (double[:], int_[:], double[:], int_[:], double[:]),
    (int_[:], double[:], double[:], int_[:], double[:]),
    (double[:], double[:], double[:], int_[:], double[:])
], '(n),(n),(m),()->(m)', cache=_NUMBA_CACHE_DEFAULT)
def _gufunc_legfit_upscale(x, y, ix, deg, out=None):  # pragma: no cover
    xynm = preprocess_nan_func(x, y, out)
    if xynm is None:
        return
    x, y, _, mask = xynm

    # interpolating function
    c = np.polynomial.legendre.legfit(x, y, deg[0])
    out[:] = np.polynomial.legendre.legval(ix, c)


@lazy_guvectorize([
    (int_[:], int_[:], int_[:], double[:]),
    (double[:], int_[:], int_[:], double[:]),
    (int_[:], double[:], int_[:], double[:]),
    (double[:], double[:], int_[:], double[:]),
], '(n),(n),()->(n)', cache=_NUMBA_CACHE_DEFAULT)
def _gufunc_lagfit(x, y, deg, out=None):  # pragma: no cover
    xynm = preprocess_nan_func(x, y, out)
    if xynm is None:
        return
    x, y, num_nan, mask = xynm

    # interpolating function
    c = np.polynomial.laguerre.lagfit(x, y, deg[0])
    yf = np.polynomial.laguerre.lagval(x, c)
    postprocess_nan_func(yf, num_nan, mask, out)


@lazy_guvectorize([
    (int_[:], int_[:], double[:], int_[:], double[:]),
    (double[:], int_[:], double[:], int_[:], double[:]),
    (int_[:], double[:], double[:], int_[:], double[:]),
    (double[:], double[:], double[:], int_[:], double[:])
], '(n),(n),(m),()->(m)', cache=_NUMBA_CACHE_DEFAULT)
def _gufunc_lagfit_upscale(x, y, ix, deg, out=None):  # pragma: no cover
    xynm = preprocess_nan_func(x, y, out)
    if xynm is None:
        return
    x, y, _, mask = xynm

    # interpolating function
    c = np.polynomial.laguerre.lagfit(x, y, deg[0])
    out[:] = np.polynomial.laguerre.lagval(ix, c)


@lazy_guvectorize([
    (int_[:], int_[:], int_[:], double[:]),
    (double[:], int_[:], int_[:], double[:]),
    (int_[:], double[:], int_[:], double[:]),
    (double[:], double[:], int_[:], double[:]),
], '(n),(n),()->(n)', cache=_NUMBA_CACHE_DEFAULT)
def _gufunc_hermfit(x, y, deg, out=None):  # pragma: no cover
    xynm = preprocess_nan_func(x, y, out)
    if xynm is None:
        return
    x, y, num_nan, mask = xynm

    # interpolating function
    c = np.polynomial.hermite.hermfit(x, y, deg[0])
    yf = np.polynomial.hermite.hermval(x, c)
    postprocess_nan_func(yf, num_nan, mask, out)


@lazy_guvectorize([
    (int_[:], int_[:], double[:], int_[:], double[:]),
    (double[:], int_[:], double[:], int_[:], double[:]),
    (int_[:], double[:], double[:], int_[:], double[:]),
    (double[:], double[:], double[:], int_[:], double[:])
], '(n),(n),(m),()->(m)', cache=_NUMBA_CACHE_DEFAULT)
def _gufunc_hermfit_upscale(x, y, ix, deg, out=None):  # pragma: no cover
    xynm = preprocess_nan_func(x, y, out)
    if xynm is None:
        return
    x, y, _, mask = xynm

    # interpolating function
    c = np.polynomial.hermite.hermfit(x, y, deg[0])
    out[:] = np.polynomial.hermite.hermval(ix, c)


POLY_FNS = {
    'polynomial': (_gufunc_polyfit, _gufunc_polyfit_upscale),
    'chebyshev': (_gufunc_chebfit, _gufunc_chebfit_upscale),
    'legendre': (_gufunc_legfit, _gufunc_legfit_upscale),
    'laguerre': (_gufunc_lagfit, _gufunc_lagfit_upscale),
    'hermite': (_gufunc_hermfit, _gufunc_hermfit_upscale),
}


def _broadcast_polyfit(x, y, ix=None, deg=0.5, poly='hermite', axis=-1):
    """Parse arguments and dispatch to the correct function.
    """
    if axis != -1:
        y = y.swapaxes(axis, -1)

    if isinstance(deg, float):
        deg = int(deg * x.size)

    gfn, gfn_upscale = POLY_FNS[poly]

    if ix is None:
        return gfn(x, y, deg)

    if isinstance(ix, int):
        # automatic upscale
        ix = np.linspace(x.min(), x.max(), ix)

    return gfn_upscale(x, y, ix, deg)


def xr_polyfit(obj, dim, ix=None, deg=0.5, poly='hermite'):
    """Fit a polynomial of degree ``deg`` using least-squares along ``dim``.

    Parameters
    ----------
    obj : xarray.Dataset or xarray.DataArray
        The object to fit.
    dim : str, optional
        The dimension to fit along.
    ix : {None, int, array_like}, optional
        If ``None``, interpolate the polynomial at the original x points.
        If ``int``, linearly space this many points along the range of the
        original data and interpolate with these.
        If array-like, interpolate at these given points.
    deg : int or float, optional
        The degree of the polynomial to fit. Used directly if integer. If float
        supplied, with ``0.0 < deg < 1.0``, the proportion of the total
        possible degree to use.
    poly : {'chebyshev', 'polynomial', 'legendre',
            'laguerre', hermite}, optional
        The type of polynomial to fit.

    Returns
    -------
    new_xobj : xarray.DataArray or xarray.Dataset

    See Also
    --------
    xr_unispline
    """
    input_core_dims = [(dim,), (dim,)]
    args = (obj[dim], obj)

    if ix is None:
        kwargs = {'ix': ix, 'axis': -1, 'deg': deg, 'poly': poly}
        output_core_dims = [(dim,)]
        return apply_ufunc(_broadcast_polyfit, *args, kwargs=kwargs,
                           input_core_dims=input_core_dims,
                           output_core_dims=output_core_dims)

    if isinstance(ix, int):
        ix = np.linspace(float(obj[dim].min()), float(obj[dim].max()), ix)

    kwargs = {'ix': ix, 'axis': -1, 'deg': deg, 'poly': poly}
    output_core_dims = [('__temp_dim__',)]

    result = apply_ufunc(_broadcast_polyfit, *args, kwargs=kwargs,
                         input_core_dims=input_core_dims,
                         output_core_dims=output_core_dims)
    result['__temp_dim__'] = ix
    return result.rename({'__temp_dim__': dim})
