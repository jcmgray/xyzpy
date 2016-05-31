""" Misc. functions """

# TODO: update timing functions script
# TODO: xarray spline resample

import numpy as np
from scipy.interpolate import splrep, splev, PchipInterpolator
import xarray as xr
from .generate import progbar
from .plot import ilineplot


def resample(x, y, n=100, **kwargs):
    ix = np.linspace(x[0], x[-1], n)
    iy = PchipInterpolator(x, y, **kwargs)(ix)
    return ix, iy


def spline_resample(x, y, n=100, **kwargs):
    ix = np.linspace(x[0], x[-1], n)
    iy = splev(ix, splrep(x, y, **kwargs))
    return ix, iy


def time_functions(funcs, func_names, setup_str, sig_str, ns, rep_func=None):
    """ Calculate and plot how a number of functions exec time scales

    Parameters
    ----------
        funcs: list of funcs
        func_names: list of function names
        setup_str: actions to perform before each function
        sig_str: how arguments from setup_str and given to funcs
        ns: range of sizes
        rep_func(n): function for computing the number of repeats

    Returns
    -------
        Plots time vs n for each function. """
    from timeit import Timer

    sz_n = len(ns)
    ts = np.zeros((sz_n, len(funcs)))

    def basic_scaling(n):
        return min(max(int(3 * (2**max(ns))/(2**n)), 1), 10000)

    rep_func = basic_scaling if rep_func is None else rep_func

    for i, n in progbar(enumerate(ns), total=sz_n):
        timers = [Timer(func.__name__+sig_str, 'n='+str(n)+';'+setup_str,
                        globals=globals()) for func in funcs]
        reps = rep_func(n)
        for j, timer in enumerate(timers):
            ts[i, j] = timer.timeit(number=reps)/reps

    ds = xr.Dataset()
    ds.coords['n'] = ns
    ds.coords['func'] = func_names
    ds['t'] = (('n', 'func'), ts)
    ilineplot(ds, 't', 'n', 'func', logy=True)
