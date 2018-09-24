"""Utility functions.
"""
import functools
import operator
import itertools
import time
import math
import sys

import tqdm
from cytoolz import isiterable
import numpy as np

from .signal import jitclass, double, int_


def prod(it):
    """Product of an iterable.
    """
    return functools.reduce(operator.mul, it)


def unzip(its, zip_level=1):
    """Split a nested iterable at a specified level, i.e. in numpy language
    transpose the specified 'axis' to be the first.

    Parameters
    ----------
    its: iterable (of iterables (of iterables ...))
        'n-dimensional' iterable to split
    zip_level: int
        level at which to split the iterable, default of 1 replicates
        ``zip(*its)`` behaviour.

    Example
    -------
    >>> x = [[(1, True), (2, False), (3, True)],
             [(7, True), (8, False), (9, True)]]
    >>> nums, bools = unzip(x, 2)
    >>> nums
    ((1, 2, 3), (7, 8, 9))
    >>> bools
    ((True, False, True), (True, False, True))

    """
    def _unzipper(its, zip_level):
        if zip_level > 1:
            return (zip(*_unzipper(it, zip_level - 1)) for it in its)
        else:
            return its

    return zip(*_unzipper(its, zip_level)) if zip_level else its


def flatten(its, n):
    """Take the n-dimensional nested iterable its and flatten it.

    Parameters
    ----------
        its : nested iterable
        n : number of dimensions

    Returns
    -------
        flattened iterable of all items
    """
    if n > 1:
        return itertools.chain(*(flatten(it, n - 1) for it in its))
    else:
        return its


def _get_fn_name(fn):
    """Try to inspect a function's name, taking into account several common
    non-standard types of function: dask, functools.partial ...
    """
    if hasattr(fn, "__name__"):
        return fn.__name__
    # try dask delayed function with key
    elif hasattr(fn, "key"):
        return fn.key.partition('-')[0]
    # try functools.partial function syntax
    elif hasattr(fn, "func"):
        return fn.func.__name__
    else:
        raise ValueError("Could not extract function name from {}".format(fn))


def progbar(it=None, nb=False, **kwargs):
    """Turn any iterable into a progress bar, with notebook option

    Parameters
    ----------
        it: iterable
            Iterable to wrap with progress bar
        nb: bool
            Whether  to display the notebook progress bar
        **kwargs: dict-like
            additional options to send to tqdm
    """
    defaults = {'ascii': True, 'smoothing': 0.0}
    # Overide defaults with custom kwargs
    settings = {**defaults, **kwargs}
    if nb:  # pragma: no cover
        return tqdm.tqdm_notebook(it, **settings)
    return tqdm.tqdm(it, **settings)


class Timer:
    """A simple context manager class for timing blocks.
    """

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.t = self.time = self.interval = self.end - self.start


def _auto_min_time(timer, min_t=0.2, repeats=5, get='min'):
    tot_t = 0
    number = 1

    while True:
        tot_t = timer.timeit(number)
        if tot_t > min_t:
            break
        number *= 2

    results = [tot_t] + timer.repeat(repeats - 1, number)

    if get == 'mean':
        return sum(results) / (number * len(results))

    return min(t / number for t in results)


def benchmark(fn, setup=None, n=None, min_t=0.2,
              repeats=5, get='min', starmap=False):
    """Benchmark the time it takes to run ``fn``.

    Parameters
    ----------
    fn : callable
        The function to time.
    setup : callable, optional
        If supplied the function that sets up the argument for ``fn``.
    n : int, optional
        If supplied, the integer to supply to ``setup`` of ``fn``.
    min_t : float, optional
        Aim to repeat function enough times to take up this many seconds.
    repeats : int, optional
        Repeat the whole procedure (with setup) this many times in order to
        take the minimum run time.
    get : {'min', 'mean'}, optional
        Return the minimum or mean time for each run.
    starmap : bool, optional
        Unpack the arguments from ``setup``, if given.

    Returns
    -------
    t : float
        The minimum, averaged, time to run ``fn`` in seconds.
    """
    from timeit import Timer

    if n is None:
        n = ""

    if setup is None:
        setup_str = ""
        stmnt_str = "fn({})".format(n)
    else:
        setup_str = "X=setup({})".format(n)
        stmnt_str = "fn(*X)" if starmap else "fn(X)"

    timer = Timer(setup=setup_str, stmt=stmnt_str,
                  globals={'setup': setup, 'fn': fn})

    return _auto_min_time(timer, min_t=min_t, repeats=repeats, get=get)


class Benchmarker:
    """Compare the performance of various ``kernels``. Internally this makes
    use of :func:`~xyzpy.benchmark`, :func:`~xyzpy.Harvester` and xyzpys
    plotting functionality.

    Parameters
    ----------
    kernels : sequence of callable
        The functions to compare performance with.
    setup : callable, optional
        If given, setup each benchmark run by suppling the size argument ``n``
        to this function first, then feeding its output to each of the
        functions.
    names : sequence of str, optional
        Alternate names to give the function, else they will be inferred.
    benchmark_opts : dict, optional
        Supplied to :func:`~xyzpy.benchmark`.
    data_name : str, optional
        If given, the file name the internal harvester will use to store
        results persistently.

    Attributes
    ----------
    harvester : xyz.Harvester
        The harvester that runs and accumulates all the data.
    ds : xarray.Dataset
        Shortcut to the harvester's full dataset.
    """

    def __init__(self, kernels, setup=None, names=None,
                 benchmark_opts=None, data_name=None):
        import xyzpy as xyz

        self.kernels = kernels
        self.names = [f.__name__ for f in kernels] if names is None else names
        self.setup = setup
        self.benchmark_opts = {} if benchmark_opts is None else benchmark_opts

        def time(n, kernel):
            fn = self.kernels[self.names.index(kernel)]
            return xyz.benchmark(fn, self.setup, n, **self.benchmark_opts)

        self.runner = xyz.Runner(time, ['time'])
        self.harvester = xyz.Harvester(self.runner, data_name=data_name)

    def run(self, ns, kernels=None, **harvest_opts):
        """Run the benchmarks. Each run accumulates rather than overwriting the
        results.

        Parameters
        ----------
        ns : sequence of int or int
            The sizes to run the benchmarks with.
        kernels : sequence of str, optional
            If given, only run the kernels with these names.
        harvest_opts
            Supplied to :meth:`~xyzpy.Harvester.harvest_combos`.
        """
        if not isiterable(ns):
            ns = (ns,)

        if kernels is None:
            kernels = self.names

        combos = {'n': ns, 'kernel': kernels}
        self.harvester.harvest_combos(combos, **harvest_opts)

    @property
    def ds(self):
        return self.harvester.full_ds

    def lineplot(self, **plot_opts):
        """Plot the benchmarking results.
        """
        plot_opts.setdefault('xlog', True)
        plot_opts.setdefault('ylog', True)
        return self.ds.xyz.lineplot('n', 'time', 'kernel', **plot_opts)

    def ilineplot(self, **plot_opts):
        """Interactively plot the benchmarking results.
        """
        plot_opts.setdefault('xlog', True)
        plot_opts.setdefault('ylog', True)
        return self.ds.xyz.ilineplot('n', 'time', 'kernel', **plot_opts)


@jitclass([
    ('count', int_),
    ('mean', double),
    ('M2', double),
])
class RunningStatistics:  # pragma: no cover
    """Numba-compiled running mean & standard deviation using Welford's
    algorithm. This is a very efficient way of keeping track of the error on
    the mean for example.

    Attributes
    ----------
    mean : float
        Current mean.
    count : int
        Current count.
    std : float
        Current standard deviation.
    var : float
        Current variance.
    err : float
        Current error on the mean.
    rel_err: float
        The current relative error.

    Examples
    --------

        >>> rs = RunningStatistics()
        >>> rs.update(1.1)
        >>> rs.update(1.4)
        >>> rs.update(1.2)
        >>> rs.update_from_it([1.5, 1.3, 1.6])
        >>> rs.mean
        1.3499999046325684

        >>> rs.std  # standard deviation
        0.17078252585383266

        >>> rs.err  # error on the mean
        0.06972167422092768

    """

    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x):
        """Add a single value ``x`` to the statistics.
        """
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def update_from_it(self, xs):
        """Add all values from iterable ``xs`` to the statistics.
        """
        for x in xs:
            self.update(x)

    def converged(self, rtol, atol):
        """Check if the stats have converged with respect to relative and
        absolute tolerance ``rtol`` and ``atol``.
        """
        return self.err < rtol * abs(self.mean) + atol

    @property
    def var(self):
        if self.count == 0:
            return np.inf
        return self.M2 / self.count

    @property
    def std(self):
        if self.count == 0:
            return np.inf
        return self.var**0.5

    @property
    def err(self):
        if self.count == 0:
            return np.inf
        return self.std / self.count**0.5

    @property
    def rel_err(self):
        if self.count == 0:
            return np.inf
        return self.err / abs(self.mean)


def estimate_from_repeats(fn, *fn_args, rtol=0.02, tol_scale=1.0, get='stats',
                          verbosity=0, min_samples=5, max_samples=1000000,
                          **fn_kwargs):
    """
    Parameters
    ----------
    fn : callable
        The function that estimates a single value.
    fn_args, optional
        Supplied to ``fn``.
    rtol : float, optional
        Relative tolerance for error on mean.
    tol_scale, optional
        The expected 'scale' of the estimate, this modifies the aboslute
        tolerance near zero to ``rtol * tol_scale``, default: 1.0.
    get : {'stats', 'samples', 'mean'}, optional
        Just get the ``RunningStatistics`` object, or the actual samples too,
        or just the actual mean estimate.
    verbosity : { 0, 1, 2}, optional
        How much informatino to show: ``0``: nothing, ``1``: progress bar just
        with iteration rate, ``2``: progress bar with running stats displayed.
    min_samples : int, optional
        Take at least this many samples before checking for convergence.
    max_samples : int, optional
        Take at maximum this many samples.
    fn_kwargs, optional
        Supplied to ``fn``.

    Returns
    -------
    rs : RunningStatistics
        Statistics about the random estimation.
    samples : list[float]
        If ``get=='samples'``, the actual samples.
    """

    rs = RunningStatistics()
    repeats = itertools.count()

    if verbosity >= 1:
        repeats = progbar(repeats)
        prec = abs(round(math.log10(tol_scale * rtol))) + 1

    if get == 'samples':
        xs = []

    try:
        for i in repeats:
            x = fn(*fn_args, **fn_kwargs)
            if get == 'samples':
                xs.append(x)
            rs.update(x)

            if verbosity >= 2:
                desc = '{}: mean: {:.{prec}f} err: {:.{prec}f}'
                repeats.set_description(
                    desc.format(rs.count, rs.mean, rs.err, prec=prec))

            # need at least min_samples to check convergence
            if (i > min_samples):
                if rs.converged(rtol, tol_scale * rtol):
                    break

            # reached the maximum number of samples to try
            if i >= max_samples - 1:
                break

    # allow user to cleanly interupt sampling with keyboard
    except KeyboardInterrupt:
        pass

    if verbosity >= 1:
        repeats.close()
        sys.stderr.flush()
        print("<RunningStatistics(mean={:.{prec}f}, err={:.{prec}f}, "
              "count={})>".format(rs.mean, rs.err, rs.count, prec=prec))

    if get == 'samples':
        return rs, xs

    if get == 'mean':
        return rs.mean

    return rs
