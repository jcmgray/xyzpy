"""Utility functions.
"""
import functools
import operator
import itertools
import tqdm
import time
from cytoolz import isiterable


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


def benchmark(fn, setup=None, n=None, min_t=0.2, repeats=5, get='min'):
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
        stmnt_str = "fn(X)"

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
        return self.ds.xyz.lineplot('n', 'time', 'kernel', **plot_opts)

    def ilineplot(self, **plot_opts):
        """Interactively plot the benchmarking results.
        """
        return self.ds.xyz.ilineplot('n', 'time', 'kernel', **plot_opts)
