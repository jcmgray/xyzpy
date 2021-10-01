"""Utility functions.
"""
import functools
import operator
import itertools
import time
import math
import sys
from collections.abc import Iterable

import tqdm
import numpy as np


class XYZError(Exception):
    pass


def isiterable(obj):
    return isinstance(obj, Iterable)


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


def getsizeof(obj):
    """Compute the real size of a python object. Taken from

    https://stackoverflow.com/a/30316760/5640201
    """
    import sys
    from types import ModuleType, FunctionType
    from gc import get_referents

    # Custom objects know their class.
    # Function objects seem to know way too much, including modules.
    # Exclude modules as well.
    excluded = type, ModuleType, FunctionType
    if isinstance(obj, excluded):
        raise TypeError(
            'getsize() does not take argument of type: {}'.format(type(obj)))

    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, excluded) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size


class Timer:
    """A very simple context manager class for timing blocks.

    Examples
    --------

    >>> from xyzpy import Timer
    >>> with Timer() as timer:
    ...     print('Doing some work!')
    ...
    Doing some work!
    >>> timer.t
    0.00010752677917480469

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


def benchmark(fn, setup=None, n=None, min_t=0.1,
              repeats=3, get='min', starmap=False):
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

    Examples
    --------

    Just a parameter-less function:

        >>> import xyzpy as xyz
        >>> import numpy as np
        >>> xyz.benchmark(lambda: np.linalg.eig(np.random.randn(100, 100)))
        0.004726233000837965

    The same but with a setup and size parameter ``n`` specified:

        >>> setup = lambda n: np.random.randn(n, n)
        >>> fn = lambda X: np.linalg.eig(X)
        >>> xyz.benchmark(fn, setup, 100)
        0.0042192734545096755
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


class RunningStatistics:  # pragma: no cover
    """Running mean & standard deviation using Welford's
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


class RunningCovariance:  # pragma: no cover
    """Running covariance class.
    """

    def __init__(self):
        self.count = 0
        self.xmean = 0.0
        self.ymean = 0.0
        self.C = 0.0

    def update(self, x, y):
        self.count += 1
        dx = x - self.xmean
        dy = y - self.ymean
        self.xmean += dx / self.count
        self.ymean += dy / self.count
        self.C += dx * (y - self.ymean)

    def update_from_it(self, xs, ys):
        for x, y in zip(xs, ys):
            self.update(x, y)

    @property
    def covar(self):
        """The covariance.
        """
        return self.C / self.count

    @property
    def sample_covar(self):
        """The covariance with "Bessel's correction".
        """
        return self.C / (self.count - 1)


class RunningCovarianceMatrix:

    def __init__(self, n=2):
        self.n = n
        self.rcs = {}
        for i in range(self.n):
            for j in range(i, self.n):
                self.rcs[i, j] = RunningCovariance()

    def update(self, *x):
        for i in range(self.n):
            for j in range(i, self.n):
                self.rcs[i, j].update(x[i], x[j])

    def update_from_it(self, *xs):
        for i in range(self.n):
            for j in range(i, self.n):
                self.rcs[i, j].update_from_it(xs[i], xs[j])

    @property
    def count(self):
        return self.rcs[0, 0].count

    @property
    def covar_matrix(self):
        covar_matrix = np.empty((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if j >= i:
                    covar_matrix[i, j] = self.rcs[i, j].covar
                else:
                    covar_matrix[i, j] = self.rcs[j, i].covar
        return covar_matrix

    @property
    def sample_covar_matrix(self):
        covar_matrix = np.empty((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if j >= i:
                    covar_matrix[i, j] = self.rcs[i, j].sample_covar
                else:
                    covar_matrix[i, j] = self.rcs[j, i].sample_covar
        return covar_matrix

    def to_uncertainties(self, bias=True):
        """Convert the accumulated statistics to correlated uncertainties,
        from which new quantities can be calculated with error automatically
        propagated.

        Parameters
        ----------
        bias : bool, optional
            If False, use the sample covariance with "Bessel's correction".

        Return
        ------
        values : tuple of uncertainties.ufloat
            The sequence of correlated variables.

        Examples
        --------

        Estimate quantities of two perfectly correlated sequences.

            >>> rcm = xyz.RunningCovarianceMatrix()
            >>> rcm.update_from_it((1, 3, 2), (2, 6, 4))
            >>> x, y = rcm.to_uncertainties(rcm)

        Calculated quantities like sums have the error propagated:

            >>> x + y
            6.0+/-2.4494897427831783

        But the covariance is also taken into account, meaning the ratio here
        can be estimated with zero error:

            >>> x / y
            0.5+/-0

        """
        import uncertainties

        means = [self.rcs[i, i].xmean for i in range(self.n)]
        if bias:
            covar = self.covar_matrix
        else:
            covar = self.sample_covar_matrix

        return uncertainties.correlated_values(means, covar)


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
    tol_scale : float, optional
        The expected 'scale' of the estimate, this modifies the aboslute
        tolerance near zero to ``rtol * tol_scale``, default: 1.0.
    get : {'stats', 'samples', 'mean'}, optional
        Just get the ``RunningStatistics`` object, or the actual samples too,
        or just the actual mean estimate.
    verbosity : { 0, 1, 2}, optional
        How much information to show:

        - ``0``: nothing
        - ``1``: progress bar just with iteration rate
        - ``2``: progress bar with running stats displayed.

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


    Examples
    --------

    Estimate the sum of ``n`` random numbers:

        >>> import numpy as np
        >>> import xyzpy as xyz
        >>> def fn(n):
        ...     return np.random.rand(n).sum()
        ...
        >>> stats = xyz.estimate_from_repeats(fn, n=10, verbosity=3)
        51: mean: 5.002 err: 0.119: : 0it [00:00, ?it/s]
        <RunningStatistics(mean=5.002, err=0.119, count=51)>

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

    finally:
        if verbosity >= 1:
            repeats.close()

    if verbosity >= 1:
        sys.stderr.flush()
        print("<RunningStatistics(mean={:.{prec}f}, err={:.{prec}f}, "
              "count={})>".format(rs.mean, rs.err, rs.count, prec=prec))

    if get == 'samples':
        return rs, xs

    if get == 'mean':
        return rs.mean

    return rs
