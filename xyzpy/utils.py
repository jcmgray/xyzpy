"""Utility functions.
"""
import functools
import operator
import itertools
import tqdm
import time


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


def _auto_min_time(timer, min_t=0.2, repeats=5):
    tot_t = 0
    number = 1

    while True:
        tot_t = timer.timeit(number)
        if tot_t > min_t:
            break
        number *= 2

    results = [tot_t] + timer.repeat(repeats - 1, number)

    return min(t / number for t in results)


def benchmark(fn, setup=None, n=None, min_t=0.2, repeats=5):
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

    return _auto_min_time(timer, min_t=min_t, repeats=repeats)
