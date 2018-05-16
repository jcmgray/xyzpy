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
        self.interval = self.end - self.start
