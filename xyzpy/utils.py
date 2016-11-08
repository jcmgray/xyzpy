"""Utility functions.
"""
import functools
import operator
import inspect

import tqdm


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
            zip(*its) behaviour.

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


def _get_fn_name(fn):
    """Try to inspect a function's name, taking into account several common
    non-standard types of function: dask, functools.partial ...
    """
    try:
        return fn.__name__
    except AttributeError:
        try:  # try dask delayed function with key
            return fn.key.partition('-')[0]
        except AttributeError:  # try functools.partial function syntax
            return fn.func.__name__


def _get_fn_args(fn):
    """Try to inspect a function's arguments.
    """
    sig = inspect.signature(fn)
    return tuple(sig.parameters)


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


def update_upon_eval(fn, pbar):
    """Decorate `fn` such that every time it is called, `pbar` is updated
    """
    def new_fn(*args, **kwargs):
        pbar.update()
        return fn(*args, **kwargs)
    return new_fn
