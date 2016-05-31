""" Generate datasets from function and parameter lists """

# TODO: handle functions that return arrays
# TODO: single access function for param_runners
# TODO: nested inner param_runner function
# TODO: multiprocessing param_runner (for ordered arguements and single arg fn)

from functools import partial
from itertools import product
import numpy as np
import xarray as xr
from tqdm import tqdm


progbar = partial(tqdm, ascii=True)


def param_runner(foo, params, num_progbars=0, _nl=0):
    """ Take a function foo and analyse it over all combinations of named
    variables' values, optionally showing progress.

    Parameters
    ----------
        foo: function to analyse
        params: list of tuples of form ((variable_name, [values]), ...)
        num_progbars: how many levels of nested progress bars to show
        _nl: internal variable used for keeping track of nested level

    Returns
    -------
        data: generator for array (list of lists) of dimension len(params) """
    # TODO: automatic multiprocessing?
    # TODO: inner function with const num_progbars, external pre and post proc
    if _nl == 0:
        if isinstance(params, dict):
            params = params.items()
        params = [*params]

    pname, pvals = params[0]

    if _nl < num_progbars:
        pvals = progbar(pvals, desc=pname)

    for pval in pvals:
        if len(params) == 1:
            yield foo(**{pname: pval})
        else:
            pfoo = partial(foo, **{pname: pval})
            yield [*param_runner(pfoo, params[1:], num_progbars, _nl=_nl+1)]


def sub_split(a, tolist=False):
    """ Split a multi-nested python list at the lowest dimension """
    return (b.astype(type(b.item(0)), copy=False).T
            for b in np.array(a, dtype=object, copy=False).T)


def np_param_runner(foo, params):
    """ Use numpy.vectorize and meshgrid to evaluate a function
    at all combinations of params, may be faster than case_runner but no
    live progress can be shown.

    Parameters
    ----------
        foo: function to evaluate
        params: list of tuples [(parameter, [parameter values]), ... ]

    Returns
    -------
        x: list of arrays, one for each return object of foo,
            each with ndim == len(params)

    # TODO: progbar? """
    params = [*params.items()] if isinstance(params, dict) else params
    prm_names, prm_vals = zip(*params)
    vprm_vals = np.meshgrid(*prm_vals, sparse=True, indexing='ij')
    vfoo = np.vectorize(foo)
    return vfoo(**{n: vv for n, vv in zip(prm_names, vprm_vals)})


def np_param_runner2(foo, params, num_progbars=0):
    """ Use numpy.vectorize and meshgrid to evaluate a function
    at all combinations of params, now with progress bar

    Parameters
    ----------
        foo: function to evaluate
        params: list of tuples [(parameter, [parameter values]), ... ]

    Returns
    -------
        x: list of arrays, one for each return object of foo,
            each with ndim == len(params) """
    # TODO: multiprocess
    pnames, pvals = zip(*(params.items() if isinstance(params, dict) else
                          params))
    pszs = [len(pval) for pval in pvals]
    pcoos = [[*range(psz)] for psz in pszs]
    first_run = True
    configs = zip(product(*pvals), product(*pcoos))
    for config, coo in progbar(configs, total=np.prod(pszs),
                               disable=num_progbars < 1):
        res = foo(**{n: vv for n, vv in zip(pnames, config)})
        # Use first result to calculate output array
        if first_run:
            multires = isinstance(res, (tuple, list))
            x = (np.empty(shape=pszs, dtype=type(res)) if not multires else
                 [np.empty(shape=pszs, dtype=type(y)) for y in res])
            first_run = False
        if multires:
            for sx, y in zip(x, res):
                sx[coo] = y
        else:
            x[coo] = res
    return x


# -------------------------------------------------------------------------- #
# Convenience functions for working with xarray                              #
# -------------------------------------------------------------------------- #

def xr_param_runner(foo, params, result_names, num_progbars=-1, use_np=False):
    """ Take a function foo and analyse it over all combinations of named
    variables values, optionally showing progress and outputing to xarray.

    Parameters
    ----------
        foo: function to analyse
        params: list of tuples of form ((variable_name, [values]), ...)
        result_names, name of dataset's main variable, i.e. the results of foo
        num_progbars: how many levels of nested progress bars to show

    Returns
    -------
        ds: xarray Dataset with appropirate coordinates. """
    params = [*params.items()] if isinstance(params, dict) else params

    data = (np_param_runner(foo, params) if use_np else
            [*param_runner(foo, params, num_progbars=num_progbars)])

    ds = xr.Dataset()
    for var, vals in params:
        ds.coords[var] = [*vals]

    if isinstance(result_names, (list, tuple)):

        for result_name, sdata in zip(result_names,
                                      data if use_np else sub_split(data)):
            ds[result_name] = ([var for var, _ in params], sdata)
    else:
        ds[result_names] = ([var for var, _ in params], data)
    return ds
