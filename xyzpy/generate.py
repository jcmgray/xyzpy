""" Generate datasets from function and parameter lists """

# TODO: handle functions that return arrays, mixed with single results.
# TODO: single access function for case_runners

from multiprocessing import Pool
from functools import partial
from itertools import product
import numpy as np
import xarray as xr
from tqdm import tqdm


progbar = partial(tqdm, ascii=True)


def sub_split(a, tolist=False):
    """ Split a multi-nested python list at the lowest dimension """
    return (b.astype(type(b.item(0)), copy=False).T
            for b in np.array(a, dtype=object, copy=False).T)


def case_runner(foo, params, num_progbars=0, calc=True, split=False):
    """ Take a function foo and analyse it over all combinations of named
    variables' values, optionally showing progress.

    Parameters
    ----------
        foo: function to analyse
        params: list of tuples of form ((variable_name, [values]), ...)
        num_progbars: how many levels of nested progress bars to show
        calc: whether to calc the results or return a generator
        split: whether to split foo's outputs into multiple arrays

    Returns
    -------
        data: list of result arrays, each with all param combinations. """
    # TODO: automatic multiprocessing?

    def sub_case_runner(foo, pvs, l=0):
        p, vs = pvs[0]
        if l < num_progbars:
            vs = progbar(vs, desc=p)
        for v in vs:
            yield (foo(**{p: v}) if len(pvs) == 1 else
                   [*sub_case_runner(partial(foo, **{p: v}), pvs[1:], l+1)])

    params = [*params.items()] if isinstance(params, dict) else [*params]
    res = sub_case_runner(foo, params)
    if calc:
        res = [*res]
        if split:
            res = [*sub_split(res)]
        return res
    return res


def parallel_case_runner(foo, params, num_progbars=0):
    """ Use numpy arrays to analyse function, now with progress bar

    Parameters
    ----------
        foo: function to evaluate
        params: list of tuples [(parameter, [parameter values]), ... ]
            or dictionary

    Returns
    -------
        x: list of arrays, one for each return object of foo,
            each with ndim == len(params) """
    # TODO: BROKEN! Need to fix coords (concurrent futures?)
    _, pvals = zip(*(params.items() if isinstance(params, dict) else
                     params))

    pszs = [len(pval) for pval in pvals]
    pcoos = [[*range(psz)] for psz in pszs]
    first_run = True

    all_configs = product(*pvals)
    all_coords = product(*pcoos)

    with Pool() as p:
        fut_res = p.imap_unordered(foo, all_configs)

        for res, coo in progbar(zip(fut_res, all_coords),
                                total=np.prod(pszs),
                                disable=num_progbars < 1):
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

def xr_case_runner(foo, params, result_names, num_progbars=0, parallel=False):
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

    data = (parallel_case_runner(foo, params) if parallel else
            [*case_runner(foo, params, num_progbars=num_progbars)])

    ds = xr.Dataset()
    for var, vals in params:
        ds.coords[var] = [*vals]

    if isinstance(result_names, (list, tuple)):

        for result_name, sdata in zip(result_names,
                                      data if parallel else sub_split(data)):
            ds[result_name] = ([var for var, _ in params], sdata)
    else:
        ds[result_names] = ([var for var, _ in params], data)
    return ds
