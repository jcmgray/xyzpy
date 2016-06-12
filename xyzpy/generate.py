""" Generate datasets from function and parameter lists """

# TODO: find NaNs in xarray and perform cases

from functools import partial
from itertools import product

from multiprocessing import Pool

import numpy as np
import xarray as xr
from tqdm import tqdm, tqdm_notebook


def progbar(it, nb=False, **kwargs):
    """
    Turn any iterable into a progress bar, with notebook version.
    """
    if nb:
        return tqdm_notebook(it, **kwargs)
    else:
        return tqdm(it, ascii=True, **kwargs)


def case_runner(fn, cases, output_dims=1, progbars=0,
                parallel=False, processes=None, progbar_opts={}):
    """
    Take a function fn and analyse it over all combinations of named
    variables' values, optionally showing progress and in parallel.

    Parameters
    ----------
        fn: function to analyse
        cases: list of tuples/dict of form ((variable_name, [values]), ...)
        output_dims: shape of the return value, here, only matters if list-like
            or not, i.e. whether to split into multiple output arrays or not.
        progbars: how many levels of nested progress bars to show
        parallel: process cases in parallel
        processes: how many processes to use, use None for automatic
        progbar_opts: dict of options for the progress bar

    Returns
    -------
        data: list of result arrays, each with all param combinations.
    """
    cases = tuple(cases.items() if isinstance(cases, dict) else cases)
    fn_args, _ = zip(*cases)
    multi_output = isinstance(output_dims, (tuple, list))

    if parallel or processes is not None:
        p = Pool(processes=processes)

        def submit_jobs(fn, cases, _l=0):
            arg, inputs = cases[0]
            for x in inputs:
                if len(cases) == 1:
                    yield p.apply_async(fn, kwds={arg: x})
                else:
                    sub_fn = partial(fn, **{arg: x})
                    yield tuple(submit_jobs(sub_fn, cases[1:], _l+1))

        def get_outputs(futs, _l=0):
            for fut in progbar(futs, disable=_l >= progbars,
                               desc=fn_args[_l], **progbar_opts):
                if _l < len(fn_args) - 1:
                    yield tuple(zip(*get_outputs(fut, _l+1)))
                else:
                    y = fut.get()
                    yield y if multi_output else [y]

        futures = tuple(submit_jobs(fn, cases))
        outputs = tuple(zip(*get_outputs(futures)))

    else:
        def sub_case_runner(fn, cases, _l=0):
            arg, inputs = cases[0]
            for x in progbar(inputs, disable=_l >= progbars,
                             desc=arg, **progbar_opts):
                if len(cases) == 1:
                    yield fn(**{arg: x}) if multi_output else [fn(**{arg: x})]
                else:
                    sub_fn = partial(fn, **{arg: x})
                    yield tuple(zip(*sub_case_runner(sub_fn, cases[1:], _l+1)))
        outputs = tuple(zip(*sub_case_runner(fn, cases)))

    return outputs if multi_output else outputs[0]


def sub_split(a, tolist=False):
    """ Split a multi-nested python list at the lowest dimension """
    return (b.astype(type(b.item(0)), copy=False).T
            for b in np.array(a, dtype=object, copy=False).T)


def numpy_case_runner(fn, cases, progbars=0, processes=None):
    """ Use numpy arrays to analyse function, now with progress bar

    Parameters
    ----------
        fn: function to evaluate
        cases: list of tuples [(parameter, [parameter values]), ... ]
            or dictionary

    Returns
    -------
        x: list of arrays, one for each return object of fn,
            each with ndim == len(cases) """

    fn_args, fn_inputs = zip(*(cases.items() if isinstance(cases, dict) else
                               cases))

    cfgs = product(*fn_inputs)
    shp_inputs = [len(inputs) for inputs in fn_inputs]
    coos = product(*(range(i) for i in shp_inputs))
    first_run = True
    p = Pool(processes=processes)
    fx = [p.apply_async(fn, kwds=dict(zip(fn_args, cfg))) for cfg in cfgs]
    for res, coo in progbar(zip(fx, coos),
                            total=np.prod(shp_inputs), disable=progbars < 1):
        res = res.get()
        if first_run:
            multires = isinstance(res, (tuple, list))
            if multires:
                x = [np.empty(shape=shp_inputs, dtype=type(y)) for y in res]
            else:
                x = np.empty(shape=shp_inputs, dtype=type(res))
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

def xr_case_runner(fn, cases, output_coords, **kwargs):
    """ Take a function fn and analyse it over all combinations of named
    variables values, optionally showing progress and outputing to xarray.

    Parameters
    ----------
        fn: function to analyse
        cases: list of tuples of form ((variable_name, [values]), ...)
        output_coords, name of dataset's main variable, i.e. the results of fn
        progbars: how many levels of nested progress bars to show

    Returns
    -------
        ds: xarray Dataset with appropirate coordinates. """
    cases = tuple(cases.items() if isinstance(cases, dict) else cases)
    data = case_runner(fn, cases, **kwargs)

    ds = xr.Dataset()
    for var, vals in cases:
        ds.coords[var] = [*vals]

    if isinstance(output_coords, (list, tuple)):
        for result_name, sdata in zip(output_coords, data):
            ds[result_name] = ([var for var, _ in cases], sdata)
    else:
        ds[output_coords] = ([var for var, _ in cases], data)
    return ds
