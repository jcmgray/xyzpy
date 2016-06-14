"""
Generate datasets from function and parameter lists
"""
# TODO: find NaNs in xarray and perform cases

from functools import partial
from itertools import product, cycle

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


def case_runner(fn, cases, constants=None, split=False, progbars=0,
                parallel=False, processes=None, progbar_opts={}):
    """
    Take a function fn and analyse it over all combinations of named
    variables' values, optionally showing progress and in parallel.

    Parameters
    ----------
        fn: function to analyse
        cases: list of tuples/dict of form ((variable_name, [values]), ...),
            all combinations of each argument will be calculated. Each
            argument range thus gets a dimension in the output array(s).
        constants: list of tuples/dict of *constant* fn argument mappings.
        split: whether to split into multiple output arrays or not.
        progbars: how many levels of nested progress bars to show.
        parallel: process cases in parallel.
        processes: how many processes to use, use None for automatic.
        progbar_opts: dict of options for the progress bar.

    Returns
    -------
        data: list of result arrays, each with all param combinations.

    Examples
    --------
    In [1]: from xyzpy import *

    In [2]: from time import sleep

    In [3]: def foo(a, b, c):
       ...:     sleep(1)
       ...:     return a + b + c, c % 2 == 0
       ...:
       ...:

    In [4]: cases = (('a', [100, 400]), ('b', [20, 50]), ('c', [3, 6]))

    In [5]: case_runner(foo, cases, progbars=3)
    a: 100%|##########| 2/2 [00:08<00:00,  4.02s/it]
    b: 100%|##########| 2/2 [00:04<00:00,  2.01s/it]
    c: 100%|##########| 2/2 [00:02<00:00,  1.00s/it]

    Out[5]:
    ((((123, False), (126, True)), ((153, False), (156, True))),
     (((423, False), (426, True)), ((453, False), (456, True))))

    In [6]: case_runner(foo, cases, progbars=3, parallel=True)  # Faster!
    a: 100%|##########| 2/2 [00:01<00:00,  1.96it/s]
    b: 100%|##########| 2/2 [00:00<00:00, 327.58it/s]
    c: 100%|##########| 2/2 [00:00<00:00, 17512.75it/s]

    Out[6]:
    ((((123, False), (126, True)), ((153, False), (156, True))),
     (((423, False), (426, True)), ((453, False), (456, True))))


    In [7]: x, y = case_runner(foo, cases, split=True)

    In [8]: x
    Out[8]: (((123, 126), (153, 156)), ((423, 426), (453, 456)))

    In [9]: y
    Out[9]: (((False, True), (False, True)), ((False, True), (False, True)))

    Notes
    -----
        1. The parallel evaluation of cases relies on the `multiprocessing`
            module, this places some restrictions of what functions can be
            used (needs to be picklable/importable).
    """

    # Prepare cases
    if isinstance(cases, dict):
        cases = tuple(cases.items())
    elif isinstance(cases[0], str):
        cases = (cases,)
    else:
        cases = tuple(cases)
    fn_args, _ = zip(*cases)

    # Prepare Function
    if constants is not None:
        fn = partial(fn, **dict(constants))

    # Evaluate cases in parallel
    if parallel or processes is not None:
        p = Pool(processes=processes)

        # Submit jobs in parallel and in nested structure
        def submit_jobs(fn, cases, _l=0):
            arg, inputs = cases[0]
            for x in inputs:
                if len(cases) == 1:
                    yield p.apply_async(fn, kwds={arg: x})
                else:
                    sub_fn = partial(fn, **{arg: x})
                    yield tuple(submit_jobs(sub_fn, cases[1:], _l+1))

        # Run through nested structure retrieving results
        def get_outputs(futs, _l=0):
            for fut in progbar(futs, disable=_l >= progbars,
                               desc=fn_args[_l], **progbar_opts):
                if _l == len(fn_args) - 1:
                    y = fut.get()
                    yield y if split else (y,)
                else:
                    yield tuple(zip(*get_outputs(fut, _l+1)))

        futures = tuple(submit_jobs(fn, cases))
        outputs = tuple(zip(*get_outputs(futures)))

    else:
        def sub_case_runner(fn, cases, _l=0):
            arg, inputs = cases[0]
            for x in progbar(inputs, disable=_l >= progbars,
                             desc=arg, **progbar_opts):
                if len(cases) == 1:
                    yield fn(**{arg: x}) if split else [fn(**{arg: x})]
                else:
                    sub_fn = partial(fn, **{arg: x})
                    yield tuple(zip(*sub_case_runner(sub_fn, cases[1:], _l+1)))

        outputs = tuple(zip(*sub_case_runner(fn, cases)))

    # Make sure left progress bars are not writter over
    if progbars > 0:
        for i in range(progbars):
            print()

    return outputs if split else outputs[0]


def xr_case_runner(fn, cases, var_names, var_dims=([],),
                   var_coords={},**kwargs):
    """
    Take a function fn and analyse it over all combinations of named
    variables values, optionally showing progress and outputing to xarray.

    Parameters
    ----------
        fn: function to analyse
        cases: list of tuples of form ((variable_name, [values]), ...)
        var_dims, name of dataset's main variable, i.e. the results of fn
        **kwargs: are passed to `case_runner`

    Returns
    -------
        ds: xarray Dataset with appropirate coordinates.
    """

    # Prepare cases
    cases = tuple(cases.items() if isinstance(cases, dict) else cases)
    case_names, _ = zip(*cases)

    # Work out if multiple variables are expected as output
    if isinstance(var_names, str):
        split = False
        var_names = (var_names,)
        var_dims = (var_dims,)
    elif len(var_names) == 1:
        split = False
    else:
        split = True

    # Generate the data
    vdatas = case_runner(fn, cases, split=split, **kwargs)
    if not split:
        vdatas = (vdatas,)

    # Set dataset coordinates
    ds = xr.Dataset(coords={**dict(cases), **dict(var_coords)})

    # Set Dataset dataarrays
    for vdata, vname, vdims in zip(vdatas, var_names, cycle(var_dims)):
        ds[vname] = (tuple(case_names) + tuple(vdims), np.asarray(vdata))

    return ds


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
