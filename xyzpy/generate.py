"""
Generate datasets from function and parameter lists
"""
# TODO: find NaNs in xarray and perform cases

import functools
import itertools
import multiprocessing

import numpy as np
import xarray as xr
import tqdm


def progbar(it, nb=False, **kwargs):
    """ Turn any iterable into a progress bar, with notebook version. """
    defaults = {'ascii': True}
    # Overide defaults with custom kwargs
    settings = {**defaults, **kwargs}
    if nb:  # pragma: no cover
        return tqdm.tqdm_notebook(it, **settings)
    else:
        return tqdm.tqdm(it, **settings)


def parse_cases(cases):
    """ Turn dicts and single tuples into proper form for case runners. """
    if isinstance(cases, dict):
        cases = tuple(cases.items())
    elif isinstance(cases[0], str):
        cases = (cases,)
    else:
        cases = tuple(cases)
    return cases


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
    cases = parse_cases(cases)
    fn_args, _ = zip(*cases)

    # Prepare Function
    if constants is not None:
        fn = functools.partial(fn, **dict(constants))

    # Evaluate cases in parallel
    if parallel or (processes is not None and processes > 1):
        p = multiprocessing.Pool(processes=processes)

        # Submit jobs in parallel and in nested structure
        def submit_jobs(fn, cases, _l=0):
            arg, inputs = cases[0]
            for x in inputs:
                if len(cases) == 1:
                    yield p.apply_async(fn, kwds={arg: x})
                else:
                    sub_fn = functools.partial(fn, **{arg: x})
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

    # Evaluate cases sequentially
    else:
        def sub_case_runner(fn, cases, _l=0):
            arg, inputs = cases[0]
            for x in progbar(inputs, disable=_l >= progbars,
                             desc=arg, **progbar_opts):
                if len(cases) == 1:
                    yield fn(**{arg: x}) if split else [fn(**{arg: x})]
                else:
                    sub_fn = functools.partial(fn, **{arg: x})
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
        var_dims = itertools.cycle(var_dims)

    # Generate the data
    vdatas = case_runner(fn, cases, split=split, **kwargs)
    if not split:
        vdatas = (vdatas,)

    # Set dataset coordinates
    ds = xr.Dataset(coords={**dict(cases), **dict(var_coords)})

    # Set Dataset dataarrays
    for vdata, vname, vdims in zip(vdatas, var_names, var_dims):
        ds[vname] = (tuple(case_names) + tuple(vdims), np.asarray(vdata))

    return ds


def config_runner(fn, fn_args, configs, constants=None, split=False,
                  progbars=0, parallel=False, processes=None, progbar_opts={}):
    """
    Evaluate a function in many different configurations, optionally in
    parallel and or with live progress.
    """
    # Prepare Function
    if constants is not None:
        fn = functools.partial(fn, **dict(constants))

    # Prepate fn_args and values
    if isinstance(fn_args, str):
        fn_args = (fn_args,)
        configs = tuple((c,) for c in configs)

    # Evaluate configurations in parallel
    if parallel or (processes is not None and processes > 1):
        p = multiprocessing.Pool(processes=processes)
        fut = tuple(p.apply_async(fn, kwds=dict(zip(fn_args, c)))
                    for c in configs)
        xs = tuple(x.get() for x in progbar(fut, total=len(configs),
                                            disable=progbars<1))

    # Evaluate configutation sequentially
    else:
        xs = tuple(fn(**{arg: y for arg, y in zip(fn_args, cnfg)})
                   for cnfg in progbar(configs, total=len(configs),
                                       disable=progbars<1))

    return xs if not split else tuple(zip(*xs))
