"""
Generate datasets from function and parameter lists
"""

# TODO: add to existing dataset
# TODO: find NaNs in xarray and perform cases

import functools
import itertools
import multiprocessing

import numpy as np
import xarray as xr
import tqdm


def progbar(it=None, nb=False, **kwargs):
    """
    Turn any iterable into a progress bar, with notebook version.
    """
    defaults = {'ascii': True, 'smoothing': 0}
    # Overide defaults with custom kwargs
    settings = {**defaults, **kwargs}
    if nb:  # pragma: no cover
        return tqdm.tqdm_notebook(it, **settings)
    else:
        return tqdm.tqdm(it, **settings)


# --------------------------------------------------------------------------- #
# COMBO_RUNNER functions: for evaluating all possible combinations of args    #
# --------------------------------------------------------------------------- #

def parse_combos(combos):
    """
    Turn dicts and single tuples into proper form for combo runners.
    """
    if isinstance(combos, dict):
        combos = tuple(combos.items())
    elif isinstance(combos[0], str):
        combos = (combos,)
    else:
        combos = tuple(combos)
    return combos


def combo_runner(fn, combos, constants=None, split=False, progbars=0,
                 parallel=False, processes=None, progbar_opts={}):
    """
    Take a function fn and analyse it over all combinations of named
    variables' values, optionally showing progress and in parallel.

    Parameters
    ----------
        fn: function to analyse
        combos: list of tuples/dict of form ((variable_name, [values]), ...),
            all combinations of each argument will be calculated. Each
            argument range thus gets a dimension in the output array(s).
        constants: list of tuples/dict of *constant* fn argument mappings.
        split: whether to split into multiple output arrays or not.
        progbars: how many levels of nested progress bars to show.
        parallel: process combos in parallel.
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

    In [4]: combos = (('a', [100, 400]), ('b', [20, 50]), ('c', [3, 6]))

    In [5]: combo_runner(foo, combos, progbars=3)
    a: 100%|##########| 2/2 [00:08<00:00,  4.02s/it]
    b: 100%|##########| 2/2 [00:04<00:00,  2.01s/it]
    c: 100%|##########| 2/2 [00:02<00:00,  1.00s/it]

    Out[5]:
    ((((123, False), (126, True)), ((153, False), (156, True))),
     (((423, False), (426, True)), ((453, False), (456, True))))

    In [6]: combo_runner(foo, combos, progbars=3, parallel=True)  # Faster!
    a: 100%|##########| 2/2 [00:01<00:00,  1.96it/s]
    b: 100%|##########| 2/2 [00:00<00:00, 327.58it/s]
    c: 100%|##########| 2/2 [00:00<00:00, 17512.75it/s]

    Out[6]:
    ((((123, False), (126, True)), ((153, False), (156, True))),
     (((423, False), (426, True)), ((453, False), (456, True))))


    In [7]: x, y = combo_runner(foo, combos, split=True)

    In [8]: x
    Out[8]: (((123, 126), (153, 156)), ((423, 426), (453, 456)))

    In [9]: y
    Out[9]: (((False, True), (False, True)), ((False, True), (False, True)))

    Notes
    -----
        1. The parallel evaluation of combos relies on the `multiprocessing`
            module, this places some restrictions of what functions can be
            used (needs to be picklable/importable).
    """

    # Prepare combos
    combos = parse_combos(combos)
    fn_args, _ = zip(*combos)

    # Prepare Function
    if constants is not None:
        fn = functools.partial(fn, **dict(constants))

    # Evaluate combos in parallel
    if parallel or (processes is not None and processes > 1):
        mp = multiprocessing.Pool(processes=processes)

        # Submit jobs in parallel and in nested structure
        def submit_jobs(fn, combos, _l=0):
            arg, inputs = combos[0]
            for x in inputs:
                if len(combos) == 1:
                    yield mp.apply_async(fn, kwds={arg: x})
                else:
                    sub_fn = functools.partial(fn, **{arg: x})
                    yield tuple(submit_jobs(sub_fn, combos[1:], _l+1))

        # Run through nested structure retrieving results
        def get_results(futs, _l=0):
            for fut in progbar(futs, disable=_l >= progbars,
                               desc=fn_args[_l], **progbar_opts):
                if _l == len(fn_args) - 1:
                    y = fut.get()
                    yield y if split else (y,)
                else:
                    yield tuple(zip(*get_results(fut, _l+1)))

        futures = tuple(submit_jobs(fn, combos))
        results = tuple(zip(*get_results(futures)))

    # Evaluate combos sequentially
    else:
        def get_ouputs_seq(fn, combos, _l=0):
            arg, inputs = combos[0]
            for x in progbar(inputs, disable=_l >= progbars,
                             desc=arg, **progbar_opts):
                if len(combos) == 1:
                    yield fn(**{arg: x}) if split else [fn(**{arg: x})]
                else:
                    sub_fn = functools.partial(fn, **{arg: x})
                    yield tuple(zip(*get_ouputs_seq(sub_fn, combos[1:], _l+1)))

        results = tuple(zip(*get_ouputs_seq(fn, combos)))

    # Make sure left progress bars are not writter over
    if progbars > 0:
        for i in range(progbars):
            print()

    return results if split else results[0]


def combos_to_ds(results, combos, var_names, var_dims=None, var_coords={}):
    """
    Convert the output of combo_runner into a `xarray.Dataset`

    Parameters
    ----------
        results: array(s) of dimension `len(combos)`
        combos: list of tuples of form ((variable_name, [values]), ...) with
            which `results` was generated.
        var_names: name(s) of output variables for a single result
        var_dims: the list of named coordinates for each single result
            variable, i.e. coordinates not generated by the combo_runner
        var_coords: dict of values for those coordinates if custom ones are
            desired.

    Returns
    -------
        ds: xarray Dataset with appropriate coordinates.
    """
    combos = parse_combos(combos)
    fn_args, _ = zip(*combos)

    # Work out if multiple variables are expected as output
    if isinstance(var_names, str):
        var_names = (var_names,)
        results = (results,)
        if var_dims is not None:
            var_dims = (var_dims,)
    elif len(var_names) == 1:
        results = (results,)

    # Allow single given dimensions to represent all result variables
    var_dims = (itertools.cycle(var_dims) if var_dims is not None else
                itertools.repeat(tuple()))

    # Set dataset coordinates
    ds = xr.Dataset(coords={**dict(combos), **dict(var_coords)})

    # Set Dataset dataarrays
    for vdata, vname, vdims in zip(results, var_names, var_dims):
        ds[vname] = (tuple(fn_args) + tuple(vdims), np.asarray(vdata))

    return ds


def combo_runner_to_ds(fn, combos, var_names, var_dims=None, var_coords={},
                       **combo_runner_settings):

    # Set split based on output var_names
    split = (False if isinstance(var_names, str) else
             False if len(var_names) == 1 else
             True)

    # Generate data for all combos
    results = combo_runner(fn, combos, split=split, **combo_runner_settings)

    # Convert to dataset
    ds = combos_to_ds(results=results, combos=combos, var_names=var_names,
                      var_dims=var_dims, var_coords=var_coords)

    return ds


# --------------------------------------------------------------------------- #
# CASE_RUNNER functions: for evaluating specific configurations of args       #
# --------------------------------------------------------------------------- #

def case_runner(fn, fn_args, cases, constants=None, split=False,
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
        cases = tuple((c,) for c in cases)

    # Evaluate configurations in parallel
    if parallel or (processes is not None and processes > 1):
        mp = multiprocessing.Pool(processes=processes)
        fut = tuple(mp.apply_async(fn, kwds=dict(zip(fn_args, c)))
                    for c in cases)
        results = tuple(x.get() for x in progbar(fut, total=len(cases),
                                                 disable=progbars < 1))

    # Evaluate configurations sequentially
    else:
        results = tuple(fn(**{arg: y for arg, y in zip(fn_args, cnfg)})
                        for cnfg in progbar(cases, total=len(cases),
                                            disable=progbars < 1))

    return results if not split else tuple(zip(*results))


def minimal_covering_coords(cases):
    """
    Take a list of cases and find the minimal covering set of coordinates
    with which to index all cases. Sort the coords if possible.
    """
    for x in zip(*cases):
        try:
            yield sorted(list(set(x)))
        except TypeError:  # unsortable
            yield list(set(x))


def all_missing_ds(coords, var_names, var_dims, var_types):
    """
    Make a dataset whose data is all missing.
    """
    # Blank dataset with appropirate coordinates
    ds = xr.Dataset(coords=coords)

    # go through var_names, adding np.nan in correct shape and type
    for vname, dims, vtype in zip(var_names, var_dims, var_types):
        shape = tuple(ds[d].size for d in dims)
        if vtype == int or vtype == float:
            # Warn about casting?
            nodata = np.tile(np.nan, shape)
        elif vtype == complex:
            nodata = np.tile(np.nan + np.nan*1.0j, shape)
        else:
            nodata = np.tile(np.nan, shape).astype(object)
        ds[vname] = (dims, nodata)

    return ds


def cases_to_ds(results, fn_args, cases, var_names, var_dims=None,
                var_coords={}, add_to_ds=None):
    """
    Take a list of results and configurations that generate them and turn it
    into a `xarray.Dataset`.

    Parameters
    ----------
        results: list(s) of results of len(cases), e.g. generated by
            `case_runner`.
        fn_args: arguments used in function that generated the results
        cases: list of configurations used to generate results
        var_names: name(s) of output variables for a single result
        var_dims: the list of named coordinates for each single result
            variable, i.e. coordinates not generated by the combo_runner
        var_coords: dict of values for those coordinates if custom ones are
            desired.

    Returns
    -------
        ds: Dataset holding all results, with coordinates described by cases

    Notes
    -----
        1. Many data types have to be converted to object in order for the
            missing values to be represented by NaNs.
    """

    # Prepare fn_args/cases var_names/results
    if isinstance(fn_args, str):
        fn_args = (fn_args,)
        cases = tuple((c,) for c in cases)

    # Prepare var_names/dims/results
    if isinstance(var_names, str):
        var_names = (var_names,)
        results = tuple((r,) for r in results)
        if var_dims is not None:
            var_dims = (var_dims,)

    if add_to_ds is None:
        # Allow single given dimensions to represent all result variables
        var_dims = (itertools.cycle(var_dims) if var_dims is not None else
                    itertools.repeat(tuple()))

        # Find minimal covering set of coordinates for fn_args
        case_coords = dict(zip(fn_args, minimal_covering_coords(cases)))

        # Create new, 'all missing' dataset if required
        ds = all_missing_ds(coords={**case_coords, **dict(var_coords)},
                            var_names=var_names,
                            var_dims=(tuple(fn_args) + tuple(next(var_dims))
                                      for i in range(len(var_names))),
                            var_types=(np.asarray(x).dtype
                                       for x in results[0]))
    else:
        ds = add_to_ds

    #  go through cases, overwriting nan with results
    for res, cfg in zip(results, cases):
        for vname, x in zip(var_names, res):
            ds[vname].loc[dict(zip(fn_args, cfg))] = np.asarray(x)

    return ds


def case_runner_to_ds(fn, fn_args, cases, var_names, var_dims=None,
                      var_coords={}, add_to_ds=None, **case_runner_settings):
    """
    Combination of `case_runner` and `cases_to_ds`. Takes a function and list
    of argument configurations and produces a `xarray.Dataset`.

    Parameters
    ----------
        fn: function to evaluate
        fn_args: names of function args
        cases: list of function arg configurations
        var_names: list of names of single fn output
        var_dims: list of list of extra dims for each fn output
        var_coords: dictionary describing custom values of var_dims
        case_runner_settings: dict to supply to `case_runner`

    Returns
    -------
        ds: dataset with minimal covering coordinates and all cases evaluted
    """
    # Generate results
    results = case_runner(fn, fn_args, cases, **case_runner_settings)

    # Convert to xarray.Dataset
    ds = cases_to_ds(results, fn_args, cases, var_names=var_names,
                     var_dims=var_dims, var_coords=var_coords,
                     add_to_ds=add_to_ds)

    return ds
