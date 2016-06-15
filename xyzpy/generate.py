"""
Generate datasets from function and parameter lists
"""

# TODO: find NaNs in xarray and perform cases
# TODO: rename --> combo_runner, case_runner

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


def parse_combos(combos):
    """ Turn dicts and single tuples into proper form for combo runners. """
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
        p = multiprocessing.Pool(processes=processes)

        # Submit jobs in parallel and in nested structure
        def submit_jobs(fn, combos, _l=0):
            arg, inputs = combos[0]
            for x in inputs:
                if len(combos) == 1:
                    yield p.apply_async(fn, kwds={arg: x})
                else:
                    sub_fn = functools.partial(fn, **{arg: x})
                    yield tuple(submit_jobs(sub_fn, combos[1:], _l+1))

        # Run through nested structure retrieving results
        def get_outputs(futs, _l=0):
            for fut in progbar(futs, disable=_l >= progbars,
                               desc=fn_args[_l], **progbar_opts):
                if _l == len(fn_args) - 1:
                    y = fut.get()
                    yield y if split else (y,)
                else:
                    yield tuple(zip(*get_outputs(fut, _l+1)))

        futures = tuple(submit_jobs(fn, combos))
        outputs = tuple(zip(*get_outputs(futures)))

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

        outputs = tuple(zip(*get_ouputs_seq(fn, combos)))

    # Make sure left progress bars are not writter over
    if progbars > 0:
        for i in range(progbars):
            print()

    return outputs if split else outputs[0]


def combos_to_ds():
    pass


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
        p = multiprocessing.Pool(processes=processes)
        fut = tuple(p.apply_async(fn, kwds=dict(zip(fn_args, c)))
                    for c in cases)
        xs = tuple(x.get() for x in progbar(fut, total=len(cases),
                                            disable=progbars < 1))

    # Evaluate configutation sequentially
    else:
        xs = tuple(fn(**{arg: y for arg, y in zip(fn_args, cnfg)})
                   for cnfg in progbar(cases, total=len(cases),
                                       disable=progbars < 1))

    return xs if not split else tuple(zip(*xs))


def combo_runner_to_ds(fn, combos, var_names, var_dims=([],), var_coords={},
                       **kwargs):
    """
    Take a function fn and analyse it over all combinations of named
    variables values, optionally showing progress and outputing to xarray.

    Parameters
    ----------
        fn: function to analyse
        combos: list of tuples of form ((variable_name, [values]), ...) with
            which to generate all function evaluation configurations.
        var_names: name(s) of output variables of `fn`
        var_dims: the list of named output coordinates for each output
            variable, i.e. coordinates not generated by the combo_runner
        var_coords: dict of values for those coordinates if custom ones are
            desired.
        **kwargs: are passed to `combo_runner`

    Returns
    -------
        ds: xarray Dataset with appropirate coordinates.

    Examples
    --------
    In [1]: from xyzpy import *

    In [2]: def foo(a, b, c):
       ...:     return a + b + c, (a + b + c) % 2 == 0
       ...:
       ...:

    In [3]: var_names = ['total', 'is_even']

    In [4]: combos = [('a', [1, 2, 3]),
                     ('b', [10, 20, 30]),
                     ('c', [100, 200, 300])]

    In [5]: combo_runner_to_ds(foo, combos, var_names)
    Out[5]:
    <xarray.Dataset>
    Dimensions:  (a: 3, b: 3, c: 3)
    Coordinates:
      * b        (b) int64 10 20 30
      * a        (a) int64 1 2 3
      * c        (c) int64 100 200 300
    Data variables:
        total    (a, b, c) int64 111 211 311 121 221 321 131 231 331 112 ...
        is_even  (a, b, c) bool False False False False False False False ...
    """

    # TODO: split out data preperation

    # Prepare combos
    combos = tuple(combos.items() if isinstance(combos, dict) else combos)
    fn_args, _ = zip(*combos)

    # Work out if multiple variables are expected as output
    if isinstance(var_names, str):
        split = False
        var_names = (var_names,)
        if var_dims != ([],):
            var_dims = (var_dims,)
    elif len(var_names) == 1:
        split = False
    else:
        split = True
    var_dims = itertools.cycle(var_dims)

    # Generate the data
    vdatas = combo_runner(fn, combos, split=split, **kwargs)
    if not split:
        vdatas = (vdatas,)

    # Set dataset coordinates
    ds = xr.Dataset(coords={**dict(combos), **dict(var_coords)})

    # Set Dataset dataarrays
    for vdata, vname, vdims in zip(vdatas, var_names, var_dims):
        ds[vname] = (tuple(fn_args) + tuple(vdims), np.asarray(vdata))

    return ds


def cases_to_ds(results, configs, case_names, var_names,
                var_dims=([],), var_coords={}):

    # Prepare case_names/configs var_names/results
    if isinstance(case_names, str):
        case_names = (case_names,)
        configs = tuple((c,) for c in configs)
    if isinstance(var_names, str):
        var_names = (var_names,)
        results = tuple((r,) for r in results)
        if var_dims != ([],):
            var_dims = (var_dims,)
    var_dims = itertools.cycle(var_dims)

    # Find required coordinates from configs
    c_coords = {arg: sorted([*{*arg_vals}])
                for arg, arg_vals in zip(case_names, zip(*configs))}

    # Add dimensions from cases and var_coords
    ds = xr.Dataset(coords={**c_coords, **dict(var_coords)})
    shape_c_coords = [ds[dim].size for dim in case_names]

    # go through var_names, adding np.nan in correct shape
    for vname, dims in zip(var_names, var_dims):
        vdims = tuple(case_names) + tuple(dims)
        shape = shape_c_coords + [ds[d].size for d in dims]
        ds[vname] = (vdims, np.tile(np.nan, shape))

    #  go through configs, overwriting nan with results
    for res, cfg in zip(results, configs):
        for vname, x in zip(var_names, res):
            ds[vname].loc[dict(zip(case_names, cfg))] = x

    return ds


def case_runner_to_ds():
    pass
