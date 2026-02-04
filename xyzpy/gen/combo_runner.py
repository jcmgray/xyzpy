"""Functions for systematically evaluating a function over all combinations."""

import functools
import itertools
import multiprocessing
import random

import numpy as np
import xarray as xr
from joblib.externals.loky import get_reusable_executor

from ..utils import progbar
from .prepare import (
    parse_cases,
    parse_combo_results,
    parse_combos,
    parse_constants,
    parse_resources,
    parse_var_coords,
    parse_var_dims,
    parse_var_names,
)


def infer_shape(x):
    """Take a nested sequence and find its shape as if it were an array.

    Examples
    --------

        >>> x = [[10, 20, 30], [40, 50, 60]]
        >>> infer_shape(x)
        (2, 3)
    """
    shape = ()

    if isinstance(x, str):
        return shape

    try:
        shape += (len(x),)
        return shape + infer_shape(x[0])
    except TypeError:
        return shape


def nan_like_result(res):
    """Take a single result of a function evaluation and calculate the same
    sequence of scalars or arrays but filled entirely with ``nan``.

    Examples
    --------

        >>> res = (True, [[10, 20, 30], [40, 50, 60]], -42.0, 'hello')
        >>> nan_like_result(res)
        (array(nan), array([[nan, nan, nan],
                            [nan, nan, nan]]), array(nan), None)

    """
    if isinstance(res, dict):
        res = xr.Dataset(res)

    if isinstance(res, (xr.Dataset, xr.DataArray)):
        return xr.full_like(res, np.nan, dtype=float)

    if isinstance(res, (bool, str)):
        # - nan gets converted to non-null value of 'nan' (str) -> needs None
        # - by covention turn bool arrays to dtype=object -> needs None
        return None

    try:
        return tuple(np.broadcast_to(np.nan, infer_shape(x)) for x in res)
    except TypeError:
        return np.nan


def _submit(executor, fn, *args, **kwds):
    """Default method for submitting to a executor.

    Parameters
    ----------
    executor : pool-executor like
        A ``multiprocessing.pool`` or an pool executor with API matching either
        ``concurrent.futures``, or an ``ipyparallel`` view.
    fn : callable
        The function to submit.
    args :
        Supplied to ``fn``.
    kwds :
        Supplied to ``fn``.

    Returns
    --------
    future
    """
    if isinstance(executor, multiprocessing.pool.Pool):
        return executor.apply_async(fn, args, kwds)

    elif hasattr(executor, "submit"):
        # concurrent.futures like API
        return executor.submit(fn, *args, **kwds)

    elif hasattr(executor, "apply_async"):
        # ipyparallel like API
        return executor.apply_async(fn, *args, **kwds)

    else:
        raise TypeError(
            "The executor supplied, {}, does not have a ``submit``"
            " or ``apply_async`` method.".format(executor)
        )


def _get_result(future):
    # don't using try-except here, because futures might raise themselves
    if hasattr(future, "result"):
        return future.result()
    if hasattr(future, "get"):
        return future.get()
    raise TypeError("Future does not have a `result` or `get` method.")


def _run_linear_executor(
    executor,
    fn,
    settings,
    verbosity=1,
):
    with progbar(total=len(settings), disable=verbosity <= 0) as pbar:
        if verbosity >= 2:
            pbar.set_description("Submitting to executor...")
        futures = [_submit(executor, fn, **kws) for kws in settings]
        results_linear = []
        for kws, future in zip(settings, futures):
            if verbosity >= 2:
                pbar.set_description(str(kws))
            results_linear.append(_get_result(future))
            pbar.update()
        return results_linear


def _run_linear_sequential(fn, settings, verbosity=1):
    results_linear = []
    with progbar(total=len(settings), disable=verbosity <= 0) as pbar:
        for kws in settings:
            if verbosity >= 2:
                pbar.set_description(str(kws))
            results_linear.append(fn(**kws))
            pbar.update()
        return results_linear


def _unflatten(store, all_combo_values, all_nan=None):
    # non-recursive nested accumulation of results into tuple array
    while all_combo_values:
        # pop out the last arg
        *all_combo_values, last = all_combo_values
        for p in itertools.product(*all_combo_values):
            # for each remaining combination, reduce last arg into tuple
            store[p] = tuple(store.pop(p + (v,), all_nan) for v in last)
    return store.pop(())


def combo_runner_core(
    fn,
    combos,
    constants,
    cases=None,
    split=False,
    flat=False,
    shuffle=False,
    parallel=False,
    num_workers=None,
    executor=None,
    verbosity=1,
    info=None,
):
    if combos:
        combo_args, combo_values = zip(*combos)
    else:
        combo_args, combo_values = (), ()

    if cases is not None:
        cases = tuple(cases)
        case_args = tuple(cases[0].keys())
        case_values = tuple(tuple(c[a] for a in case_args) for c in cases)
        case_coords = {arg: set() for arg in case_args}
    else:
        # single empty case and everything is in the combos
        cases = ()
        case_args = ()
        case_values = ((),)
        case_coords = {}

    if not set(case_args).isdisjoint(combo_args):
        raise ValueError(
            f"Variables can't appear in both ``cases`` and ``combos``, "
            f"currently found combo variables {combo_args} and case variables"
            f"{case_args}."
        )

    # order arguments will be iterated over
    fn_args = case_args + combo_args
    # key location for each case to map into array
    locs = []
    # the actual list of all kwargs supplied to each fn call
    settings = []

    for case_params in case_values:
        # keep track of every case value we see to form union later
        for arg, v in zip(case_args, case_params):
            case_coords[arg].add(v)

        for combo_params in itertools.product(*combo_values):
            loc = case_params + combo_params
            kws = dict(zip(fn_args, loc))
            kws.update(constants)
            locs.append(loc)
            settings.append(kws)

    if shuffle:
        random.seed(int(shuffle))
        enum_settings = list(enumerate(settings))
        random.shuffle(enum_settings)
        enum, settings = zip(*enum_settings)

    run_linear_opts = {"fn": fn, "settings": settings, "verbosity": verbosity}

    if executor == "ray":
        from .ray_executor import RayExecutor

        executor = RayExecutor(num_cpus=num_workers)

    if executor is not None:
        # custom pool supplied
        results_linear = _run_linear_executor(executor, **run_linear_opts)
    elif parallel or num_workers:
        # else for parallel, by default use a process pool-exceutor

        if (
            # bools are ints, so check for that first since True != 1 here
            (not isinstance(parallel, bool))
            and isinstance(parallel, int)
            and (num_workers is None)
        ):
            # assume parallel is the number of workers
            num_workers = parallel

        executor = get_reusable_executor(num_workers)
        results_linear = _run_linear_executor(executor, **run_linear_opts)
    else:
        results_linear = _run_linear_sequential(**run_linear_opts)

    if shuffle:
        enum_results = sorted(zip(enum, results_linear), key=lambda x: x[0])
        _, results_linear = zip(*enum_results)

    # try and put the union of case coordinates into a reasonable order
    for arg in case_args:
        try:
            case_coords[arg] = sorted(case_coords[arg])
        except TypeError:  # unsortable
            case_coords[arg] = list(case_coords[arg])

    # find the equivalent combos as if all coordinates had been run
    combos_cases = tuple(case_coords.values())
    all_combo_values = combos_cases + combo_values

    def process_results(r):
        if flat:
            # just return the list of results
            return tuple(r)

        results_mapped = dict(zip(locs, r))

        if not cases:
            # we ran all combinations -> no missing data
            return _unflatten(results_mapped, combo_values)

        # unpack dict into nested tuple, ready for numpy
        all_nan = nan_like_result(r[0])
        results = _unflatten(results_mapped, all_combo_values, all_nan)

        return results

    if info is not None:
        # optionally return some extra labelling information
        if flat:
            info["settings"] = settings
        else:
            info["fn_args"] = fn_args
            info["all_combo_values"] = all_combo_values

    if split:
        # put each output variable into a seperate results at the top level
        return tuple(process_results(r) for r in zip(*results_linear))
    else:
        return process_results(results_linear)


def combo_runner(
    fn,
    combos=None,
    *,
    cases=None,
    constants=None,
    split=False,
    flat=False,
    shuffle=False,
    parallel=False,
    executor=None,
    num_workers=None,
    verbosity=1,
):
    """Take a function ``fn`` and compute it over all combinations of named
    variables values, optionally showing progress and in parallel.

    Parameters
    ----------
    fn : callable
        Function to analyse.
    combos : dict_like[str, iterable]
        All combinations of each argument to values mapping will be computed.
        Each argument range thus gets a dimension in the output array(s).
    cases  : sequence of mappings, optional
        Optional list of specific configurations. If both ``combos`` and
        ``cases`` are given, then the function is computed for all
        sub-combinations in ``combos`` for each case in ``cases``, arguments
        can thus only appear in one or the other. Note that missing
        combinations of arguments will be represented by ``nan`` if creating a
        nested array.
    constants : dict, optional
        Constant function arguments. Unlike ``combos`` and ``cases``, these
        won't produce dimensions in the output result when ``flat=False``.
    split : bool, optional
        Whether to split (unzip) the outputs of ``fn`` into multiple output
        arrays or not.
    flat : bool, optional
        Whether to return a flat list of results or to return a nested
        tuple suitable to be supplied to ``numpy.array``.
    shuffle : bool or int, optional
        If given, compute the results in a random order (using ``random.seed``
        and ``random.shuffle``), which can be helpful for distributing
        resources when not all cases are computationally equal.
    parallel : bool, optional
        Process combos in parallel, default number of workers picked.
    executor : executor-like pool, optional
        Submit all combos to this pool executor. Must have ``submit`` or
        ``apply_async`` methods and API matching either ``concurrent.futures``
        or an ``ipyparallel`` view. Pools from ``multiprocessing.pool`` are
        also  supported.
    num_workers : int, optional
        Explicitly choose how many workers to use, None for automatic.
    verbosity : {0, 1, 2}, optional
        How much information to display:

            - 0: nothing,
            - 1: just progress,
            - 2: all information.

    Returns
    -------
    data : nested tuple
        Nested tuple containing all combinations of running ``fn`` if
        ``flat == False`` else a flat list of results.

    Examples
    --------

        >>> def fn(a, b, c, d):
        ...     return str(a) + str(b) + str(c) + str(d)


    Run all possible combos::

        >>> xyz.combo_runner(
        ...     fn,
        ...     combos={
        ...         'a': [1, 2],
        ...         'b': [3, 4],
        ...         'c': [5, 6],
        ...         'd': [7, 8],
        ...     },
        ... )
        100%|##########| 16/16 [00:00<00:00, 84733.41it/s]

        (((('1357', '1358'), ('1367', '1368')),
          (('1457', '1458'), ('1467', '1468'))),
         ((('2357', '2358'), ('2367', '2368')),
          (('2457', '2458'), ('2467', '2468'))))

    Run only a selection of cases::

        >>> xyz.combo_runner(
        ...     fn,
        ...     cases=[
        ...         {'a': 1, 'b': 3, 'c': 5, 'd': 7},
        ...         {'a': 2, 'b': 4, 'c': 6, 'd': 8},
        ...     ],
        ... )
        100%|##########| 2/2 [00:00<00:00, 31418.01it/s]
        (((('1357', nan), (nan, nan)),
          ((nan, nan), (nan, nan))),
         (((nan, nan), (nan, nan)),
          ((nan, nan), (nan, '2468'))))

    Run only certain cases of some args, but all combinations of others::

        >>> xyz.combo_runner(
        ...     fn,
        ...     cases=[
        ...         {'a': 1, 'b': 3},
        ...         {'a': 2, 'b': 4},
        ...     ],
        ...     combos={
        ...         'c': [3, 4],
        ...         'd': [4, 5],
        ...     },
        ... )
        100%|##########| 8/8 [00:00<00:00, 92691.80it/s]
        (((('1334', '1335'), ('1344', '1345')),
          ((nan, nan), (nan, nan))),
         (((nan, nan), (nan, nan)),
          (('2434', '2435'), ('2444', '2445'))))

    """
    # Prepare combos
    cases = parse_cases(cases)
    combos = parse_combos(combos)
    constants = parse_constants(constants)

    # Submit to core combo runner
    return combo_runner_core(
        fn=fn,
        combos=combos,
        cases=cases,
        constants=constants,
        split=split,
        flat=flat,
        shuffle=shuffle,
        parallel=parallel,
        executor=executor,
        num_workers=num_workers,
        verbosity=verbosity,
    )


def multi_concat(results, dims):
    """Concatenate a nested list of xarray objects along several dimensions."""
    if len(dims) == 1:
        return xr.concat(
            [
                # if a dict, convert to dataset
                xr.Dataset(obj)
                if isinstance(obj, dict)
                # else assume it's a dataset or datarray
                else obj
                for obj in results
            ],
            dim=dims[0],
        )
    else:
        return xr.concat(
            [multi_concat(sub_results, dims[1:]) for sub_results in results],
            dim=dims[0],
        )


def get_ndim_first(x, ndim):
    """Return the first element from the ndim-nested list x."""
    return x if ndim == 0 else get_ndim_first(x[0], ndim - 1)


def results_to_ds(
    results,
    combos,
    var_names,
    var_dims,
    var_coords,
    constants=None,
    attrs=None,
):
    """Convert the output of combo_runner into a :class:`xarray.Dataset`."""
    fn_args = tuple(x for x, _ in combos)
    results = parse_combo_results(results, var_names)

    if len(results) != len(var_names):
        raise ValueError(
            f"Wrong number of results ({len(results)}) for "
            f"{len(var_names)} ``var_names``: {var_names}."
        )

    # Check if the results are an array of xarray objects
    xobj_results = isinstance(
        get_ndim_first(results, len(fn_args) + 1),
        (dict, xr.Dataset, xr.DataArray),
    )

    if xobj_results:
        # concat them all together, no var_names needed
        ds = multi_concat(results[0], fn_args)
        # Set dataset coordinates
        for fn_arg, vals in combos:
            ds[fn_arg] = vals
    else:
        # create a new dataset using the given arrays and var_names
        ds = xr.Dataset(
            coords={**dict(combos), **dict(var_coords)},
            data_vars={
                name: (fn_args + var_dims[name], np.asarray(data))
                for data, name in zip(results, var_names)
            },
        )

    if attrs:
        ds.attrs = attrs

    # Add constants to attrs, but filter out those which should be coords
    if constants:
        for k, v in constants.items():
            if callable(v):
                # should't store functions as attrs or coords
                continue

            if k in ds.dims:
                ds.coords[k] = v
            else:
                try:
                    ds.attrs[k] = v
                except Exception as e:
                    import warnings

                    warnings.warn(
                        f"Failed to add constant {k}={v} to dataset attrs: {e}"
                    )
    return ds


def results_to_df(
    results_linear,
    settings,
    attrs,
    resources,
    var_names,
):
    """Convert the output of combo_runner into a :class:`pandas.DataFrame`."""
    import pandas as pd

    # construct as list of single dict entries
    data = []
    for row, result in zip(settings, results_linear):
        # don't record resources
        for k in resources:
            row.pop(k, None)

        # add in the attrs, note this isn't quite equivalent to dataset case,
        # as we add the attributes for every entry -> limitation of dataframe
        if attrs:
            row.update(attrs)

        # add in the output variables
        try:
            row.update(dict(zip(var_names, result)))
        except TypeError:
            row.update(dict(zip(var_names, [result])))

        data.append(row)

    # convert to dataframe
    return pd.DataFrame(data)


def combo_runner_to_ds(
    fn,
    combos,
    var_names,
    *,
    var_dims=None,
    var_coords=None,
    cases=None,
    constants=None,
    resources=None,
    attrs=None,
    shuffle=False,
    parse=True,
    to_df=False,
    parallel=False,
    num_workers=None,
    executor=None,
    verbosity=1,
):
    """Evaluate a function over all cases and combinations and output to a
    :class:`xarray.Dataset`.

    Parameters
    ----------
    fn : callable
        Function to evaluate.
    combos : dict_like[str, iterable]
        Mapping of each individual function argument to sequence of values.
    var_names : str, sequence of strings, or None
        Variable name(s) of the output(s) of `fn`, set to None if
        fn outputs data already labelled in a Dataset or DataArray.
    var_dims : sequence of either strings or string sequences, optional
        'Internal' names of dimensions for each variable, the values for
        each dimension should be contained as a mapping in either
        `var_coords` (not needed by `fn`) or `constants` (needed by `fn`).
    var_coords : mapping, optional
        Mapping of extra coords the output variables may depend on.
    cases : sequence of dicts, optional
        Individual cases to run for some or all function arguments.
    constants : mapping, optional
        Arguments to `fn` which are not iterated over, these will be
        recorded either as attributes or coordinates if they are named
        in `var_dims`.
    resources : mapping, optional
        Like `constants` but they will not be recorded.
    attrs : mapping, optional
        Any extra attributes to store.
    parallel : bool, optional
        Process combos in parallel, default number of workers picked.
    executor : executor-like pool, optional
        Submit all combos to this pool executor. Must have ``submit`` or
        ``apply_async`` methods and API matching either ``concurrent.futures``
        or an ``ipyparallel`` view. Pools from ``multiprocessing.pool`` are
        also  supported.
    num_workers : int, optional
        Explicitly choose how many workers to use, None for automatic.
    verbosity : {0, 1, 2}, optional
        How much information to display:

        - 0: nothing,
        - 1: just progress,
        - 2: all information.

    Returns
    -------
    ds : xarray.Dataset or pandas.DataFrame
        Multidimensional labelled dataset contatining all the results if
        ``to_df=False`` (the default), else a pandas dataframe with results
        as labelled rows.
    """
    if to_df:
        if var_names is None:
            raise ValueError("Can't coerce dataset output into dataframe.")
        if var_dims is not None and any(var_dims.values()):
            raise ValueError("Dataframes don't support internal dimensions.")
        if var_coords:
            raise ValueError("Dataframes don't support internal dimensions.")

    if parse:
        combos = parse_combos(combos)
        cases = parse_cases(cases)
        constants = parse_constants(constants)
        resources = parse_resources(resources)
        var_names = parse_var_names(var_names)
        var_dims = parse_var_dims(var_dims, var_names=var_names)
        var_coords = parse_var_coords(var_coords)

    if cases or to_df:
        info = {}
    else:
        info = None

    # Generate data for all combos
    results = combo_runner_core(
        fn=fn,
        combos=combos,
        cases=cases,
        constants={**resources, **constants},
        parallel=parallel,
        num_workers=num_workers,
        executor=executor,
        verbosity=verbosity,
        info=info,
        split=(not to_df) and (len(var_names) > 1),
        flat=to_df,
        shuffle=shuffle,
    )

    if to_df:
        # convert flat tuple of results to dataframe
        return results_to_df(
            results,
            settings=info["settings"],
            attrs=attrs,
            resources=resources,
            var_names=var_names,
        )

    if cases:
        # if we have cases, then need to find the effective full combos
        # -> results contains nan placeholders for non-run cases
        combos = tuple(zip(info["fn_args"], info["all_combo_values"]))

    # convert to dataset
    ds = results_to_ds(
        results,
        combos,
        var_names=var_names,
        var_dims=var_dims,
        var_coords=var_coords,
        constants=constants,
        attrs=attrs,
    )

    return ds


combo_runner_to_df = functools.partial(combo_runner_to_ds, to_df=True)
