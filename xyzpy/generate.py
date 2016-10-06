"""Generate datasets from function and parameter lists
"""

# TODO: 'resources', big arguments, not to store anywhere. ------------------ #
# TODO: allow combo_runner_to_ds to use output vars as coords --------------- #
# TODO: automatically count the number of progbars (i.e. no. of len > 1 coos) #
# TODO: combos add to existing dataset. ------------------------------------- #
# TODO: save to ds every case. For case_runner only? ------------------------ #
# TODO: function for printing ranges of runs done. -------------------------- #
# TODO: logging ------------------------------------------------------------- #
# TODO: catch attribute error from multiprocessing unimportable functions --- #
# TODO: proper docs --------------------------------------------------------- #
# TODO: pause / finish early interactive commands. -------------------------- #

import functools
import itertools
import numpy as np
import xarray as xr
import tqdm

from .parallel_work import Pool


def progbar(it=None, nb=False, **tqdm_settings):
    """Turn any iterable into a progress bar, with notebook option.
    """
    defaults = {'ascii': True, 'smoothing': 0}
    # Overide defaults with custom tqdm_settings
    settings = {**defaults, **tqdm_settings}
    if nb:  # pragma: no cover
        return tqdm.tqdm_notebook(it, **settings)
    return tqdm.tqdm(it, **settings)


# --------------------------------------------------------------------------- #
# COMBO_RUNNER functions: for evaluating all possible combinations of args    #
# --------------------------------------------------------------------------- #

def _parse_combos(combos):
    """ Turn dicts and single tuples into proper form for combo runners. """
    if isinstance(combos, dict):
        return tuple(combos.items())
    elif isinstance(combos[0], str):
        return (combos,)
    return tuple(combos)


def combo_runner(fn, combos,
                 constants=None,
                 resources=None,
                 split=False,
                 progbars=0,
                 parallel=False,
                 processes=None,
                 parallel_backend='MP',
                 progbar_opts=None):
    """Take a function fn and analyse it over all combinations of named
    variables' values, optionally showing progress and in parallel.

    Parameters
    ----------
        fn: callable
            function to analyse
        combos: mapping of individual fn arguments to iterable of values
            All combinations of each argument will be calculated. Each
            argument range thus gets a dimension in the output array(s).
        constants:
            list of tuples/dict of *constant* fn argument mappings.
        split:
            whether to split into multiple output arrays or not.
        progbars:
            how many levels of nested progress bars to show.
        parallel: bool
            process combos in parallel, default number of workers picked.
        processes: int
            Explicitly choose how many processes to use, None for automatic.
        progbar_opts: dict
            Options for the progress bar.

    Returns
    -------
        data:
            list of result arrays, each with all param combinations in nested
            tuples.
    """
    # Prepare combos
    combos = _parse_combos(combos)
    fn_args, _ = zip(*combos)

    constants = dict() if constants is None else dict(constants)
    resources = dict() if resources is None else dict(resources)

    # Prepare Function
    if len(constants) + len(resources) > 0:
        fn = functools.partial(fn, **constants, **resources)

    if progbar_opts is None:
        progbar_opts = dict()

    # Evaluate combos in parallel
    if parallel or (processes is not None and processes > 1):
        with Pool(n=processes, backend=parallel_backend) as pool:

            # Submit jobs in parallel and in nested structure
            def submit_jobs(fn, combos):
                arg, inputs = combos[0]
                for x in inputs:
                    if len(combos) == 1:
                        yield pool.submit(fn, **{arg: x})
                    else:
                        sub_fn = functools.partial(fn, **{arg: x})
                        yield tuple(submit_jobs(sub_fn, combos[1:]))

            # Run through nested structure retrieving results
            def get_results(futs, _l=0, _pl=0):
                len1 = (len(futs) == 1)
                for fut in progbar(futs, disable=(_pl >= progbars or len1),
                                   desc=fn_args[_l], **progbar_opts):
                    if _l == len(fn_args) - 1:
                        y = fut.result()
                        yield y if split else (y,)
                    else:
                        _npl = _pl if len1 else _pl+1
                        yield tuple(zip(*get_results(fut, _l + 1, _npl)))

            futures = tuple(submit_jobs(fn, combos))
            results = tuple(zip(*get_results(futures)))

    # Evaluate combos sequentially
    else:
        def get_ouputs_seq(fn, combos, _pl=0):
            arg, inputs = combos[0]
            len1 = (len(inputs) == 1)
            for x in progbar(inputs, disable=(_pl >= progbars or len1),
                             desc=arg, **progbar_opts):
                if len(combos) == 1:
                    yield fn(**{arg: x}) if split else [fn(**{arg: x})]
                else:
                    sub_fn = functools.partial(fn, **{arg: x})
                    _npl = (_pl if len1 else _pl+1)
                    yield tuple(zip(*get_ouputs_seq(sub_fn, combos[1:], _npl)))

        results = tuple(zip(*get_ouputs_seq(fn, combos)))

    return results if split else results[0]


def combos_to_ds(results, combos, var_names, var_dims=None, var_coords=None,
                 constants=None, attrs=None):
    """ Convert the output of combo_runner into a `xarray.Dataset`

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
    combos = _parse_combos(combos)
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

    if var_coords is None:
        var_coords = dict()

    # Set dataset coordinates
    ds = xr.Dataset(coords={**dict(combos), **dict(var_coords)}, attrs=attrs)

    # TODO: add_to_ds
    # check if all coords are available
    # check if all values to be set are nan, or overwite is set.

    # Set Dataset dataarrays
    for vdata, vname, vdims in zip(results, var_names, var_dims):
        ds[vname] = (tuple(fn_args) + tuple(vdims), np.asarray(vdata))

    # Add non-coordinate constants to attrs
    if constants is not None:
        for constant, value in constants.items():
            if constant not in ds.coords:
                ds.attrs[constant] = value

    return ds


def combo_runner_to_ds(fn, combos, var_names, var_dims=None, var_coords=None,
                       constants=None, attrs=None, **combo_runner_settings):
    """ Evalute combos and output to a Dataset. """
    if var_coords is None:
        var_coords = dict()

    # Set split based on output var_names
    split = (False if isinstance(var_names, str) else
             False if len(var_names) == 1 else
             True)
    # Generate data for all combos
    results = combo_runner(fn, combos, split=split, constants=constants,
                           **combo_runner_settings)
    # Convert to dataset
    ds = combos_to_ds(results, combos,
                      var_names=var_names,
                      var_dims=var_dims,
                      var_coords=var_coords,
                      constants=constants,
                      attrs=attrs)
    return ds


# --------------------------------------------------------------------------- #
# CASE_RUNNER functions: for evaluating specific configurations of args       #
# --------------------------------------------------------------------------- #

def case_runner(fn, fn_args, cases,
                constants=None,
                split=False,
                progbars=0,
                parallel=False,
                processes=None,
                parallel_backend='MULTIPROCESSING',
                progbar_opts=None):
    """ Evaluate a function in many different configurations, optionally in
    parallel and or with live progress.

    Parameters
    ----------
        fn: function with which to evalute cases with
        fn_args: names of case arguments that fn takes
        cases: list settings that fn_args take
        constants: constant fn args that won't be iterated over
        split: whether to split fn's output into multiple lists
        progbars: whether to show (in this case only 1) progbar
        parallel: whether to evaluate cases in parallel
        processes: how any processes to use for parallel processing
        progbar_opts: options to send to progbar

    Returns
    -------
        results: list of fn output for each case
    """

    # Prepare Function
    if constants is not None:
        fn = functools.partial(fn, **dict(constants))

    # Prepate fn_args and values
    if isinstance(fn_args, str):
        fn_args = (fn_args,)
        cases = tuple((c,) for c in cases)

    if progbar_opts is None:
        progbar_opts = dict()

    # Evaluate configurations in parallel
    if parallel or (processes is not None and processes > 1):
        with Pool(n=processes, backend=parallel_backend) as pool:
            fut = tuple(pool.submit(fn, **dict(zip(fn_args, c)))
                        for c in cases)
            results = tuple(x.result() for x in progbar(fut, total=len(cases),
                                                        disable=progbars < 1))

    # Evaluate configurations sequentially
    else:
        results = tuple(fn(**{arg: y for arg, y in zip(fn_args, cnfg)})
                        for cnfg in progbar(cases, total=len(cases),
                                            disable=progbars < 1))

    return results if not split else tuple(zip(*results))


def minimal_covering_coords(cases):
    """ Take a list of cases and find the minimal covering set of coordinates
    with which to index all cases. Sort the coords if possible. """
    for x in zip(*cases):
        try:
            yield sorted(list(set(x)))
        except TypeError:  # unsortable
            yield list(set(x))


def all_missing_ds(coords, var_names, var_dims, var_types):
    """ Make a dataset whose data is all missing. """
    # Blank dataset with appropirate coordinates
    ds = xr.Dataset(coords=coords)
    for v_name, v_dims, v_type in zip(var_names, var_dims, var_types):
        shape = tuple(ds[d].size for d in v_dims)
        if v_type == int or v_type == float:
            # Warn about upcasting int to float?
            nodata = np.tile(np.nan, shape)
        elif v_type == complex:
            nodata = np.tile(np.nan + np.nan*1.0j, shape)
        else:
            nodata = np.tile(None, shape).astype(object)
        ds[v_name] = (v_dims, nodata)
    return ds


def cases_to_ds(results, fn_args, cases, var_names, var_dims=None,
                var_coords=None, add_to_ds=None, overwrite=False):
    """ Take a list of results and configurations that generate them and turn it
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

    if var_coords is None:
        var_coords = dict()

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
            if not overwrite:
                if not ds[vname].loc[dict(zip(fn_args, cfg))].isnull().all():
                    raise ValueError("Existing data and `overwrite` = False")
            try:
                len(x)
                ds[vname].loc[dict(zip(fn_args, cfg))] = np.asarray(x)
            except TypeError:
                ds[vname].loc[dict(zip(fn_args, cfg))] = x

    return ds


def case_runner_to_ds(fn, fn_args, cases, var_names, var_dims=None,
                      var_coords=None, add_to_ds=None, overwrite=False,
                      **case_runner_settings):
    """ Combination of `case_runner` and `cases_to_ds`. Takes a function and
    list of argument configurations and produces a `xarray.Dataset`.

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
        ds: dataset with minimal covering coordinates and all cases
            evaluated.
    """
    if var_coords is None:
        var_coords = dict()

    # Generate results
    results = case_runner(fn, fn_args, cases, **case_runner_settings)
    # Convert to xarray.Dataset
    ds = cases_to_ds(results, fn_args, cases,
                     var_names=var_names,
                     var_dims=var_dims,
                     var_coords=var_coords,
                     add_to_ds=add_to_ds,
                     overwrite=overwrite)
    return ds


# --------------------------------------------------------------------------- #
# Update or add new values                                                    #
# --------------------------------------------------------------------------- #

def find_missing_cases(ds, var_dims=None):
    """ Find all cases in a dataset with missing data.

    Parameters
    ----------
        ds: Dataset in which to find missing data
        var_dims: internal variable dimensions (i.e. to ignore)

    Returns
    -------
        (m_fn_args, m_cases): function arguments and missing cases.
    """
    # Parse var_dims
    var_dims = (() if var_dims is None else
                (var_dims,) if isinstance(var_dims, str) else
                var_dims)
    # Find all configurations
    fn_args = tuple(coo for coo in ds.coords if coo not in var_dims)
    var_names = tuple(ds.data_vars)
    all_cases = itertools.product(*(ds[arg].data for arg in fn_args))

    # Only return those corresponding to all missing data
    def gen_missing_list():
        for case in all_cases:
            sub_ds = ds.loc[dict(zip(fn_args, case))]
            if all(sub_ds[v].isnull().all() for v in var_names):
                yield case

    return fn_args, tuple(gen_missing_list())


def fill_missing_cases(ds, fn, var_names, var_dims=None, var_coords=None,
                       **case_runner_settings):
    """ Take a dataset and function etc. and fill its missing data in

    Parameters
    ----------
        ds: Dataset to analyse and fill
        fn: function to use to fill missing cases
        var_names: output variable names of function
        var_dims: output varialbe named dimensions of function
        var_coords: dictionary of coords for output dims
        **case_runner_settings: settings sent to `case_runner`
    Returns
    -------
        None: (filling performed in-place)
    """
    if var_coords is None:
        var_coords = dict()

    # Find missing cases
    fn_args, missing_cases = find_missing_cases(ds, var_dims)
    # Evaluate and add to Dataset
    case_runner_to_ds(fn, fn_args, missing_cases,
                      var_names=var_names,
                      var_dims=var_dims,
                      var_coords=var_coords,
                      add_to_ds=ds,
                      **case_runner_settings)
