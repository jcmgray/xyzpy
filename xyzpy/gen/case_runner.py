"""Functions for systematically evaluating a function over specific cases.
"""

import functools
import itertools

import numpy as np
import xarray as xr

from ..utils import progbar, _choose_executor_depr_pool
from .prepare import (
    _parse_fn_args,
    _parse_cases,
    _parse_case_results,
    _parse_var_names,
    _parse_var_dims,
    _parse_var_coords,
    _parse_constants,
    _parse_resources
)


from .combo_runner import _combo_runner


class SingleArgFn:

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, kws, **kwargs):
        return self.fn(**kws, **kwargs)


def _case_runner(fn, fn_args, cases, constants,
                 split=False,
                 parallel=False,
                 num_workers=None,
                 executor=None,
                 verbosity=1,
                 pool=None):
    """Core case runner, i.e. without parsing of arguments.
    """
    executor = _choose_executor_depr_pool(executor, pool)

    # Turn the function into a single arg function to send to combo_runner
    sfn = SingleArgFn(fn)

    if isinstance(cases[0], dict):
        combos = (('kws', cases),)
    else:
        combos = (('kws', [dict(zip(fn_args, case)) for case in cases]),)

    return _combo_runner(sfn, combos,
                         constants=constants,
                         split=split,
                         parallel=parallel,
                         num_workers=num_workers,
                         executor=executor,
                         verbosity=verbosity)


def case_runner(fn, fn_args, cases,
                constants=None,
                split=False,
                parallel=False,
                executor=None,
                num_workers=None,
                verbosity=1,
                pool=None):
    """Evaluate a function in many different configurations, optionally in
    parallel and or with live progress.

    Parameters
    ----------
    fn : callable
        Function with which to evalute cases with
    fn_args : tuple
        Names of case arguments that fn takes
    cases : tuple of tuple
        List settings that ``fn_args`` take.
    constants : dict, optional
        See :func:`~xyzpy.combo_runner`.
    split : bool, optional
        See :func:`~xyzpy.combo_runner`.
    parallel : bool, optional
        See :func:`~xyzpy.combo_runner`.
    executor : executor-like pool, optional
        See :func:`~xyzpy.combo_runner`.
    num_workers : int, optional
        See :func:`~xyzpy.combo_runner`.
    verbosity : {0, 1, 2}, optional
        See :func:`~xyzpy.combo_runner`.

    Returns
    -------
        results : list of fn output for each case
    """
    executor = _choose_executor_depr_pool(executor, pool)

    # Prepare fn_args and values
    fn_args = _parse_fn_args(fn, fn_args)
    cases = _parse_cases(cases)
    constants = _parse_constants(constants)

    return _case_runner(fn, fn_args, cases,
                        constants=constants,
                        split=split,
                        parallel=parallel,
                        num_workers=num_workers,
                        executor=executor,
                        verbosity=verbosity)


def find_union_coords(cases):
    """Take a list of cases and find the union of coordinates
    with which to index all cases. Sort the coords if possible.
    """
    for x in zip(*cases):
        try:
            yield sorted(set(x))
        except TypeError:  # unsortable
            yield list(set(x))


def all_missing_ds(coords, var_names, all_dims, var_types, attrs=None):
    """Make a dataset whose data is all missing.

    Parameters
    ----------
    coords : dict
        coordinates of dataset
    var_names : tuple
        names of each variable in dataset
    all_dims : tuple
        corresponding list of dimensions for each variable
    var_types : tuple
        corresponding list of types for each variable
    """
    # Blank dataset with appropirate coordinates
    ds = xr.Dataset(coords=coords, attrs=attrs)

    for v_name, v_dims, v_type in zip(var_names, all_dims, var_types):

        shape = tuple(ds[d].size for d in v_dims)

        if v_type == int or v_type == float:
            # Warn about upcasting int to float?
            nodata = np.tile(np.nan, shape)
        elif v_type == complex:
            nodata = np.tile(np.nan + np.nan * 1.0j, shape)
        else:
            nodata = np.tile(None, shape).astype(object)
        ds[v_name] = (v_dims, nodata)

    return ds


def _cases_to_df(results, fn_args, cases, var_names, var_dims=None,
                 var_coords=None, constants=None, attrs=None):
    """Turn cases and results into a ``pandas.DataFrame``.
    """
    import pandas as pd

    if var_names is None:
        raise ValueError("Can't coerce dataset output into dataframe.")
    if var_dims is not None and any(var_dims.values()):
        raise ValueError("Dataframes don't support internal dimensions.")
    if var_coords:
        raise ValueError("Dataframes don't support internal dimensions.")
    if attrs:
        raise ValueError("Dataframes don't support attributes.")

    results = _parse_case_results(results, var_names)
    N = len(results)

    df = pd.DataFrame(dict(itertools.chain(
        # the function argument columns
        zip(fn_args, zip(*cases)),
        zip(constants.keys(), ([c] * N for c in constants.values())),
        # the function result columns
        zip(var_names, zip(*results)),
    )))

    return df


def _cases_to_ds(results, fn_args, cases, var_names, add_to_ds=None,
                 var_dims=None, var_coords=None, constants=None, attrs=None,
                 overwrite=False):
    """ Take a list of results and configurations that generate them and turn
    it into a `xarray.Dataset`.

    Parameters
    ----------
    results : list[tuple]
        List(s) of results of len(cases), e.g. generated by `case_runner`.
    fn_args : list[str]
        Arguments used in function that generated the results.
    cases : list[tuple]
        List of configurations used to generate results.
    var_names : list[str]
        Name(s) of output variables for a single result.
    var_dims : dict[str: tuple[str]]
        The list of named coordinates for each single result variable, i.e.
        coordinates not generated by the combo_runner.
    var_coords : dict[str, array]
        dict of values for those coordinates if custom ones are desired.

    Returns
    -------
    ds : xarray.Dataset
        Dataset holding all results, with coordinates described by cases

    Notes
    -----
    1. Many data types have to be converted to object in order for the
        missing values to be represented by NaNs.
    """
    results = _parse_case_results(results, var_names)

    # check if Dataset already returned -> can just merge
    if isinstance(results[0][0], (xr.Dataset, xr.DataArray)):
        ds0 = (add_to_ds,) if add_to_ds is not None else ()
        return xr.merge([*ds0, *(r[0] for r in results)])

    if add_to_ds is not None:
        ds = add_to_ds
    else:
        # need to find minimal covering set of coordinates for fn_args

        if isinstance(cases[0], dict):
            fn_args = tuple(cases[0].keys())
            cases = tuple(tuple(c[a] for a in fn_args) for c in cases)

        case_coords = dict(zip(fn_args, find_union_coords(cases)))

        # Create new, 'all missing' dataset if required
        ds = all_missing_ds(
            coords={**case_coords, **var_coords},
            var_names=var_names, attrs=attrs,
            all_dims=tuple(fn_args + var_dims[k] for k in var_names),
            var_types=(np.asarray(x).dtype for x in results[0])
        )

        if constants:
            newattrs = {k: v for k, v in constants.items() if k not in ds.dims}
            ds.attrs.update(newattrs)

    # Go through cases, overwriting nan with results
    for res, cfg in zip(results, cases):

        cfg = [[c] for c in cfg]

        for vname, x in zip(var_names, res):
            loc = dict(zip(fn_args, cfg))

            if not overwrite:
                if not ds[vname].loc[loc].isnull().all():
                    raise ValueError(
                        "Existing data for variable {} at position {} and "
                        "`overwrite` = False.".format(vname, loc))
            try:
                len(x)
                ds[vname].loc[loc] = np.asarray(x)

            except (TypeError, ValueError):
                ds[vname].loc[loc] = x

    return ds


def case_runner_to_ds(fn, fn_args, cases, var_names,
                      var_dims=None,
                      var_coords=None,
                      constants=None,
                      resources=None,
                      attrs=None,
                      add_to_ds=None,
                      overwrite=False,
                      parse=True,
                      to_df=False,
                      **case_runner_settings):
    """ Combination of `case_runner` and `_cases_to_ds`. Takes a function and
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
    constants : dict_like, optional
    resources :  dict_like, optional
    attrs : dict_like, optional
    add_to_ds : bool, optional
    overwrite : bool, optional
    parse : bool, optional
    to_df : bool, optional

    Returns
    -------
    ds : xarray.Dataset
        Dataset with minimal covering coordinates and all cases
        evaluated.
    """
    if parse:
        fn_args = _parse_fn_args(fn, fn_args)
        cases = _parse_cases(cases)
        constants = _parse_constants(constants)
        resources = _parse_resources(resources)
        var_names = _parse_var_names(var_names)
        var_dims = _parse_var_dims(var_dims, var_names)
        var_coords = _parse_var_coords(var_coords)

    # Generate results
    results = _case_runner(fn, fn_args, cases,
                           constants={**constants, **resources},
                           **case_runner_settings)

    if to_df:
        # Convert to pandas.DataFrame
        ds = _cases_to_df(results, fn_args, cases,
                          var_names=var_names,
                          var_dims=var_dims,
                          var_coords=var_coords,
                          constants=constants,
                          attrs=attrs)
    else:
        # Convert to xarray.Dataset
        ds = _cases_to_ds(results, fn_args, cases,
                          var_names=var_names,
                          var_dims=var_dims,
                          var_coords=var_coords,
                          constants=constants,
                          attrs=attrs,
                          add_to_ds=add_to_ds,
                          overwrite=overwrite)
    return ds


case_runner_to_df = functools.partial(case_runner_to_ds, to_df=True)


# --------------------------------------------------------------------------- #
# Update or add new values                                                    #
# --------------------------------------------------------------------------- #

def find_missing_cases(ds, ignore_dims=None, show_progbar=False):
    """Find all cases in a dataset with missing data.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset in which to find missing data
    ignore_dims : set (optional)
        internal variable dimensions (i.e. to ignore)
    show_progbar : bool (optional)
        Show the current progress

    Returns
    -------
    missing_fn_args, missing_cases :
        Function arguments and missing cases.
    """
    # Parse ignore_dims
    ignore_dims = ({ignore_dims} if isinstance(ignore_dims, str) else
                   set(ignore_dims) if ignore_dims else set())

    # Find all configurations
    fn_args = tuple(coo for coo in ds.dims if coo not in ignore_dims)
    var_names = tuple(ds.data_vars)
    all_cases = itertools.product(*(ds[arg].data for arg in fn_args))

    # Only return those corresponding to all missing data
    def gen_missing_list():
        for case in progbar(all_cases, disable=not show_progbar):
            sub_ds = ds.loc[dict(zip(fn_args, case))]
            if all(sub_ds[v].isnull().all() for v in var_names):
                yield case

    return fn_args, tuple(gen_missing_list())


def fill_missing_cases(ds, fn, var_names,
                       var_dims=None,
                       var_coords=None,
                       constants=None,
                       resources=None,
                       **case_runner_settings):
    """ Take a dataset and function etc. and fill its missing data in

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to analyse and fill
    fn : callable
        Function to use to fill missing cases
    var_names : tuple
        Output variable names of function
    var_dims : dict
        Output variabe named dimensions of function
    var_coords : dict
        Dictionary of coords for output dims
    case_runner_settings:
        Supplied to :func:`~xyzpy.case_runner`.

    Returns
    -------
    xarray.Dataset
    """
    var_names = _parse_var_names(var_names)
    var_dims = _parse_var_dims(var_dims, var_names)
    var_coords = _parse_var_coords(var_coords)
    constants = _parse_constants(constants)
    resources = _parse_resources(resources)

    # Gather all internal dimensions
    ignore_dims = set(itertools.chain.from_iterable(var_dims.values()))

    # Find missing cases
    fn_args, missing_cases = find_missing_cases(ds, ignore_dims=ignore_dims)

    # Generate missing results
    results = _case_runner(fn, fn_args, missing_cases,
                           constants={**constants, **resources},
                           **case_runner_settings)

    # Add to dataset
    return _cases_to_ds(results, fn_args, missing_cases,
                        var_names=var_names,
                        var_dims=var_dims,
                        var_coords=var_coords,
                        add_to_ds=ds)
