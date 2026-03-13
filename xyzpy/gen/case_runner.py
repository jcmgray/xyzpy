"""Functions for systematically evaluating a function over specific cases."""

import functools
import itertools

import numpy as np
import xarray as xr

from .combo_runner import (
    combo_runner_core,
    combo_runner_to_ds,
)
from .prepare import (
    parse_cases,
    parse_combos,
    parse_constants,
    parse_fn_args,
    parse_resources,
    parse_var_coords,
    parse_var_dims,
    parse_var_names,
)


def case_runner(
    fn,
    fn_args,
    cases,
    combos=None,
    constants=None,
    split=False,
    shuffle=False,
    parse=True,
    parallel=False,
    executor=None,
    num_workers=None,
    verbosity=1,
):
    """Simple case runner that outputs the raw tuple of results.

    Parameters
    ----------
    fn : callable
        Function with which to evalute cases with
    fn_args : tuple
        Names of case arguments that fn takes, can be ``None`` if each case is
        a ``dict``.
    cases : iterable[tuple] or iterable[dict]
        List of specific configurations that ``fn_args`` should take. If
        ``fn_args`` is ``None``, each case should be a ``dict``.
    combos : dict_like[str, iterable], optional
        Optional specification of sub-combinations.
    constants : dict, optional
        Constant function arguments.
    split : bool, optional
        See :func:`~xyzpy.combo_runner`.
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
        results : list of fn output for each case
    """
    if parse:
        # Prepare fn_args and values
        fn_args = parse_fn_args(fn, fn_args)
        cases = parse_cases(cases, fn_args)
        combos = parse_combos(combos)
        constants = parse_constants(constants)

    return combo_runner_core(
        fn,
        cases=cases,
        combos=combos,
        constants=constants,
        parallel=parallel,
        num_workers=num_workers,
        executor=executor,
        verbosity=verbosity,
        split=split,
        flat=True,
        shuffle=shuffle,
    )


def case_runner_to_ds(
    fn,
    fn_args,
    cases,
    var_names,
    var_dims=None,
    var_coords=None,
    combos=None,
    constants=None,
    resources=None,
    attrs=None,
    shuffle=False,
    to_df=False,
    parse=True,
    parallel=False,
    num_workers=None,
    executor=None,
    verbosity=1,
):
    """Takes a list of ``cases`` to run ``fn`` over, possibly in parallel, and
    outputs a :class:`xarray.Dataset`.

    Parameters
    ----------
    fn : callable
        Function to evaluate.
    fn_args : str or iterable[str]
        Names and order of arguments to ``fn``, can be ``None`` if ``cases``
        are supplied as dicts.
    cases: iterable[tuple] or iterable[dict]
        List of configurations used to generate results.
    var_names : str or iterable of str
        Variable name(s) of the output(s) of ``fn``.
    var_dims : sequence of either strings or string sequences, optional
        'Internal' names of dimensions for each variable, the values for
        each dimension should be contained as a mapping in either
        `var_coords` (not needed by `fn`) or `constants` (needed by `fn`).
    var_coords : mapping, optional
        Mapping of extra coords the output variables may depend on.
    combos : dict_like[str, iterable], optional
        If specified, run all combinations of some arguments in these mappings.
    constants : mapping, optional
        Arguments to `fn` which are not iterated over, these will be
        recorded either as attributes or coordinates if they are named
        in `var_dims`.
    resources : mapping, optional
        Like `constants` but they will not be recorded.
    attrs : mapping, optional
        Any extra attributes to store.
    shuffle : bool or int, optional
        If given, compute the results in a random order (using ``random.seed``
        and ``random.shuffle``), which can be helpful for distributing
        resources when not all cases are computationally equal.
    parse : bool, optional
        Whether to perform parsing of the inputs arguments.
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
    ds : xarray.Dataset
        Dataset with minimal covering coordinates and all cases
        evaluated.
    """
    if parse:
        # Prepare fn_args and values
        fn_args = parse_fn_args(fn, fn_args)
        cases = parse_cases(cases, fn_args)
        combos = parse_combos(combos)
        constants = parse_constants(constants)
        resources = parse_resources(resources)
        var_names = parse_var_names(var_names)
        var_dims = parse_var_dims(var_dims, var_names=var_names)
        var_coords = parse_var_coords(var_coords)

    return combo_runner_to_ds(
        fn=fn,
        combos=combos,
        var_names=var_names,
        var_dims=var_dims,
        var_coords=var_coords,
        cases=cases,
        constants=constants,
        resources=resources,
        attrs=attrs,
        shuffle=shuffle,
        to_df=to_df,
        parallel=parallel,
        num_workers=num_workers,
        executor=executor,
        verbosity=verbosity,
        parse=False,
    )


case_runner_to_df = functools.partial(case_runner_to_ds, to_df=True)


# --------------------------------------------------------------------------- #
# Update or add new values                                                    #
# --------------------------------------------------------------------------- #


def is_case_missing(ds, setting, method="isnull"):
    """Does the dataset or dataarray ``ds`` not contain any non-null data for
    single location ``setting``?

    Note that this only returns true if *all* data across *all* variables is
    completely missing at the location.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Dataset or dataarray to look in.
    setting : dict[str, hashable]
        Coordinate location to check.

    Returns
    -------
    missing : bool
    """
    import numpy as np

    try:
        sds = ds.sel(setting)

        if method == "isnull":
            sds = sds.isnull()
        elif method == "isfinite":
            sds = ~np.isfinite(sds)
        else:
            raise ValueError("Unknown method: {}".format(method))

        nds = sds.all()
    except KeyError:
        # coordinates not present at all
        return True

    try:
        # cocatenate answer for each variable if Dataset
        nds = nds.to_array().all()
    except AttributeError:
        # already a DataArray
        pass

    return nds.item()


def parse_into_cases(combos=None, cases=None, ds=None, method="isnull"):
    """Convert maybe ``combos`` and maybe ``cases`` to a single list of
    ``cases`` only, also optionally filtering based on whether any data at each
    location is already present in Dataset or DataArray ``ds``.

    Note that this only checks whether *all* data across *all* variables is
    completely missing at the location. To check against a single variable only
    simply supply a DataArray instead of a Dataset, e.g. ``ds=ds["var_name"]``.

    Parameters
    ----------
    combos : dict_like[str, iterable], optional
        Parameter combinations.
    cases : iterable[dict], optional
        Parameter configurations.
    ds : xarray.Dataset or xarray.DataArray, optional
        Dataset or DataArray in which to check for existing data.
    method : {"isnull", "isfinite"}, optional
        How to determine whether data is missing when ``ds`` is supplied.
        "isnull" checks for null/nan values, while "isfinite" checks for all
        non-finite values (i.e. inf or nan).

    Returns
    -------
    new_cases : iterable[dict]
        The combined and possibly filtered list of cases.
    """
    if combos is None:
        combos = {}
    elif not isinstance(combos, dict):
        combos = dict(combos)

    if combos:
        combo_keys, combo_values = zip(*combos.items())
    else:
        combo_keys, combo_values = [], []

    if cases is None:
        cases = [{}]

    new_cases = []

    if ds is None:
        # we can just flatten all cases and combos
        for case in cases:
            for combo in itertools.product(*combo_values):
                setting = case | dict(zip(combo_keys, combo))
                new_cases.append(setting)
        return new_cases

    # else we need to check against existing data
    existing_coords = {
        dim: {v: i for i, v in enumerate(ds.coords[dim].values)}
        for dim in ds.dims
    }

    # first we sort into cases which are outside of the existing coordinates
    cases_inside = []
    ilocs = []
    for case in cases:
        for combo in itertools.product(*combo_values):
            setting = case | dict(zip(combo_keys, combo))
            # build its index, or break if outside
            iloc = []
            for dim, val in setting.items():
                idx = existing_coords[dim].get(val, None)
                if idx is None:
                    # missing value, don't need to check actual data
                    new_cases.append(setting)
                    break
                iloc.append(idx)
            else:
                # no break - location is in bounds
                cases_inside.append(setting)
                ilocs.append(iloc)

    # now we check if the actual data for the cases inside is finite
    # this fancy index extracts the values at the inside case locations
    indices = tuple(np.array(col) for col in zip(*ilocs))

    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()

    if method == "isnull":
        ds = ds.isnull()
    elif method == "isfinite":
        ds = ~np.isfinite(ds)
    else:
        raise ValueError("Unknown method: {}".format(method))

    missing = None
    for v in ds.data_vars:
        missing_v = ds[v].values[indices]
        if missing_v.ndim > 1:
            # sub-coordinates, we need to reduce over
            missing_v = missing_v.reshape(missing_v.shape[0], -1).any(axis=1)

        # reduce across variables, requiring *all* to be missing
        if missing is None:
            missing = missing_v
        else:
            missing = missing & missing_v

    # turn into indices of missing cases, and add to missing list
    new_cases.extend(cases_inside[i] for i in np.nonzero(missing)[0])

    return new_cases


def find_missing_cases(ds, ignore_dims=None, method="isnull"):
    """Find all cases in a dataset or DataArray with missing data.

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
        Dataset or DataArray in which to find missing data
    ignore_dims : set, optional
        Internal variable dimensions (i.e. to ignore). By default (None) this
        is set to any dimensions that don't appear on all variables.

    Returns
    -------
    cases_missing : iterable[dict]
        List of cases with missing data, where each case is a dict mapping from
        dimension name to coordinate value.
    """
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset()

    if ignore_dims is None:
        # default to ignoring any dimensions that don't appear on all variables
        ignore_dims = set(
            dim
            for dim in ds.dims
            if not all(dim in ds[v].dims for v in ds.data_vars)
        )
    elif isinstance(ignore_dims, str):
        ignore_dims = {ignore_dims}
    elif ignore_dims:
        ignore_dims = set(ignore_dims)
    else:
        ignore_dims = set()

    combos = {
        dim: ds.coords[dim].values for dim in ds.dims if dim not in ignore_dims
    }

    return parse_into_cases(combos=combos, ds=ds, method=method)
