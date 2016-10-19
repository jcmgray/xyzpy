"""Functions for systematically evaluating a function over all combinations.
"""
# TODO: allow combo_runner_to_ds to use output vars as coords --------------- #

import xarray as xr
import numpy as np
from dask.delayed import delayed, compute

from ..utils import _get_fn_name, prod, progbar, update_upon_eval, unzip
from ..parallel import _dask_get, DaskTqdmProgbar
from .prepare import (
    _parse_var_names,
    _parse_var_dims,
    _parse_constants,
    _parse_resources,
    _parse_progbar_opts,
    _parse_var_coords,
    _parse_combos,
)


def _nested_submit(fn, combos, kwds, delay=False):
    """Recursively submit jobs as delayed objects.

    Parameters
    ----------
        fn : callable
            Function to submit jobs to.
        combos : tuple mapping individual fn arguments to iterable of values
            Mapping of each argument and all its possible values.
        kwds : dict
            Constant keyword arguments not to iterate over.
        delay : bool
            Whether to wrap the function as `delayed` (for parallel eval).

    Returns
    -------
    results : list
        Nested lists of results.
    """
    arg, inputs = combos[0]
    if len(combos) == 1:
        if delay:
            return [delayed(fn)(**kwds, **{arg: x}) for x in inputs]
        else:
            return [fn(**kwds, **{arg: x}) for x in inputs]
    else:
        return [_nested_submit(fn, combos[1:], {**kwds, arg: x}, delay=delay)
                for x in inputs]


# Core Combo Runner --------------------------------------------------------- #

def _combo_runner(fn, combos, constants=None,
                  split=False,
                  parallel=False,
                  num_workers=None,
                  scheduler='t',
                  hide_progbar=False,
                  progbar_opts=None):
    """Core combo runner, i.e. no parsing of arguments.
    """
    fn_name = _get_fn_name(fn)

    # Evaluate combos in parallel
    if parallel or num_workers:
        with DaskTqdmProgbar(fn_name, disable=hide_progbar, **progbar_opts):
            jobs = _nested_submit(fn, combos, constants, delay=True)
            if scheduler and isinstance(scheduler, str):
                scheduler = _dask_get(scheduler, num_workers=num_workers)
            results = compute(*jobs, get=scheduler, num_workers=num_workers)

    # Evaluate combos sequentially
    else:
        n = prod(len(x) for _, x in combos)
        with progbar(total=n, disable=hide_progbar, **progbar_opts) as p:
            # Wrap the function such that the progbar is updated upon each call
            fn = update_upon_eval(fn, p)
            results = _nested_submit(fn, combos, constants)

    return list(unzip(results, len(combos))) if split else results


def combo_runner(fn, combos, constants=None,
                 split=False,
                 parallel=False,
                 num_workers=None,
                 scheduler='t',
                 hide_progbar=False,
                 progbar_opts=None):
    """Take a function fn and analyse it over all combinations of named
    variables' values, optionally showing progress and in parallel.

    Parameters
    ----------
        fn : callable
            Function to analyse.
        combos : mapping of individual fn arguments to iterable of values
            All combinations of each argument will be calculated. Each
            argument range thus gets a dimension in the output array(s).
        constants : dict
            List of tuples/dict of *constant* fn argument mappings.
        split : bool
            Whether to split into multiple output arrays or not.
        hide_progbar : bool
            Whether to disable the progress bar.
        parallel : bool
            Process combos in parallel, default number of workers picked.
        num_workers : int
            Explicitly choose how many workers to use, None for automatic.
        scheduler : str or dask.get instance
            Specify scheduler to use for the parallel work.
        progbar_opts : dict
            Options for the progress bar.

    Returns
    -------
        data:
            list of result arrays, each with all param combinations in nested
            tuples.
    """
    # Prepare combos
    combos = _parse_combos(combos)
    constants = _parse_constants(constants)
    progbar_opts = _parse_progbar_opts(progbar_opts)

    # Submit to core combo runner
    return _combo_runner(fn=fn, combos=combos, constants=constants,
                         split=split, parallel=parallel,
                         num_workers=num_workers, scheduler=scheduler,
                         hide_progbar=hide_progbar, progbar_opts=progbar_opts)


def _combos_to_ds(results, combos, var_names, var_dims, var_coords,
                  constants=None, attrs=None):
    """Convert the output of combo_runner into a `xarray.Dataset`

    Parameters
    ----------
        results :
            array(s) of dimension `len(combos)`
        combos :
            list of tuples of form ((variable_name, [values]), ...) with
            which `results` was generated.
        var_names : list-like of str or 2-tuples.
            name(s) of output variables for a single result
        var_dims :
            the list of named coordinates for each single result
            variable, i.e. coordinates not generated by the combo_runner
        var_coords :
            dict of values for those coordinates if custom ones are
            desired.

    Returns
    -------
        xarray.Dataset
    """
    fn_args = tuple(x for x, _ in combos)
    if len(var_names) == 1:
        results = (results,)

    # Set dataset coordinates
    ds = xr.Dataset(
        coords={**dict(combos), **dict(var_coords)},
        data_vars={name: (fn_args + var_dims[name], np.asarray(data))
                   for data, name in zip(results, var_names)},
        attrs=attrs)

    # TODO: merge into add_to_ds

    # Add constants to attrs, but filter out those which are already coords
    if constants:
        ds.attrs.update({k: v for k, v in constants.items()
                         if k not in ds.coords})

    return ds


def combo_runner_to_ds(fn, combos, var_names,
                       var_dims=None,
                       var_coords=None,
                       constants=None,
                       resources=None,
                       attrs=None,
                       progbar_opts=None,
                       **combo_runner_settings):
    """Evaluate a function over all combinations and output to a Dataset.

    Parameters
    ----------
        fn: callable
            Function to evaluate.
        combos: mapping
            Mapping of each individual function argument to iterable of values.
        var_names: str or iterable of strings
            Variable name(s) of the output(s) of `fn`.
        var_dims: iterable of strings or iterable of iterable of strings
            'Internal' names of dimensions for each variable, the values for
            each dimension should be contiained as a mapping in either
            `var_coords` (not needed by `fn`) or `constants` (needed by `fn`).
        var_coords: mapping
            Mapping of extra coords the output variables may depend on.
        constants: mapping
            Arguments to `fn` which are not iterated over, these will be
            recorded either as attributes or coordinates if they are used.
        resources: mapping
            Like `constants` but they will not be recorded.
        attrs: mapping
            Any extra attributes to store.
        **combo_runner_settings: dict-like
            Arguments supplied to `combo_runner`.

    Returns
    -------
        xarray.Dataset
    """
    # Parse inputs
    combos = _parse_combos(combos)
    var_names = _parse_var_names(var_names)
    var_dims = _parse_var_dims(var_dims, var_names=var_names)
    var_coords = _parse_var_coords(var_coords)
    constants = _parse_constants(constants)
    resources = _parse_resources(resources)
    progbar_opts = _parse_progbar_opts(progbar_opts)

    # Set split based on var_names format
    split = not (isinstance(var_names, str) or len(var_names) == 1)

    # Generate data for all combos
    results = _combo_runner(fn, combos, constants={**resources, **constants},
                            split=split, progbar_opts=progbar_opts,
                            **combo_runner_settings)
    # Convert to dataset
    ds = _combos_to_ds(results, combos, var_names=var_names, var_dims=var_dims,
                       var_coords=var_coords, constants=constants, attrs=attrs)
    return ds
