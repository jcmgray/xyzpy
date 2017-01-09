"""Functions for systematically evaluating a function over all combinations.
"""
# TODO: allow/encourage results to be a dict? ------------------------------- #
# TODO: add_to_ds, skip_completed? ------------------------------------------ #
# TODO: add straight to array, ds ... --------------------------------------- #
# TODO: allow combo_runner_to_ds to use output vars as coords --------------- #
# TODO: better checks for var_name compatilibtiy with fn_args eg. ----------- #

import xarray as xr
import numpy as np
from dask.delayed import delayed, compute
import distributed
from ..utils import (
    unzip,
    flatten,
    _get_fn_name,
    prod,
    progbar,
    update_upon_eval,
)
from .dask_stuff import (
    DaskTqdmProgbar,
    dask_scheduler_get,
    distributed_getter,
    distributed_getter_stored,
    make_distributed_submit_with_callback,
    make_distributed_submit_with_callback_replicate,
)
from .prepare import (
    _parse_var_names,
    _parse_var_dims,
    _parse_constants,
    _parse_resources,
    _parse_var_coords,
    _parse_combos,
    _parse_combo_results,
)


def _default_submit(pool, fn, *args, **kwds):
    """Default method for submitting to a pool.
    """
    try:
        future = pool.submit(fn, *args, **kwds)
    except AttributeError:
        future = pool.apply_async(fn, *args, **kwds)
    return future


def nested_submit(fn, combos, kwds, delay=False, pool=None,
                  submitter=_default_submit):
    """Recursively submit jobs directly, as delayed objects or to a pool.

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
        pool : Executor pool
            Pool-executor-like class implementing a submit method.

    Returns
    -------
    results : list
        Nested lists of results.
    """
    arg, inputs = combos[0]
    if len(combos) == 1:
        if pool:
            return [submitter(pool, fn, **kwds, **{arg: x}) for x in inputs]
        elif delay:
            return [delayed(fn, pure=True)(**kwds, **{arg: x}) for x in inputs]
        else:
            return [fn(**kwds, **{arg: x}) for x in inputs]
    else:
        return [nested_submit(fn, combos[1:], {**kwds, arg: x}, delay=delay,
                              pool=pool) for x in inputs]


def getter_with_progress(pbar=None):
    """
    """
    def getter(future):
        try:
            res = future.result()
        except AttributeError:
            res = future.get()
        pbar.update()
        return res

    return getter


def nested_get(futures, ndim, getter):
    """Recusively get results from nested futures.
    """
    return ([getter(fut) for fut in futures] if ndim == 1 else
            [nested_get(fut, ndim - 1, getter) for fut in futures])


def _mpi_combo_runner_pool(fn, combos, constants, hide_progbar, n,
                           num_workers=None):
    from mpi4py.futures import MPIPoolExecutor
    with progbar(total=n, disable=hide_progbar) as pbar:
        getter = getter_with_progress(pbar)
        with MPIPoolExecutor(num_workers) as pool:
            futures = nested_submit(fn, combos, constants, pool=pool)
            results = nested_get(futures, len(combos), getter)
    return results


def _combo_runner(fn, combos, constants,
                  split=False,
                  parallel=False,
                  num_workers=None,
                  scheduler='m',
                  pool=None,
                  hide_progbar=False):
    """Core combo runner, i.e. no parsing of arguments.
    """
    n = prod(len(x) for _, x in combos)
    ndim = len(combos)

    # Use a supplied pool to run combos
    if isinstance(pool, distributed.Client):
        with progbar(total=n, disable=hide_progbar) as pbar:
            if parallel == 'as_completed':
                futures = nested_submit(fn, combos, constants, pool=pool)
                for f in distributed.as_completed(flatten(futures, ndim)):
                    f._stored_result = f.result()
                    f.release()
                    pbar.update()
                results = nested_get(futures, ndim, distributed_getter_stored)

            elif parallel == 'callback':
                submitter = make_distributed_submit_with_callback(pbar)
                futures = nested_submit(fn, combos, constants, pool=pool,
                                        submitter=submitter)
                results = nested_get(futures, ndim, distributed_getter_stored)

            elif parallel == 'replicate':
                submitter = make_distributed_submit_with_callback_replicate(
                    pbar, pool)
                futures = nested_submit(fn, combos, constants, pool=pool,
                                        submitter=submitter)
                results = nested_get(futures, ndim, lambda f: f.result())

            else:
                futures = nested_submit(fn, combos, constants, pool=pool)
                getter = update_upon_eval(distributed_getter, pbar)
                results = nested_get(futures, ndim, getter)

    elif pool is not None:
        with progbar(total=n, disable=hide_progbar) as pbar:
            futures = nested_submit(fn, combos, constants, pool=pool)
            getter = getter_with_progress(pbar)
            results = nested_get(futures, ndim, getter)

    # Spawn an mpi pool to run combos
    elif parallel == 'mpi_spawn':
        results = _mpi_combo_runner_pool(fn, combos, constants, hide_progbar,
                                         n=n, num_workers=num_workers)

    # Evaluate combos using dask
    elif parallel or num_workers:
        fn_name = _get_fn_name(fn)
        with DaskTqdmProgbar(fn_name, disable=hide_progbar):
            jobs = nested_submit(fn, combos, constants, delay=True)
            if scheduler and isinstance(scheduler, str):
                scheduler = dask_scheduler_get(scheduler,
                                               num_workers=num_workers)
            results = compute(*jobs, get=scheduler, num_workers=num_workers)

    # Evaluate combos sequentially
    else:
        with progbar(total=n, disable=hide_progbar) as p:
            # Wrap the function such that the progbar is updated upon each call
            fn = update_upon_eval(fn, p)
            results = nested_submit(fn, combos, constants)

    return list(unzip(results, ndim)) if split else results


def combo_runner(fn, combos, constants=None,
                 split=False,
                 parallel=False,
                 scheduler='m',
                 pool=None,
                 num_workers=None,
                 hide_progbar=False):
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

    Returns
    -------
        data:
            list of result arrays, each with all param combinations in nested
            tuples.
    """
    # Prepare combos
    combos = _parse_combos(combos)
    constants = _parse_constants(constants)

    # Submit to core combo runner
    return _combo_runner(fn, combos,
                         constants=constants,
                         split=split,
                         parallel=parallel,
                         scheduler=scheduler,
                         pool=pool,
                         num_workers=num_workers,
                         hide_progbar=hide_progbar)


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
    results = _parse_combo_results(results, var_names)
    # Set dataset coordinates
    ds = xr.Dataset(coords={**dict(combos), **dict(var_coords)}, attrs=attrs,
                    data_vars={name: (fn_args + var_dims[name],
                                      np.asarray(data))
                               for data, name in zip(results, var_names)})
    # Add constants to attrs, but filter out those which are already coords
    if constants:
        ds.attrs.update({k: v for k, v in constants.items()
                         if k not in ds.dims})
    return ds


def combo_runner_to_ds(fn, combos, var_names,
                       var_dims=None,
                       var_coords=None,
                       constants=None,
                       resources=None,
                       attrs=None,
                       parse=True,
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
    if parse:
        combos = _parse_combos(combos)
        var_names = _parse_var_names(var_names)
        var_dims = _parse_var_dims(var_dims, var_names=var_names)
        var_coords = _parse_var_coords(var_coords)
        constants = _parse_constants(constants)
        resources = _parse_resources(resources)

    # Generate data for all combos
    results = _combo_runner(fn, combos, constants={**resources, **constants},
                            split=len(var_names) > 1,
                            **combo_runner_settings)
    # Convert to dataset
    ds = _combos_to_ds(results, combos,
                       var_names=var_names,
                       var_dims=var_dims,
                       var_coords=var_coords,
                       constants=constants,
                       attrs=attrs)
    return ds
