"""Functions for systematically evaluating a function over all combinations.
"""
import functools
from concurrent import futures as cf
import numpy as np
import xarray as xr

from ..utils import (
    unzip,
    flatten,
    prod,
    progbar,
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


def default_submitter(pool, fn, *args, **kwds):
    """Default method for submitting to a pool.
    """
    try:
        future = pool.submit(fn, *args, **kwds)
    except AttributeError:
        future = pool.apply_async(fn, *args, **kwds)
    return future


def nested_submit(fn, combos, kwds, pool=None, submitter=default_submitter):
    """Recursively submit jobs directly, as delayed objects or to a pool.

    Parameters
    ----------
    fn : callable
        Function to submit jobs to.
    combos : tuple mapping individual fn arguments to sequence of values
        Mapping of each argument and all its possible values.
    kwds : dict
        Constant keyword arguments not to iterate over.
    pool : Executor pool
        Pool-executor-like class implementing a submit method.

    Returns
    -------
    results : nested tuple
        Nested tuples of results.
    """
    arg, inputs = combos[0]

    # have reached most nested level?
    if len(combos) == 1:
        if pool:
            return tuple(submitter(pool, fn, **kwds, **{arg: x})
                         for x in inputs)
        else:
            return tuple(fn(**kwds, **{arg: x}) for x in inputs)
    # else recurse through current level
    else:
        return tuple(nested_submit(fn, combos[1:], {**kwds, arg: x}, pool=pool,
                                   submitter=submitter) for x in inputs)


def default_getter(pbar=None):
    """Generate the default function to get a result from a future, updating
    the progress bar ``pbar`` in the process.
    """
    if pbar:
        def getter(future):
            try:
                res = future.result()
            except AttributeError:
                res = future.get()
            pbar.update()
            return res
    else:
        def getter(future):
            try:
                res = future.result()
            except AttributeError:
                res = future.get()
            return res
    return getter


def nested_get(futures, ndim, getter):
    """Recusively get results from nested futures.
    """
    return (tuple(getter(fut) for fut in futures) if ndim == 1 else
            tuple(nested_get(fut, ndim - 1, getter) for fut in futures))


def _combo_runner_pool(fn, combos, constants, n, ndim, pool, verbosity=1):
    """Submit and retrieve combos from a generic pool.
    """
    with progbar(total=n, disable=verbosity <= 0) as pbar:

        if verbosity >= 2:
            pbar.set_description("Processing with pool")

        futures = nested_submit(fn, combos, constants, pool=pool)
        getter = default_getter(pbar)
        return nested_get(futures, ndim, getter)


def _combo_runner_parallel(fn, combos, constants, n, ndim,
                           num_workers, verbosity=1):
    """Submit and retrieve combos from a ProcessPoolExecutor.
    """
    with cf.ProcessPoolExecutor(max_workers=num_workers) as pool:

        with progbar(total=n, disable=verbosity <= 0) as pbar:

            if verbosity >= 2:
                pbar.set_description(
                    "Processing with {} workers".format(pool._max_workers)
                )

            futures = nested_submit(fn, combos, constants, pool=pool)
            for f in cf.as_completed(flatten(futures, ndim)):
                pbar.update()
            return nested_get(futures, ndim, default_getter())


def update_upon_eval(fn, pbar, verbosity=1):
    """Decorate `fn` such that every time it is called, `pbar` is updated
    """

    @functools.wraps(fn)
    def new_fn(**kwargs):
        if verbosity >= 2:
            pbar.set_description(str(kwargs))
        result = fn(**kwargs)
        pbar.update()
        return result

    return new_fn


def _combo_runner_sequential(fn, combos, constants, n, ndim, verbosity=1):
    """Run combos in a sequential manner.
    """
    with progbar(total=n, disable=verbosity <= 0) as pbar:

        # Wrap the function such that the progbar is updated upon each call
        fn = update_upon_eval(fn, pbar, verbosity=verbosity)
        return nested_submit(fn, combos, constants)


def _combo_runner(fn, combos, constants, split=False,
                  parallel=False, num_workers=None, pool=None,
                  verbosity=1):
    """Core combo runner, i.e. no parsing of arguments.
    """
    n = prod(len(x) for _, x in combos)
    ndim = len(combos)

    kws = {'fn': fn, 'combos': combos, 'constants': constants, 'n': n,
           'ndim': ndim, 'verbosity': verbosity}

    # Custom pool supplied
    if pool is not None:
        results = _combo_runner_pool(pool=pool, **kws)

    # Else for parallel, by default use a process pool exceutor
    elif parallel or num_workers:
        results = _combo_runner_parallel(num_workers=num_workers, **kws)

    # Evaluate combos sequentially
    else:
        results = _combo_runner_sequential(**kws)

    return tuple(unzip(results, ndim)) if split else results


def combo_runner(fn, combos, *, constants=None, split=False,
                 parallel=False, pool=None, num_workers=None,
                 verbosity=1):
    """Take a function fn and analyse it over all combinations of named
    variables' values, optionally showing progress and in parallel.

    Parameters
    ----------
    fn : callable
        Function to analyse.
    combos : mapping of individual fn arguments to sequence of values
        All combinations of each argument will be calculated. Each
        argument range thus gets a dimension in the output array(s).
    constants : dict, optional
        List of tuples/dict of *constant* fn argument mappings.
    split : bool, optional
        Whether to split (unzip) into multiple output arrays or not.
    parallel : bool, optional
        Process combos in parallel, default number of workers picked.
    pool : executor-like pool, optional
        Submit all combos to this pool. Must have ``submit`` or ``apply_async``
        methods.
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
        Nested tuple containing all combinations of running ``fn``.
    """
    # Prepare combos
    combos = _parse_combos(combos)
    constants = _parse_constants(constants)

    # Submit to core combo runner
    return _combo_runner(fn, combos, constants=constants, split=split,
                         parallel=parallel, pool=pool, num_workers=num_workers,
                         verbosity=verbosity)


def multi_concat(results, dims):
    """Concatenate a nested list of xarray objects along several dimensions.
    """
    if len(dims) == 1:
        return xr.concat(results, dim=dims[0])
    else:
        return xr.concat([multi_concat(sub_results, dims[1:])
                          for sub_results in results], dim=dims[0])


def get_ndim_first(x, ndim):
    """Return the first element from the ndim-nested list x.
    """
    return (x if ndim == 0 else
            get_ndim_first(x[0], ndim - 1))


def _combos_to_ds(results, combos, var_names, var_dims, var_coords,
                  constants=None, attrs=None):
    """Convert the output of combo_runner into a `xarray.Dataset`.
    """
    fn_args = tuple(x for x, _ in combos)
    results = _parse_combo_results(results, var_names)

    # Check if the results are an array of xarray objects
    xobj_results = isinstance(get_ndim_first(results, len(fn_args) + 1),
                              (xr.Dataset, xr.DataArray))

    if xobj_results:
        # concat them all together, no var_names needed
        ds = multi_concat(results[0], fn_args)
        # Set dataset coordinates
        for fn_arg, vals in combos:
            ds[fn_arg] = vals
    else:
        # create a new dataset using the given arrays and var_names
        ds = xr.Dataset(
            coords={
                **dict(combos),
                **dict(var_coords)
            },
            data_vars={
                name: (fn_args + var_dims[name], np.asarray(data))
                for data, name in zip(results, var_names)
            })

    if attrs:
        ds.attrs = attrs

    # Add constants to attrs, but filter out those which should be coords
    if constants:
        for k, v in constants.items():
            if k in ds.dims:
                ds.coords[k] = v
            else:
                ds.attrs[k] = v
    return ds


def combo_runner_to_ds(fn, combos, var_names, *,
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
    fn : callable
        Function to evaluate.
    combos : mapping
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
    constants : mapping, optional
        Arguments to `fn` which are not iterated over, these will be
        recorded either as attributes or coordinates if they are named
        in `var_dims`.
    resources : mapping, optional
        Like `constants` but they will not be recorded.
    attrs : mapping, optional
        Any extra attributes to store.
    combo_runner_settings
        Arguments supplied to :func:`~xyzpy.combo_runner`.

    Returns
    -------
    ds : xarray.Dataset
        Multidimensional labelled dataset contatining all the results.
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
