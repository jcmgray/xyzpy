"""Functions for systematically evaluating a function over specific cases.
"""
import functools
import itertools

from ..utils import progbar
from .prepare import (
    parse_fn_args,
    parse_cases,
    parse_combos,
    parse_constants,
    parse_resources,
    parse_var_names,
    parse_var_dims,
    parse_var_coords,
)
from .combo_runner import (
    combo_runner_core,
    combo_runner_to_ds,
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
    combos : iterable[tuple] or iterable[dict], optional
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
