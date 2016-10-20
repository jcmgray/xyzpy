from cytoolz import isiterable


def dictify(x):
    return dict(x) if x else dict()


# combo_runner -------------------------------------------------------------- #

def _parse_combos(combos):
    """Turn dicts and single tuples into proper form for combo runners.
    """
    if isinstance(combos, dict):
        combos = tuple(combos.items())
    elif isinstance(combos[0], str):
        combos = (combos,)
    return tuple((arg, list(vals)) for arg, vals in combos)


def _parse_combo_results(results, var_names):
    """
    """
    if isinstance(var_names, str) or len(var_names) == 1:
        results = (results,)
    return results


# case_runner --------------------------------------------------------------- #

def _parse_fn_args(fn_args):
    """
    """
    return (fn_args,) if isinstance(fn_args, str) else tuple(fn_args)


def _parse_cases(cases):
    """
    """
    cases = tuple(cases)
    if isinstance(cases[0], str) or not isiterable(cases[0]):
        cases = tuple((c,) for c in cases)
    return cases


def _parse_case_results(results, var_names):
    """
    """
    if isinstance(var_names, str) or len(var_names) == 1:
        results = tuple((r,) for r in results)
    return results


# common variable description ----------------------------------------------- #

def _parse_var_names(var_names):
    """
    """
    return (var_names,) if isinstance(var_names, str) else tuple(var_names)


def _parse_var_dims(var_dims, var_names):
    """Parse function mapping parameters into standard form.

    Parameters
    ----------
        var_names : tuple of str, or str
            * tuple of str
                List of names of var_names.
            * str
                Single named output.
        var_dims : dict, tuple, or str
            * dict
                Mapping of each output to its dimensions, each either str
                or tuple of str. The keys themselves can be a tuple of several
                output names if they all have the same dimensions.
            * tuple
                List of output dimensions directly corresponding to list of
                var_names. Must be same length as `var_names`
            * str
                Only allowed for single output with single dimension.
    """
    new_var_dims = {k: () for k in var_names}  # default to empty tuple

    if not var_dims:
        return new_var_dims
    # check if single output with single dim
    elif isinstance(var_dims, str):
        if len(var_names) != 1:
            raise ValueError("When `var_dims` is specified as a single "
                             "string, there must be a single output, for "
                             "which it is the single dimension.")
        var_dims = {var_names[0]: (var_dims,)}
    # check for direct correspondence to var_names
    elif (isinstance(var_dims, (tuple, list)) and
          any(isinstance(x, str) or x[0] not in var_names for x in var_dims)):
        if len(var_dims) != len(var_names):
            raise ValueError("`var_dims` cannot be interpreted as a "
                             "mapping of `var_names` to their dimensions "
                             "and is the wrong length to be in one to "
                             "one correspondence.")
        var_dims = dict(zip(var_names, var_dims))
    # assume dict-like
    else:
        var_dims = dict(var_dims)

    # update new_var_dims, splitting var_names defined as having the same
    #   dims, and making sure all dims are tuple.
    try:
        for k, v in var_dims.items():
            v = (v,) if isinstance(v, str) else tuple(v)
            if isinstance(k, str):
                assert k in var_names
                new_var_dims[k] = v
            else:
                for sub_k in k:
                    assert sub_k in var_names
                    new_var_dims[sub_k] = v
    except AssertionError:
        raise ValueError("An unexpected output name was specified in the "
                         "output dimensions mapping.")

    return new_var_dims

_parse_var_coords = dictify
_parse_constants = dictify
_parse_resources = dictify
