# TODO: function_str i.e. (a, b, c) -> (x[t], y[t,w], z)


from cytoolz import isiterable


def _str_2_tuple(x):
    """Ensure `x` is at least a 1-tuple of str.
    """
    return (x,) if isinstance(x, str) else tuple(x)


def dictify(x):
    """Ensure `x` is a dict.
    """
    if isinstance(x, dict):
        return x
    elif x:
        return dict(x)
    else:
        return dict()


def _parse_fn_args(fn_args):
    if fn_args is None:
        return None

    return _str_2_tuple(fn_args)


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
    if var_names is not None and (isinstance(var_names, str) or
                                  len(var_names) == 1):
        results = (results,)
    return results


# case_runner --------------------------------------------------------------- #

def _parse_cases(cases):
    """
    """

    # cases = {'a': 1, 'b': 2, 'c': 3} --> ({'a': 1, 'b': 2, 'c': 3},)
    if isinstance(cases, dict):
        return (cases,)

    cases = tuple(cases)
    # e.g. if fn_args = ('a',) and cases = (1, 10, 100)
    #     we want cases --> ((1,), (10,), (100,))
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
    return ((None,) if var_names is None else
            (var_names,) if isinstance(var_names, str) else
            tuple(var_names))


def _parse_var_dims(var_dims, var_names):
    """Parse function mapping parameters into standard form.

    Parameters
    ----------
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
    var_names : tuple of str, str, or None
        * tuple of str
            List of names of var_names.
        * str
            Single named output.
        * None
            Automatic result output using Dataset/DataArray, in this case
            check that var_dims is None as well.
    """
    if var_names is None:
        if var_dims is not None:
            raise ValueError("Cannot specify variable dimensions if using"
                             "automatic dataset output (var_names=None).")
        return dict()

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
                if k not in var_names:
                    raise KeyError

                new_var_dims[k] = v

            else:
                for sub_k in k:
                    if sub_k not in var_names:
                        raise KeyError

                    new_var_dims[sub_k] = v

    except KeyError:
        raise ValueError("An unexpected output name was specified in the "
                         "output dimensions mapping.")

    return new_var_dims


_parse_var_coords = dictify
_parse_constants = dictify
_parse_resources = dictify
_parse_attrs = dictify
