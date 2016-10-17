"""
"""
import xarray as xr

from .combo_runner import combo_runner_to_ds
from .case_runner import case_runner_to_ds


def parse_output_mapping(outputs, output_dims=None):
    """Parse function mapping parameters into standard form.

    Parameters
    ----------
        outputs : tuple of str, or str
            * tuple of str
                List of names of outputs.
            * str
                Single named output.
        output_dims : dict, tuple, or str
            * dict
                Mapping of each output to its dimensions, each either str
                or tuple of str. The keys themselves can be a tuple of several
                output names if they all have the same dimensions.
            * tuple
                List of output dimensions directly corresponding to list of
                outputs. Must be same length as `outputs`
            * str
                Only allowed for single output with single dimension.

    """
    outputs = (outputs,) if isinstance(outputs, str) else tuple(outputs)
    new_output_dims = {k: () for k in outputs}  # default to empty tuple

    if not output_dims:
        return outputs, new_output_dims
    # check if single output with single dim
    elif isinstance(output_dims, str):
        if len(outputs) != 1:
            raise ValueError("When `output_dims` is specified as a single "
                             "string, there must be a single output, for "
                             "which it is the single dimension.")
        output_dims = {outputs[0]: (output_dims,)}
    # check for direct correspondence to outputs
    elif (isinstance(output_dims, (tuple, list)) and
          any(isinstance(x, str) or x[0] not in outputs for x in output_dims)):
        if len(output_dims) != len(outputs):
            raise ValueError("`output_dims` cannot be interpreted as a "
                             "mapping of `outputs` to their dimensions "
                             "and is the wrong length to be in one to "
                             "one correspondence.")
        output_dims = dict(zip(outputs, output_dims))
    # assume dict-like
    else:
        output_dims = dict(output_dims)

    # update new_output_dims, splitting outputs defined as having the same
    #   dims, and making sure all dims are tuple.
    try:
        for k, v in output_dims.items():
            v = (v,) if isinstance(v, str) else tuple(v)
            if isinstance(k, str):
                assert k in outputs
                new_output_dims[k] = v
            else:
                for sub_k in k:
                    assert sub_k in outputs
                    new_output_dims[sub_k] = v
    except AssertionError:
        raise ValueError("An unexpected output name was specified in the "
                         "output dimensions mapping.")

    return outputs, new_output_dims


class Runner(object):
    """Container class containing all the information needed to systematically
    run a function over many parameters and capture the output in a dataset.
    """

    def __init__(self, fn,
                 var_names,
                 fn_args=None,
                 var_dims=None,
                 var_coords=None,
                 constants=None,
                 resources=None,
                 attrs=None,
                 **default_runner_settings):
        """
        Parameters
        ----------
            fn : callable
                Function to run.
            var_names : str, or sequence of str
                The ordered name(s) of the ouput variable(s) of `fn`.
            fn_args: str, or sequence of str (optional)
                The ordered name(s) of the input arguments(s) of `fn`. This is
                only needed if the cases or combos supplied are not dict-like.
            var_dims :

            var_coords :

            constants :

            resources :

            attrs :

            **default_runner_settings :


        """
        self.fn = fn
        self.fn_args = fn_args
        self.var_names = var_names
        self.var_dims = var_dims
        self.var_coords = var_coords
        self.constants = constants
        self.resources = resources
        self.attrs = attrs
        self.default_runner_settings = default_runner_settings

    def combo_run(self, combos, **kwargs):
        """
        """
        return combo_runner_to_ds(
            fn=self.fn,
            combos=combos,
            var_names=self.var_names,
            var_dims=self.var_dims,
            var_coords=self.var_coords,
            constants=self.constants,
            resources=self.resources,
            attrs=self.attrs,
            **{**self.default_runner_settings, **kwargs})

    def case_run(self, cases, **kwargs):
        """
        """
        if self.fn_args is None:
            raise ValueError("Please specify the function argument names.")
        return case_runner_to_ds(
            fn=self.fn,
            fn_args=self.fn_args,
            cases=cases,
            var_names=self.var_names,
            var_dims=self.var_dims,
            var_coords=self.var_coords,
            constants=self.constants,
            resources=self.resources,
            attrs=self.attrs,
            **{**self.default_runner_settings, **kwargs})


class Collector(object):
    """Container class for collecting and aggregating data to disk.
    """
    def __init__(self, data_name, engine='h5netcdf'):
        """
        """
        self.data_name = data_name
        self.engine = engine

    def merge_save(self, new_ds):
        with xr.open_dataset(self.data_name, engine=self.engine) as old_ds:
            old_ds.merge(new_ds, compat='no_conflicts', inplace=True)
            old_ds.to_netcdf(self.data_name, engine=self.engine)
