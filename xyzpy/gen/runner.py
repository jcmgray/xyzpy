"""
"""
import xarray as xr

from .combo_runner import combo_runner_to_ds
from .case_runner import case_runner_to_ds


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
