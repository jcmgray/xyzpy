"""
"""
# TODO: harvest_cases
# TODO: logging
# TODO: file lock
# TODO: harvester getter last_ds from runner
# TODO: save_on=() : save every iteration of this parameter
# TODO: split_on=() : split save files upon each iteration of this param
# TODO: setters with parsers
# TODO: keep last n datasets

import os
import xarray as xr

from .prepare import (
    _parse_fn_args,
    _parse_var_names,
    _parse_var_dims,
    _parse_var_coords,
    _parse_constants,
    _parse_resources,
    _parse_combos,
    _parse_cases,
)
from .combo_runner import combo_runner_to_ds
from .case_runner import case_runner_to_ds


class Runner(object):
    """Container class with all the information needed to systematically
    run a function over many parameters and capture the output in a dataset.
    """
    def __init__(self, fn, var_names,
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
            fn_args : str, or sequence of str (optional)
                The ordered name(s) of the input arguments(s) of `fn`. This is
                only needed if the cases or combos supplied are not dict-like.
            var_dims : dict-like
                Mapping of output variables to their named internal dimensions.
            var_coords : dict-like
                Mapping of output variables named internal dimensions to the
                actual values they take.
            constants : dict-like
                Constants arguments to be supplied to `fn`. These can be used
                as 'var_dims', and will be saved as coords if so, otherwise
                as attributes.
            resources : dict-like
                Like `constants` but not saved to the the dataset, e.g. if
                very big.
            attrs : dict-like
                Any other miscelleous information to be saved with the
                dataset.
            **default_runner_settings :
                These keyword arguments will be supplied as defaults to any
                runner.
        """
        self.fn = fn
        self.var_names = _parse_var_names(var_names)
        if fn_args:
            self.fn_args = _parse_fn_args(fn_args)
        self.var_dims = _parse_var_dims(var_dims, self.var_names)
        self.var_coords = _parse_var_coords(var_coords)
        self.constants = _parse_constants(constants)
        self.resources = _parse_resources(resources)
        self.attrs = attrs
        self.default_runner_settings = default_runner_settings

    def run_combos(self, combos, **runner_settings):
        """Run combos using the function map and save to dataset.

        Parameters
        ----------
            combos : tuple of form ((str, seq), *)
                The values of each function argument with which to evaluate
                all combinations.
            **runner_settings :
                Keyword arguments supplied to `combo_runner`
        """
        combos = _parse_combos(combos)
        self.last_ds = combo_runner_to_ds(
            self.fn, combos, self.var_names,
            var_dims=self.var_dims,
            var_coords=self.var_coords,
            constants=self.constants,
            resources=self.resources,
            attrs=self.attrs,
            parse=False,
            **{**self.default_runner_settings, **runner_settings})

    def run_cases(self, cases, **runner_settings):
        """Run cases using the function map and save to dataset.

        Parameters
        ----------
            cases : tuple of form ((arg1_val, *), *)
                A list of cases, each
            **runner_settings :
                Keyword arguments supplied to `case_runner`
        """
        if self.fn_args is None:
            raise ValueError("Please specify the function argument names.")
        cases = _parse_cases(cases)
        self.last_ds = case_runner_to_ds(
            fn=self.fn,
            fn_args=self.fn_args,
            cases=cases,
            var_names=self.var_names,
            var_dims=self.var_dims,
            var_coords=self.var_coords,
            constants=self.constants,
            resources=self.resources,
            attrs=self.attrs,
            parse=False,
            **{**self.default_runner_settings, **runner_settings})


class Harvester(object):
    """Container class for collecting and aggregating data to disk.
    """
    def __init__(self, runner, data_name, engine='h5netcdf'):
        """
        Parameters
        ----------
            runner : Runner instance
                Performs the runs and describes the results.
            data_name : str
                Base file path to save data to.
            engine : str (optional)
                Internal netcdf engine for xarray to use.
        """
        self.r = runner
        self.data_name = data_name
        self.engine = engine

    def merge_save(self, new_ds, create_new=True):
        """
        """
        # Check file exists and can be written to.
        if os.access(self.data_name, os.W_OK):
            # Open, merge new data and close.
            self.full_ds = xr.open_dataset(self.data_name, engine=self.engine)
            self.full_ds.load()
            self.full_ds.close()
            self.full_ds.merge(new_ds, compat='no_conflicts', inplace=True)
            self.full_ds.to_netcdf(self.data_name, engine=self.engine)
        elif create_new:
            self.full_ds = new_ds.copy(deep=True)
            self.full_ds.to_netcdf(self.data_name, engine=self.engine)
        else:
            raise OSError("The file '{}' can not be accesed and `create_new` "
                          "is set to False.".format(self.data_name))

    def harvest_combos(self, combos, save=True, **runner_settings):
        """Run combos, automatically merging into an on-disk dataset.
        """
        self.r.run_combos(combos, **runner_settings)
        if save:
            self.merge_save(self.r.last_ds)

    def harvest_cases(self, cases, save=True, **runner_settings):
        # TODO
        pass
