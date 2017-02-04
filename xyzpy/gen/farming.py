"""
"""
# TODO: tests
# TODO: harvest_cases
# TODO: logging
# TODO: file lock
# TODO: harvester getter last_ds from runner
# TODO: save_on=() : save every iteration of this parameter
# TODO: split_on=() : split save files upon each iteration of this param
# TODO: keep last n datasets
# TODO: try auto_combine before merge

import os
import shutil

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
from ..manage import load_ds, save_ds


# --------------------------------------------------------------------------- #
#                                   RUNNER                                    #
# --------------------------------------------------------------------------- #

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
        self._var_names = _parse_var_names(var_names)
        if fn_args:
            self._fn_args = _parse_fn_args(fn_args)
        self._var_dims = _parse_var_dims(var_dims, self._var_names)
        self._var_coords = _parse_var_coords(var_coords)
        self._constants = _parse_constants(constants)
        self._resources = _parse_resources(resources)
        self.attrs = attrs
        self.default_runner_settings = default_runner_settings

    # Attributes which should be parsed ------------------------------------- #

    def _get_fn_args(self):
        return self._fn_args

    def _set_fn_args(self, fn_args):
        self._fn_args = _parse_fn_args(fn_args)

    def _del_fn_args(self):
        self._fn_args = None
    fn_args = property(_get_fn_args, _set_fn_args, _del_fn_args,
                       "List of the names of the arguments that the "
                       "Runner's function takes.")

    def _get_var_names(self):
        return self._var_names

    def _set_var_names(self, var_names):
        self._var_names = _parse_var_names(var_names)

    def _del_var_names(self):
        self._var_names = None
    var_names = property(_get_var_names, _set_var_names, _del_var_names,
                         "List of the names of the variables that the "
                         "Runner's function produces.")

    def _get_var_dims(self):
        return self._var_dims

    def _set_var_dims(self, var_dims, var_names=None):
        if var_names is None:
            var_names = self._var_names
        self._var_dims = _parse_var_dims(var_dims, var_names)

    def _del_var_dims(self):
        self._var_dims = None
    var_dims = property(_get_var_dims, _set_var_dims, _del_var_dims,
                        "Mapping of each output variable to its named "
                        "dimensions")

    def _get_var_coords(self):
        return self._var_coords

    def _set_var_coords(self, var_coords):
        self._var_coords = _parse_var_coords(var_coords)

    def _del_var_coords(self):
        self._var_coords = None
    var_coords = property(_get_var_coords, _set_var_coords, _del_var_coords,
                          "Mapping of each variable named dimension to its "
                          "coordinate values.")

    def _get_constants(self):
        return self._constants

    def _set_constants(self, constants):
        self._constants = _parse_constants(constants)

    def _del_constants(self):
        self._constants = None
    constants = property(_get_constants, _set_constants, _del_constants,
                         "Mapping of constant arguments supplied to the "
                         "Runner's function.")

    def _get_resources(self):
        return self._resources

    def _set_resources(self, resources):
        self._resources = _parse_resources(resources)

    def _del_resources(self):
        self._resources = None
    resources = property(_get_resources, _set_resources, _del_resources,
                         "Mapping of constant arguments supplied to the "
                         "Runner's function that are *not* saved with the "
                         "dataset.")

    # Running methods ------------------------------------------------------- #

    def run_combos(self, combos, constants=(), **runner_settings):
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
            self.fn, combos, self._var_names,
            var_dims=self._var_dims,
            var_coords=self._var_coords,
            constants={**self._constants, **dict(constants)},
            resources=self._resources,
            attrs=self.attrs,
            parse=False,
            **{**self.default_runner_settings, **runner_settings})
        return self.last_ds

    def run_cases(self, cases, **runner_settings):
        """Run cases using the function map and save to dataset.

        Parameters
        ----------
            cases : tuple of form ((arg1_val, *), *)
                A list of cases, each
            **runner_settings :
                Keyword arguments supplied to `case_runner`
        """
        cases = _parse_cases(cases)
        self.last_ds = case_runner_to_ds(
            fn=self.fn,
            fn_args=self._fn_args,
            cases=cases,
            var_names=self._var_names,
            var_dims=self._var_dims,
            var_coords=self._var_coords,
            constants=self._constants,
            resources=self._resources,
            attrs=self.attrs,
            parse=False,
            **{**self.default_runner_settings, **runner_settings})
        return self.last_ds


# --------------------------------------------------------------------------- #
#                                 HARVESTER                                   #
# --------------------------------------------------------------------------- #

class Harvester(object):
    """Container class for collecting and aggregating data to disk.
    """

    def __init__(self, runner, data_name, engine='h5netcdf', full_ds=None):
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
        self.runner = runner
        self.data_name = data_name
        self.engine = engine
        self._full_ds = full_ds

    def load_ds(self):
        """Load the full dataset from disk.
        """
        self._full_ds = load_ds(self.data_name, engine=self.engine)
        return self._full_ds

    @property
    def full_ds(self):
        """Get the dataset containing all saved runs.
        """
        if self._full_ds is None:
            self.load_ds()
        return self._full_ds

    @property
    def last_ds(self):
        """The dataset containing last runs data.
        """
        return self.runner.last_ds

    def delete_ds(self, backup=True):
        """Delete the on-disk dataset.
        """
        if backup:
            import datetime
            ts = '{:%Y%m%d-%H%M%S}'.format(datetime.datetime.now())
            shutil.copy(self.data_name, self.data_name + '.BAK-{}'.format(ts))
        os.remove(self.data_name)

    def merge_into_full_ds(self, new_ds):
        """Merge a new dataset into the in-memory full dataset.
        """
        if self._full_ds is None:
            self._full_ds = new_ds.copy(deep=True)
        else:
            self._full_ds.merge(new_ds, compat='no_conflicts', inplace=True)

    def merge_save(self, new_ds):
        """Merge a new dataset into the full, on-disk dataset.
        """
        # Check file exists and can be written to
        if os.access(self.data_name, os.W_OK):
            # Open, merge new data and close.
            self._full_ds = load_ds(self.data_name, engine=self.engine)
            self._full_ds.merge(new_ds, compat='no_conflicts', inplace=True)
            save_ds(self._full_ds, self.data_name, engine=self.engine)
        # Check that it is not read-only
        elif os.path.isfile(self.data_name):  # pragma: no cover
            raise OSError("The file '{}' exists but cannot be written "
                          "to".format(self.data_name))
        # Else just create it new
        else:
            self._full_ds = new_ds.copy(deep=True)
            self._full_ds.to_netcdf(self.data_name, engine=self.engine)

    def harvest_combos(self, combos, save=True, **runner_settings):
        """Run combos, automatically merging into an on-disk dataset.
        """
        self.runner.run_combos(combos, **runner_settings)
        if save:
            self.merge_save(self.runner.last_ds)
        else:
            self.merge_into_full_ds(self.runner.last_ds)

    def harvest_cases(self, cases, save=True, **runner_settings):
        """Run cases, automatically merging into an on-disk dataset.
        """
        self.runner.run_cases(cases, **runner_settings)
        if save:
            self.merge_save(self.runner.last_ds)
        else:
            self.merge_into_full_ds(self.runner.last_ds)
