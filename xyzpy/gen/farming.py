"""Objects for labelling and succesively running functions.
"""

import os
import shutil

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
    _parse_attrs,
)
from .combo_runner import combo_runner_to_ds
from .case_runner import case_runner_to_ds
from .batch import Crop
from ..manage import load_ds, save_ds


# --------------------------------------------------------------------------- #
#                                   RUNNER                                    #
# --------------------------------------------------------------------------- #

class Runner(object):
    """Container class with all the information needed to systematically
    run a function over many parameters and capture the output in a dataset.

    Parameters
    ----------
    fn : callable
        Function that produces a single instance of a result.
    var_names : str, sequence of str, or None
        The ordered name(s) of the ouput variable(s) of `fn`. Set this
        explicitly to None if `fn` outputs already labelled data as a
        :class:`~xarray.Dataset` or :class:`~xarray.DataArray`.
    fn_args : str, or sequence of str, optional
        The ordered name(s) of the input arguments(s) of `fn`. This is only
        needed if the cases or combos supplied are not dict-like.
    var_dims : dict-like, optional
        Mapping of output variables to their named internal dimensions, can be
        the names of ``constants``.
    var_coords : dict-like, optional
        Mapping of output variables named internal dimensions to the actual
        values they take.
    constants : dict-like, optional
        Constants arguments to be supplied to `fn`. These can be used as
        'var_dims', and will be saved as coords if so, otherwise as attributes.
    resources : dict-like, optional
        Like `constants` but not saved to the the dataset, e.g. if very big.
    attrs : dict-like, optional
        Any other miscelleous information to be saved with the dataset.
    default_runner_settings
        These keyword arguments will be supplied as defaults to any runner.
    """

    def __init__(self, fn, var_names,
                 fn_args=None,
                 var_dims=None,
                 var_coords=None,
                 constants=None,
                 resources=None,
                 attrs=None,
                 **default_runner_settings):
        self.fn = fn
        self._var_names = _parse_var_names(var_names)
        self._fn_args = (_parse_fn_args(fn_args) if fn_args is not None else
                         None)
        self._var_dims = _parse_var_dims(var_dims, self._var_names)
        self._var_coords = _parse_var_coords(var_coords)
        self._constants = _parse_constants(constants)
        self._resources = _parse_resources(resources)
        self._attrs = _parse_attrs(attrs)
        self.default_runner_settings = default_runner_settings

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

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
        combos : tuple of form ((str, seq), ...)
            The values of each function argument with which to evaluate all
            combinations.
        constants : dict (optional)
            Extra constant arguments for this run, repeated arguments will
            take precedence over stored constants but for this run only.
        runner_settings
            Keyword arguments supplied to :func:`~xyzpy.combo_runner`.
        """
        combos = _parse_combos(combos)
        self.last_ds = combo_runner_to_ds(
            self.fn, combos, self._var_names,
            var_dims=self._var_dims,
            var_coords=self._var_coords,
            constants={**self._constants, **dict(constants)},
            resources=self._resources,
            attrs=self._attrs,
            parse=False,
            **{**self.default_runner_settings, **runner_settings})
        return self.last_ds

    def run_cases(self, cases, constants=(), **runner_settings):
        """Run cases using the function and save to dataset.

        Parameters
        ----------
        cases : sequence of mappings or tuples
            A sequence of cases.
        constants : dict (optional)
            Extra constant arguments for this run, repeated arguments will
            take precedence over stored constants but for this run only.
        runner_settings
            Supplied to :func:`~xyzpy.case_runner`.
        """
        cases = _parse_cases(cases)
        self.last_ds = case_runner_to_ds(
            fn=self.fn,
            fn_args=self._fn_args,
            cases=cases,
            var_names=self._var_names,
            var_dims=self._var_dims,
            var_coords=self._var_coords,
            constants={**self._constants, **dict(constants)},
            resources=self._resources,
            attrs=self._attrs,
            parse=False,
            **{**self.default_runner_settings, **runner_settings})
        return self.last_ds

    def Crop(self,
             name=None,
             parent_dir=None,
             save_fn=None,
             batchsize=None,
             num_batches=None):
        """Return a Crop instance with this runner, from which ``fn``
        will be set, and then combos can be sown, grown, and reaped into the
        ``Runner.last_ds``. See :class:`~xyzpy.Crop`.

        Returns
        -------
        Crop
        """
        return Crop(runner=self,
                    name=name,
                    parent_dir=parent_dir,
                    save_fn=save_fn,
                    batchsize=batchsize,
                    num_batches=num_batches)

    def __repr__(self):
        string = "<xyzpy.Runner>\n"
        string += "    fn: {self.fn}\n"

        if self.fn_args is not None:
            string += "    fn_args: {self.fn_args}\n"

        string += "    var_names: {self.var_names}\n"

        if self.var_dims is not None:
            string += "    var_dims: {self.var_dims}\n"

        if self.constants:
            self._constants_list = list(self.constants)
            string += "    constants: {self._constants_list}\n"

        if self.resources:
            self._resources_list = list(self.resources)
            string += "    resources: {self._resources_list}\n"

        return string.format(self=self)


# --------------------------------------------------------------------------- #
#                                 HARVESTER                                   #
# --------------------------------------------------------------------------- #

class Harvester(object):
    """Container class for collecting and aggregating data to disk.

    Parameters
    ----------
    runner : Runner
        Performs the runs and describes the results.
    data_name : str, optional
        Base file path to save data to.
    chunks : int or dict, optional
        If not None, passed to xarray so that the full dataset is loaded and
        merged into with on-disk dask arrays.
    engine : str, optional
        Engine to use to save and load datasets.
    full_ds : xarray.Dataset, optional
        Initialize the Harvester with this dataset as the intitial full
        dataset.

    Members
    -------
    full_ds : xarray.Dataset
        Dataset containing all data harvested so far, by default synced to
        disk.
    last_ds : xarray.Dataset
        Dataset containing just the data from the last harvesting run.
    """

    def __init__(self, runner, data_name=None, chunks=None,
                 engine='h5netcdf', full_ds=None):
        self.runner = runner
        self.data_name = data_name
        self.engine = engine
        self.chunks = chunks
        self._full_ds = full_ds

    @property
    def last_ds(self):
        """Dataset containing the last runs' data.
        """
        return self.runner.last_ds

    def load_full_ds(self, chunks=None, engine=None):
        """Load the disk dataset into ``full_ds``.

        Parameters
        ----------
        chunks : int or dict, optional
            If not None, passed to xarray so that the full dataset is loaded
            and merged into with on-disk dask arrays.
        engine : str, optional
            Engine to use to save and load datasets.
        """
        if engine is None:
            engine = self.engine

        if chunks is None:
            chunks = self.chunks

        # Check file exists and can be written to
        if os.access(self.data_name, os.W_OK):
            self._full_ds = load_ds(self.data_name,
                                    engine=engine,
                                    chunks=chunks)

        # Do nothing if file does not exist at all
        elif not os.path.isfile(self.data_name):  # pragma: no cover
            pass

        # Catch read-only errors etc.
        else:
            raise OSError("The file '{}' exists but cannot be written "
                          "to".format(self.data_name))

    @property
    def full_ds(self):
        """Dataset containing all saved runs.
        """
        if self._full_ds is None:
            self.load_full_ds()
        return self._full_ds

    def save_full_ds(self, new_full_ds=None, engine=None):
        """Save `full_ds` onto disk.

        Parameters
        ----------
        new_full_ds : xarray.Dataset, optional
            Save this dataset as the new full dataset, else use the current
            full datset.
        engine : str, optional
            Engine to use to save and load datasets.
        """
        if engine is None:
            engine = self.engine

        if new_full_ds is not None:
            if os.path.exists(self.data_name):
                os.remove(self.data_name)
            self._full_ds = new_full_ds

        save_ds(self._full_ds, self.data_name, engine=engine)

    def delete_ds(self, backup=False):
        """Delete the on-disk dataset, optionally backing it up first.
        """
        if backup:
            import datetime
            ts = '{:%Y%m%d-%H%M%S}'.format(datetime.datetime.now())
            shutil.copy(self.data_name, self.data_name + '.BAK-{}'.format(ts))

        os.remove(self.data_name)

    def add_ds(self, new_ds,
               sync=True,
               overwrite=None,
               chunks=None,
               engine=None):
        """Merge a new dataset into the in-memory full dataset.

        Parameters
        ----------
        new_ds : xr.Dataset or xr.DataArray
            Data to be merged into the full dataset.
        sync : bool, optional
            If True (default), load and save the disk dataset before
            and after merging in the new data.
        overwrite : {None, False, True}, optional
            How to combine data from the new run into the current full_ds:

            - ``None`` (default): attempt the merge and only raise if
              data conflicts.
            - ``True``: overwrite conflicting current data with
              that from the new dataset.
            - ``False``: drop any conflicting data from the new dataset.

        chunks : int or dict, optional
            If not None, passed to xarray so that the full dataset is loaded
            and merged into with on-disk dask arrays.
        engine : str, optional
            Engine to use to save and load datasets.
        """
        if isinstance(new_ds, xr.DataArray):
            new_ds = new_ds.to_dataset()

        # get default chunks if not overidden
        if chunks is None:
            chunks = self.chunks
        if chunks is not None:
            new_ds = new_ds.chunk(chunks)

        # only sync with disk if data name present
        sync_with_disk = sync and self.data_name is not None
        if sync_with_disk:
            self.load_full_ds(chunks=chunks, engine=engine)

        if self._full_ds is None:
            # No full ds yet, deep copy to maintain distinction between
            #   'full_ds' and 'last_ds'.
            new_full_ds = new_ds.copy(deep=True)

        else:
            # Overwrite with new data
            if overwrite is True:
                new_full_ds = new_ds.combine_first(self._full_ds)
            # Overwrite nothing
            elif overwrite is False:
                new_full_ds = self._full_ds.combine_first(new_ds)
            # Merge, raising error if the two datasets conflict
            else:
                new_full_ds = self._full_ds.merge(
                    new_ds, compat='no_conflicts', inplace=False)

        if sync_with_disk:
            self.save_full_ds(new_full_ds, engine=engine)
        else:
            self._full_ds = new_full_ds

    def harvest_combos(self, combos, *,
                       sync=True,
                       overwrite=None,
                       chunks=None,
                       engine=None,
                       **runner_settings):
        """Run combos, automatically merging into an on-disk dataset.

        Parameters
        ---------
        combos : mapping_like
            The combos to run.
        sync : bool, optional
            If True (default), load and save the disk dataset before
            and after merging in the new data.
        overwrite : {None, False, True}, optional

            - ``None`` (default): attempt the merge and only raise if
              data conflicts.
            - ``True``: overwrite conflicting current data with
              that from the new dataset.
            - ``False``: drop any conflicting data from the new dataset.

        chunks : bool, optional
            If not None, passed passed to xarray so that the full dataset is
            loaded and merged into with on-disk dask arrays.
        engine : str, optional
            Engine to use to save and load datasets.
        runner_settings
            Supplied to :func:`~xyzpy.combo_runner`.
        """

        ds = self.runner.run_combos(combos, **runner_settings)
        self.add_ds(ds, sync=sync, overwrite=overwrite,
                    chunks=chunks, engine=engine)

    def harvest_cases(self, cases, *,
                      sync=True,
                      overwrite=None,
                      chunks=None,
                      engine=None,
                      **runner_settings):
        """Run cases, automatically merging into an on-disk dataset.

        Parameters
        ---------
        cases : list of dict or tuple
            The cases to run.
        sync : bool, optional
            If True (default), load and save the disk dataset before
            and after merging in the new data.
        overwrite : {None, False, True}, optional
            What to do regarding clashes with old data:

            - ``None`` (default): attempt the merge and only raise if
              data conflicts.
            - ``True``: overwrite conflicting current data with
              that from the new dataset.
            - ``False``: drop any conflicting data from the new dataset.

        chunks : bool, optional
            If not None, passed passed to xarray so that the full dataset is
            loaded and merged into with on-disk dask arrays.
        engine : str, optional
            Engine to use to save and load datasets.
        runner_settings
            Supplied to :func:`~xyzpy.case_runner`.
        """
        ds = self.runner.run_cases(cases, **runner_settings)
        self.add_ds(ds, sync=sync, overwrite=overwrite, chunks=chunks,
                    engine=engine)

    def Crop(self,
             name=None,
             parent_dir=None,
             save_fn=None,
             batchsize=None,
             num_batches=None):
        """Return a Crop instance with this Harvester, from which `fn`
        will be set, and then combos can be sown, grown, and reaped into the
        ``Harvester.full_ds``. See :class:`~xyzpy.Crop`.

        Returns
        -------
        Crop
        """
        return Crop(harvester=self,
                    name=name,
                    parent_dir=parent_dir,
                    save_fn=save_fn,
                    batchsize=batchsize,
                    num_batches=num_batches)

    def __repr__(self):
        string = ("<xyzpy.Harvester>\n"
                  "Runner: {self.runner}"
                  "Sync file -->\n"
                  "    {self.data_name}    [{self.engine}]")

        return string.format(self=self)
