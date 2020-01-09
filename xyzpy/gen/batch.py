import os
import shutil
from itertools import chain
from time import sleep
from glob import glob
import warnings
import pickle
import copy
import math

import joblib
from joblib.externals import cloudpickle
import numpy as np
import xarray as xr

from ..utils import _get_fn_name, prod, progbar
from .combo_runner import (
    _combo_runner,
    combo_runner_to_ds,
)
from .case_runner import (
    _case_runner,
    case_runner_to_ds,
)
from .prepare import (
    _parse_combos,
    _parse_constants,
    _parse_attrs,
    _parse_cases,
)
from .farming import Runner, Harvester, Sampler


BTCH_NM = "xyz-batch-{}.jbdmp"
RSLT_NM = "xyz-result-{}.jbdmp"
FNCT_NM = "xyz-function.clpkl"
INFO_NM = "xyz-settings.jbdmp"


class XYZError(Exception):
    pass


# --------------------------------- parsing --------------------------------- #

def parse_crop_details(fn, crop_name, crop_parent):
    """Work out how to structure the sowed data.

    Parameters
    ----------
    fn : callable, optional
        Function to infer name crop_name from, if not given.
    crop_name : str, optional
        Specific name to give this set of runs.
    crop_parent : str, optional
        Specific directory to put the ".xyz-{crop_name}/" folder in
        with all the cases and results.

    Returns
    -------
    crop_location : str
        Full path to the crop-folder.
    crop_name : str
        Name of the crop.
    crop_parent : str
        Parent folder of the crop.
    """
    if crop_name is None:
        if fn is None:
            raise ValueError("Either `fn` or `crop_name` must be give.")
        crop_name = _get_fn_name(fn)

    crop_parent = crop_parent if crop_parent is not None else os.getcwd()
    crop_location = os.path.join(crop_parent, ".xyz-{}".format(crop_name))

    return crop_location, crop_name, crop_parent


def _parse_fn_farmer(fn, farmer):
    if farmer is not None:
        if fn is not None:
            warnings.warn("'fn' is ignored if a 'Runner', 'Harvester', or "
                          "'Sampler' is supplied as the 'farmer' kwarg.")
        fn = farmer.fn

    return fn, farmer


def infer_shape(x):
    """Take a nested sequence and find its shape as if it were an array.

    Examples
    --------

        >>> x = [[10, 20, 30], [40, 50, 60]]
        >>> infer_shape(x)
        (2, 3)
    """
    shape = ()
    try:
        shape += (len(x),)
        return shape + infer_shape(x[0])
    except TypeError:
        return shape


def nan_like_result(res):
    """Take a single result of a function evaluation and calculate the same
    sequence of scalars or arrays but filled entirely with ``nan``.

    Examples
    --------

        >>> res = (True, [[10, 20, 30], [40, 50, 60]], -42.0)
        >>> nan_like_result(res)
        (array(nan), array([[nan, nan, nan],
                            [nan, nan, nan]]), array(nan))

    """
    if isinstance(res, (xr.Dataset, xr.DataArray)):
        return xr.full_like(res, np.nan, dtype=float)

    try:
        return tuple(np.broadcast_to(np.nan, infer_shape(x)) for x in res)
    except TypeError:
        return np.nan


def calc_clean_up_default_res(crop, clean_up, allow_incomplete):
    """Logic for choosing whether to automatically clean up a crop, and what,
    if any, the default all-nan result should be.
    """
    if clean_up is None:
        clean_up = not allow_incomplete

    if allow_incomplete:
        default_result = crop.all_nan_result
    else:
        default_result = None

    return clean_up, default_result


def check_ready_to_reap(crop, allow_incomplete, wait):
    if not (allow_incomplete or wait or crop.is_ready_to_reap()):
        raise XYZError("This crop is not ready to reap yet - results are "
                       "missing. You can reap only finished batches by setting"
                       " ``allow_incomplete=True``, but be aware this will "
                       "represent all missing batches with ``np.nan`` and thus"
                       " might effect data-types.")


class Crop(object):
    """Encapsulates all the details describing a single 'crop', that is,
    its location, name, and batch size/number. Also allows tracking of
    crop's progress, and experimentally, automatic submission of
    workers to grid engine to complete un-grown cases. Can also be instantiated
    directly from a :class:`~xyzpy.Runner` or :class:`~xyzpy.Harvester` or
    :class:`~Sampler.Crop` instance.

    Parameters
    ----------
    fn : callable, optional
        Target function - Crop `name` will be inferred from this if
        not given explicitly. If given, `Sower` will also default
        to saving a version of `fn` to disk for `batch.grow` to use.
    name : str, optional
        Custom name for this set of runs - must be given if `fn`
        is not.
    parent_dir : str, optional
        If given, alternative directory to put the ".xyz-{name}/"
        folder in with all the cases and results.
    save_fn : bool, optional
        Whether to save the function to disk for `batch.grow` to use.
        Will default to True if `fn` is given.
    batchsize : int, optional
        How many cases to group into a single batch per worker.
        By default, batchsize=1. Cannot be specified if `num_batches`
        is.
    num_batches : int, optional
        How many total batches to aim for, cannot be specified if
        `batchsize` is.
    farmer : {xyzpy.Runner, xyzpy.Harvester, xyzpy.Sampler}, optional
        A Runner, Harvester or Sampler, instance, from which the `fn` can be
        inferred and which can also allow the Crop to reap itself straight to a
        dataset or dataframe.
    autoload : bool, optional
        If True, check for the existence of a Crop written to disk
        with the same location, and if found, load it.

    See Also
    --------
    Runner.Crop, Harvester.Crop, Sampler.Crop
    """

    def __init__(self, *,
                 fn=None,
                 name=None,
                 parent_dir=None,
                 save_fn=None,
                 batchsize=None,
                 num_batches=None,
                 farmer=None,
                 autoload=True):

        self._fn, self.farmer = _parse_fn_farmer(fn, farmer)

        self.name = name
        self.parent_dir = parent_dir
        self.save_fn = save_fn
        self.batchsize = batchsize
        self.num_batches = num_batches
        self._batch_remainder = None
        self._all_nan_result = None

        # Work out the full directory for the crop
        self.location, self.name, self.parent_dir = \
            parse_crop_details(self._fn, self.name, self.parent_dir)

        # try loading crop information if it exists
        if autoload and self.is_prepared():
            self._sync_info_from_disk()

        # Save function so it can be automatically loaded with all deps?
        if (fn is None) and (save_fn is True):
            raise ValueError("Must specify a function for it to be saved!")
        self.save_fn = save_fn is not False

    @property
    def runner(self):
        if isinstance(self.farmer, Runner):
            return self.farmer
        elif isinstance(self.farmer, (Harvester, Sampler)):
            return self.farmer.runner
        else:
            return None

    # ------------------------------- methods ------------------------------- #

    def choose_batch_settings(self, *, combos=None, cases=None):
        """Work out how to divide all cases into batches, i.e. ensure
        that ``batchsize * num_batches >= num_cases``.
        """
        if int(combos is not None) + int(cases is not None) != 1:
            raise ValueError("Can only supply one of 'combos' or 'cases'.")

        if combos is not None:
            n = prod(len(x) for _, x in combos)
        else:
            n = len(cases)

        if (self.batchsize is not None) and (self.num_batches is not None):
            # Check that they are set correctly
            pos_tot = self.batchsize * self.num_batches
            if not (n <= pos_tot < n + self.batchsize):
                raise ValueError("`batchsize` and `num_batches` cannot both"
                                 "be specified if they do not not multiply"
                                 "to the correct number of total cases.")

        # Decide based on batchsize
        elif self.num_batches is None:
            if self.batchsize is None:
                self.batchsize = 1

            if not isinstance(self.batchsize, int):
                raise TypeError("`batchsize` must be an integer.")
            if self.batchsize < 1:
                raise ValueError("`batchsize` must be >= 1.")

            self.num_batches = math.ceil(n / self.batchsize)
            self._batch_remainder = 0

        # Decide based on num_batches:
        else:
            if not isinstance(self.num_batches, int):
                raise TypeError("`num_batches` must be an integer.")
            if self.num_batches < 1:
                raise ValueError("`num_batches` must be >= 1.")

            self.batchsize, self._batch_remainder = divmod(n, self.num_batches)

    def ensure_dirs_exists(self):
        """Make sure the directory structure for this crop exists.
        """
        os.makedirs(os.path.join(self.location, "batches"), exist_ok=True)
        os.makedirs(os.path.join(self.location, "results"), exist_ok=True)

    def save_info(self, combos=None, cases=None, fn_args=None):
        """Save information about the sowed cases.
        """
        # If saving Harvester or Runner, strip out function information so
        #   as just to use pickle.
        if self.farmer is not None:
            farmer_copy = copy.deepcopy(self.farmer)
            farmer_copy.fn = None
            farmer_pkl = cloudpickle.dumps(farmer_copy)
        else:
            farmer_pkl = None

        joblib.dump({
            'combos': combos,
            'cases': cases,
            'fn_args': fn_args,
            'batchsize': self.batchsize,
            'num_batches': self.num_batches,
            '_batch_remainder': self._batch_remainder,
            'farmer': farmer_pkl,
        }, os.path.join(self.location, INFO_NM))

    def load_info(self):
        """Load the full settings from disk.
        """
        sfile = os.path.join(self.location, INFO_NM)

        if not os.path.isfile(sfile):
            raise XYZError("Settings can't be found at {}.".format(sfile))
        else:
            return joblib.load(sfile)

    def _sync_info_from_disk(self, only_missing=True):
        """Load information about the saved cases.
        """
        settings = self.load_info()
        self.batchsize = settings['batchsize']
        self.num_batches = settings['num_batches']
        self._batch_remainder = settings['_batch_remainder']

        farmer_pkl = settings['farmer']
        farmer = None if farmer_pkl is None else pickle.loads(farmer_pkl)

        fn, farmer = _parse_fn_farmer(None, farmer)

        # if crop already has a harvester/runner. (e.g. was instantiated from
        # one) by default don't overwrite from disk
        if (self.farmer) is None or (not only_missing):
            self.farmer = farmer

        if self.fn is None:
            self.load_function()

    def save_function_to_disk(self):
        """Save the base function to disk using cloudpickle
        """
        joblib.dump(cloudpickle.dumps(self._fn),
                    os.path.join(self.location, FNCT_NM))

    def load_function(self):
        """Load the saved function from disk, and try to re-insert it back into
        Harvester or Runner if present.
        """
        self._fn = cloudpickle.loads(joblib.load(
            os.path.join(self.location, FNCT_NM)))

        if self.farmer is not None:
            if self.farmer.fn is None:
                self.farmer.fn = self._fn
            else:
                # TODO: check equality?
                raise XYZError("Trying to load this Crop's function, {}, from "
                               "disk but its farmer already has a function "
                               "set: {}.".format(self._fn, self.farmer.fn))

    def prepare(self, combos=None, cases=None, fn_args=None):
        """Write information about this crop and the supplied combos to disk.
        Typically done at start of sow, not when Crop instantiated.
        """
        self.ensure_dirs_exists()
        if self.save_fn:
            self.save_function_to_disk()
        self.save_info(combos=combos, cases=cases, fn_args=fn_args)

    def is_prepared(self):
        """Check whether this crop has been written to disk.
        """
        return os.path.exists(os.path.join(self.location, INFO_NM))

    def calc_progress(self):
        """Calculate how much progressed has been made in growing the cases.
        """
        if self.is_prepared():
            self._sync_info_from_disk()
            self._num_sown_batches = len(glob(
                os.path.join(self.location, "batches", BTCH_NM.format("*"))))
            self._num_results = len(glob(
                os.path.join(self.location, "results", RSLT_NM.format("*"))))
        else:
            self._num_sown_batches = -1
            self._num_results = -1

    def is_ready_to_reap(self):
        self.calc_progress()
        return (
            self._num_results > 0 and
            (self._num_results == self.num_sown_batches)
        )

    def missing_results(self):
        """
        """
        self.calc_progress()

        def no_result_exists(x):
            return not os.path.isfile(
                os.path.join(self.location, "results", RSLT_NM.format(x)))

        return tuple(filter(no_result_exists, range(1, self.num_batches + 1)))

    def delete_all(self):
        # delete everything
        shutil.rmtree(self.location)

    @property
    def all_nan_result(self):
        if self._all_nan_result is None:
            result_files = glob(
                os.path.join(self.location, "results", RSLT_NM.format("*"))
            )
            if not result_files:
                raise XYZError("To infer an all-nan result requires at least "
                               "one finished result.")
            reference_result = joblib.load(result_files[0])[0]
            self._all_nan_result = nan_like_result(reference_result)

        return self._all_nan_result

    def __str__(self):
        # Location and name, underlined
        if not os.path.exists(self.location):
            return self.location + "\n * Not yet sown, or already reaped * \n"

        loc_len = len(self.location)
        name_len = len(self.name)

        self.calc_progress()
        percentage = 100 * self._num_results / self.num_batches

        # Progress bar
        total_bars = 20
        bars = int(percentage * total_bars / 100)

        return ("\n"
                "{location}\n"
                "{under_crop_dir}{under_crop_name}\n"
                "{num_results} / {total} batches of size {bsz} completed\n"
                "[{done_bars}{not_done_spaces}] : {percentage:.1f}%"
                "\n").format(
            location=self.location,
            under_crop_dir="-" * (loc_len - name_len),
            under_crop_name="=" * name_len,
            num_results=self._num_results,
            total=self.num_batches,
            bsz=self.batchsize,
            done_bars="#" * bars,
            not_done_spaces=" " * (total_bars - bars),
            percentage=percentage,
        )

    def __repr__(self):
        if not os.path.exists(self.location):
            progress = "*reaped or unsown*"
        else:
            self.calc_progress()
            progress = "{}/{}".format(self._num_results, self.num_batches)

        msg = "<Crop(name='{}', progress={}, batchsize={})>"
        return msg.format(self.name, progress, self.batchsize)

    def parse_constants(self, constants=None):
        constants = _parse_constants(constants)

        if self.runner is not None:
            constants = {**self.runner._constants, **constants}
            constants = {**self.runner._resources, **constants}

        return constants

    def sow_combos(self, combos, constants=None, verbosity=1):
        """Sow to disk.
        """
        combos = _parse_combos(combos)
        constants = self.parse_constants(constants)

        # Sort to ensure order remains same for reaping results
        #   (don't want to hash kwargs)
        combos = sorted(combos, key=lambda x: x[0])

        self.choose_batch_settings(combos=combos)
        self.prepare(combos=combos)

        with Sower(self) as sow_fn:
            _combo_runner(fn=sow_fn, combos=combos, constants=constants,
                          verbosity=verbosity)

    def sow_cases(self, fn_args, cases, constants=None, verbosity=1):
        cases = _parse_cases(cases)

        constants = self.parse_constants(constants)

        self.choose_batch_settings(cases=cases)
        self.prepare(fn_args=fn_args, cases=cases)

        with Sower(self) as sow_fn:
            _case_runner(fn=sow_fn, fn_args=fn_args, cases=cases,
                         constants=constants, verbosity=verbosity)

    def sow_samples(self, n, combos=None, constants=None, verbosity=1):
        fn_args, cases = self.farmer.gen_cases_fnargs(n, combos)
        self.sow_cases(fn_args, cases,
                       constants=constants, verbosity=verbosity)

    def grow(self, batch_ids, **combo_runner_opts):
        """Grow specific batch numbers using this process.
        """
        if isinstance(batch_ids, int):
            batch_ids = (batch_ids,)

        _combo_runner(grow, combos=(('batch_number', batch_ids),),
                      constants={'verbosity': 0, 'crop': self},
                      **combo_runner_opts)

    def grow_missing(self, **combo_runner_opts):
        """Grow any missing results using this process.
        """
        self.grow(batch_ids=self.missing_results(), **combo_runner_opts)

    def reap_combos(self, wait=False, clean_up=None, allow_incomplete=False):
        """Reap already sown and grown results from this crop.

        Parameters
        ----------
        wait : bool, optional
            Whether to wait for results to appear. If false (default) all
            results need to be in place before the reap.
        clean_up : bool, optional
            Whether to delete all the batch files once the results have been
            gathered. If left as ``None`` this will be automatically set to
            ``not allow_incomplete``.
        allow_incomplete : bool, optional
            Allow only partially completed crop results to be reaped,
            incomplete results will all be filled-in as nan.

        Returns
        -------
        results : nested tuple
            'N-dimensional' tuple containing the results.
        """
        check_ready_to_reap(self, allow_incomplete, wait)

        clean_up, default_result = calc_clean_up_default_res(
            self, clean_up, allow_incomplete
        )

        # load same combinations as cases saved with
        settings = joblib.load(os.path.join(self.location, INFO_NM))

        with Reaper(self, num_batches=settings['num_batches'],
                    wait=wait, default_result=default_result) as reap_fn:

            results = _combo_runner(fn=reap_fn, constants={},
                                    combos=settings['combos'])

        if clean_up:
            self.delete_all()

        return results

    def reap_combos_to_ds(self,
                          var_names=None,
                          var_dims=None,
                          var_coords=None,
                          constants=None,
                          attrs=None,
                          parse=True,
                          wait=False,
                          clean_up=None,
                          allow_incomplete=False):
        """Reap a function over sowed combinations and output to a Dataset.

        Parameters
        ----------
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
        wait : bool, optional
            Whether to wait for results to appear. If false (default) all
            results need to be in place before the reap.
        clean_up : bool, optional
            Whether to delete all the batch files once the results have been
            gathered. If left as ``None`` this will be automatically set to
            ``not allow_incomplete``.
        allow_incomplete : bool, optional
            Allow only partially completed crop results to be reaped,
            incomplete results will all be filled-in as nan.

        Returns
        -------
        xarray.Dataset or pandas.Dataframe
            Multidimensional labelled dataset contatining all the results.
        """
        check_ready_to_reap(self, allow_incomplete, wait)

        clean_up, default_result = calc_clean_up_default_res(
            self, clean_up, allow_incomplete
        )

        # Load exact same combinations as cases saved with
        settings = joblib.load(os.path.join(self.location, INFO_NM))

        if parse:
            constants = _parse_constants(constants)
            attrs = _parse_attrs(attrs)

        with Reaper(self, num_batches=settings['num_batches'],
                    wait=wait, default_result=default_result) as reap_fn:

            # Move constants into attrs, so as not to pass them to the Reaper
            #   when if fact they were meant for the original function.
            opts = dict(
                fn=reap_fn,
                var_names=var_names,
                var_dims=var_dims,
                var_coords=var_coords,
                constants={},
                resources={},
                parse=parse
            )

            if settings['combos'] is not None:
                opts['combos'] = settings['combos']
                opts['attrs'] = {**attrs, **constants}
                data = combo_runner_to_ds(**opts)
            else:
                opts['fn_args'] = settings['fn_args']
                opts['cases'] = settings['cases']
                opts['to_df'] = True
                data = case_runner_to_ds(**opts)

        if clean_up:
            self.delete_all()

        return data

    def reap_runner(self, runner, wait=False,
                    clean_up=None, allow_incomplete=False):
        """Reap a Crop over sowed combos and save to a dataset defined by a
        Runner.
        """
        # Can ignore `Runner.resources` as they play no part in desecribing the
        #   output, though they should be supplied to sow and thus grow.
        ds = self.reap_combos_to_ds(
            var_names=runner._var_names,
            var_dims=runner._var_dims,
            var_coords=runner._var_coords,
            constants=runner._constants,
            attrs=runner._attrs,
            parse=False,
            wait=wait,
            clean_up=clean_up,
            allow_incomplete=allow_incomplete)
        runner.last_ds = ds
        return ds

    def reap_harvest(self, harvester, wait=False, sync=True, overwrite=None,
                     clean_up=None, allow_incomplete=False):
        """Reap a Crop over sowed combos and merge with the dataset defined by
        a Harvester.
        """
        if harvester is None:
            raise ValueError("Cannot reap and harvest if no Harvester is set.")

        ds = self.reap_runner(harvester.runner, wait=wait, clean_up=clean_up,
                              allow_incomplete=allow_incomplete)

        if sync:
            harvester.add_ds(ds, sync=sync, overwrite=overwrite)

        return ds

    def reap_samples(self, sampler, wait=False, sync=True,
                     clean_up=None, allow_incomplete=False):
        if sampler is None:
            raise ValueError("Cannot reap samples without a 'Sampler'.")

        df = self.reap_runner(sampler.runner, wait=wait, clean_up=clean_up,
                              allow_incomplete=allow_incomplete)

        if sync:
            sampler._last_df = df
            sampler.add_df(df, sync=sync)

        return df

    def reap(self, wait=False, sync=True, overwrite=None,
             clean_up=None, allow_incomplete=False):
        """Reap sown and grown combos from disk. Return a dataset if a runner
        or harvester is set, otherwise, the raw nested tuple.

        Parameters
        ----------
        wait : bool, optional
            Whether to wait for results to appear. If false (default) all
            results need to be in place before the reap.
        sync : bool, optional
            Immediately sync the new dataset with the on-disk full dataset or
            dataframe if a harvester or sampler is used.
        overwrite : bool, optional
            How to compare data when syncing to on-disk dataset.
            If ``None``, (default) merge as long as no conflicts.
            ``True``: overwrite with the new data. ``False``, discard any
            new conflicting data.
        clean_up : bool, optional
            Whether to delete all the batch files once the results have been
            gathered. If left as ``None`` this will be automatically set to
            ``not allow_incomplete``.
        allow_incomplete : bool, optional
            Allow only partially completed crop results to be reaped,
            incomplete results will all be filled-in as nan.

        Returns
        -------
        nested tuple or xarray.Dataset
        """
        opts = dict(clean_up=clean_up, wait=wait,
                    allow_incomplete=allow_incomplete)

        if isinstance(self.farmer, Runner):
            return self.reap_runner(self.farmer, **opts)

        if isinstance(self.farmer, Harvester):
            opts['overwrite'] = overwrite
            return self.reap_harvest(self.farmer, **opts)

        if isinstance(self.farmer, Sampler):
            return self.reap_samples(self.farmer, **opts)

        return self.reap_combos(**opts)

    def check_bad(self, delete_bad=True):
        """Check that the result dumps are not bad -> sometimes length does not
        match the batch. Optionally delete these so that they can be re-grown.

        Parameters
        ----------
        delete_bad : bool
            Delete bad results as they are come across.

        Returns
        -------
        bad_ids : tuple
            The bad batch numbers.
        """
        # XXX: work out why this is needed sometimes on network filesystems.
        result_files = glob(
            os.path.join(self.location, "results", RSLT_NM.format("*")))

        bad_ids = []

        for result_file in result_files:
            # load corresponding batch file to check length.
            result_num = os.path.split(
                result_file)[-1].strip("xyz-result-").strip(".jbdmp")
            batch_file = os.path.join(
                self.location, "batches", BTCH_NM.format(result_num))

            batch = joblib.load(batch_file)

            try:
                result = joblib.load(result_file)
                unloadable = False
            except Exception as e:
                unloadable = True
                err = e

            if unloadable or (len(result) != len(batch)):
                msg = "result {} is bad".format(result_file)
                msg += "." if not delete_bad else " - deleting it."
                msg += " Error was: {}".format(err) if unloadable else ""
                print(msg)

                if delete_bad:
                    os.remove(result_file)

                bad_ids.append(result_num)

        return tuple(bad_ids)

    #  ----------------------------- properties ----------------------------- #

    def _get_fn(self):
        return self._fn

    def _set_fn(self, fn):
        if self.save_fn is None and fn is not None:
            self.save_fn = True
        self._fn = fn

    def _del_fn(self):
        self._fn = None
        self.save_fn = False

    fn = property(_get_fn, _set_fn, _del_fn,
                  "Function to save with the Crop for automatic loading and "
                  "running. Default crop name will be inferred from this if"
                  "not given explicitly as well.")

    @property
    def num_sown_batches(self):
        """Total number of batches to be run/grown.
        """
        self.calc_progress()
        return self._num_sown_batches

    @property
    def num_results(self):
        self.calc_progress()
        return self._num_results


def load_crops(directory='.'):
    """Automatically load all the crops found in the current directory.

    Parameters
    ----------
    directory : str, optional
        Which directory to load the crops from, defaults to '.' - the current.

    Returns
    -------
    dict[str, Crop]
        Mapping of the crop name to the Crop.
    """
    import os
    import re

    folders = next(os.walk(directory))[1]
    crop_rgx = re.compile('^\.xyz-(.+)')

    names = []
    for folder in folders:
        match = crop_rgx.match(folder)
        if match:
            names.append(match.groups(1)[0])

    return {name: Crop(name=name) for name in names}


class Sower(object):
    """Class for sowing a 'crop' of batched combos to then 'grow' (on any
    number of workers sharing the filesystem) and then reap.
    """

    def __init__(self, crop):
        """
        Parameters
        ----------
            crop : xyzpy.batch.Crop instance
                Description of where and how to store the cases and results.
        """
        self.crop = crop
        # Internal:
        self._batch_cases = []  # collects cases to be written in single batch
        self._counter = 0  # counts how many cases are in batch so far
        self._batch_counter = 0  # counts how many batches have been written

    def save_batch(self):
        """Save the current batch of cases to disk using joblib.dump
         and start the next batch.
        """
        self._batch_counter += 1
        joblib.dump(self._batch_cases, os.path.join(
            self.crop.location, "batches", BTCH_NM.format(self._batch_counter))
        )
        self._batch_cases = []
        self._counter = 0

    # Context manager #

    def __enter__(self):
        return self

    def __call__(self, **kwargs):
        self._batch_cases.append(kwargs)
        self._counter += 1

        # when the number of cases doesn't divide the number of batches we
        #     distribute the remainder among the first crops.
        extra_batch = self._batch_counter < self.crop._batch_remainder

        if self._counter == self.crop.batchsize + int(extra_batch):
            self.save_batch()

    def __exit__(self, exception_type, exception_value, traceback):
        # Make sure any overfill also saved
        if self._batch_cases:
            self.save_batch()


def grow(batch_number, crop=None, fn=None, check_mpi=True,
         verbosity=2, debugging=False):
    """Automatically process a batch of cases into results. Should be run in an
    ".xyz-{fn_name}" folder.

    Parameters
    ----------
    batch_number : int
        Which batch to 'grow' into a set of results.
    crop : xyzpy.batch.Crop instance
        Description of where and how to store the cases and results.
    fn : callable, optional
        If specified, the function used to generate the results, otherwise
        the function will be loaded from disk.
    check_mpi : bool, optional
        Whether to check if the process is rank 0 and only save results if
        so - allows mpi functions to be simply used. Defaults to true,
        this should only be turned off if e.g. a pool of workers is being
        used to run different ``grow`` instances.
    verbosity : {0, 1, 2}, optional
        How much information to show.
    debugging : bool, optional
        Set logging level to DEBUG.
    """
    if debugging:
        import logging
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

    if crop is None:
        if os.path.relpath('.', '..')[:5] != ".xyz-":
            raise XYZError("`grow` should be run in a "
                           "\"{crop_parent}/.xyz-{crop_name}\" folder, else "
                           "`crop_parent` and `crop_name` (or `fn`) should be "
                           "specified.")
        crop_location = os.getcwd()
    else:
        crop_location = crop.location

    # load function
    if fn is None:
        fn = cloudpickle.loads(
            joblib.load(os.path.join(crop_location, FNCT_NM)))

    # load cases to evaluate
    cases = joblib.load(
        os.path.join(crop_location, "batches", BTCH_NM.format(batch_number)))

    if len(cases) == 0:
        raise ValueError("Something has gone wrong with the loading of "
                         "batch {} ".format(BTCH_NM.format(batch_number)) +
                         "for the crop at {}.".format(crop.location))

    # maybe want to run grow as mpiexec (i.e. `fn` itself in parallel),
    # so only save and delete on rank 0
    if check_mpi and 'OMPI_COMM_WORLD_RANK' in os.environ:  # pragma: no cover
        rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    elif check_mpi and 'PMI_RANK' in os.environ:  # pragma: no cover
        rank = int(os.environ['PMI_RANK'])
    else:
        rank = 0

    if rank == 0:
        results = []

        descr = "Batch {}".format(batch_number)

        pbar = progbar(range(len(cases)), disable=verbosity <= 0, desc=descr)
        for i in pbar:
            if verbosity >= 2:
                pbar.set_description(descr + ": {}".format(cases[i]))

            # compute and store result!
            results.append(fn(**cases[i]))

        if len(results) != len(cases):
            raise ValueError("Something has gone wrong with processing "
                             "batch {} ".format(BTCH_NM.format(batch_number)) +
                             "for the crop at {}.".format(crop.location))

        # save to results
        joblib.dump(tuple(results), os.path.join(
            crop_location, "results", RSLT_NM.format(batch_number)))
    else:
        for case in cases:
            # worker: just help compute the result!
            fn(**case)


# --------------------------------------------------------------------------- #
#                              Gathering results                              #
# --------------------------------------------------------------------------- #

class Reaper(object):
    """Class that acts as a stateful function to retrieve already sown and
    grow results.
    """

    def __init__(self, crop, num_batches, wait=False, default_result=None):
        """Class for retrieving the batched, flat, 'grown' results.

        Parameters
        ----------
            crop : xyzpy.batch.Crop instance
                Description of where and how to store the cases and results.
        """
        self.crop = crop

        files = (
            os.path.join(self.crop.location, "results", RSLT_NM.format(i + 1))
            for i in range(num_batches)
        )

        def _load(x):

            use_default = (
                (default_result is not None) and
                (not wait) and
                (not os.path.isfile(x))
            )

            # actual result doesn't exist yet - use the default if specified
            if use_default:
                res = (default_result,) * crop.batchsize
            else:
                res = joblib.load(x)

            if (res is None) or len(res) == 0:
                raise ValueError("Something not right: result {} contains "
                                 "no data upon joblib.load".format(x))
            return res

        def wait_to_load(x):
            while not os.path.exists(x):
                sleep(0.2)

            if os.path.isfile(x):
                return _load(x)
            else:
                raise ValueError("{} is not a file.".format(x))

        self.results = chain.from_iterable(map(
            wait_to_load if wait else _load, files))

    def __enter__(self):
        return self

    def __call__(self, **kwargs):
        return next(self.results)

    def __exit__(self, exception_type, exception_value, traceback):
        # Check everything gone acccording to plan
        if tuple(self.results):
            raise XYZError("Not all results reaped!")


# --------------------------------------------------------------------------- #
#                     Automatic Batch Submission Scripts                      #
# --------------------------------------------------------------------------- #

_SGE_HEADER = """#!/bin/bash -l
#$ -S /bin/bash
#$ -l h_rt={hours}:{minutes}:{seconds},mem={gigabytes}G
#$ -l tmpfs={temp_gigabytes}G
{extra_resources}
#$ -N {name}
mkdir -p {output_directory}
#$ -wd {output_directory}
#$ -pe {pe} {num_procs}
#$ -t {run_start}-{run_stop}
"""

_PBS_HEADER = """#!/bin/bash -l
#PBS -lselect={num_nodes}:ncpus={num_procs}:mem={gigabytes}gb
#PBS -lwalltime={hours:02}:{minutes:02}:{seconds:02}
{extra_resources}
#PBS -N {name}
#PBS -J {run_start}-{run_stop}
"""

_BASE = """export OMP_NUM_THREADS={num_threads}
export MKL_NUM_THREADS={num_threads}
export OPENBLAS_NUM_THREADS={num_threads}
{shell_setup}
tmpfile=$(mktemp .xyzpy-qsub.XXXXXXXX)
cat <<EOF > $tmpfile
{setup}
from xyzpy.gen.batch import grow, Crop
crop = Crop(name='{name}', parent_dir='{parent_dir}')
"""

_QSUB_SGE_GROW_ALL_SCRIPT = (
    "grow($SGE_TASK_ID, crop=crop, debugging={debugging})\n"
)

_QSUB_PBS_GROW_ALL_SCRIPT = (
    "grow($PBS_ARRAY_INDEX, crop=crop, debugging={debugging})\n"
)

_QSUB_SGE_GROW_PARTIAL_SCRIPT = """batch_ids = {batch_ids}
grow(batch_ids[$SGE_TASK_ID - 1], crop=crop, debugging={debugging})
"""


_QSUB_PBS_GROW_PARTIAL_SCRIPT = """batch_ids = {batch_ids}
grow(batch_ids[$PBS_ARRAY_INDEX - 1], crop=crop, debugging={debugging})
"""


_BASE_QSUB_SCRIPT_END = """EOF
{launcher} $tmpfile
rm $tmpfile
"""


def gen_qsub_script(
    crop, batch_ids=None, *,
    hours=None,
    minutes=None,
    seconds=None,
    gigabytes=2,
    num_procs=1,
    num_threads=None,
    num_nodes=1,
    launcher='python',
    setup="#",
    shell_setup="",
    mpi=False,
    temp_gigabytes=1,
    output_directory=None,
    extra_resources=None,
    debugging=False,
    scheduler='sge',
):
    """Generate a qsub script to grow a Crop.

    Parameters
    ----------
    crop : Crop
        The crop to grow.
    batch_ids : int or tuple[int]
        Which batch numbers to grow, defaults to all missing batches.
    hours : int
        How many hours to request, default=0.
    minutes : int, optional
        How many minutes to request, default=20.
    seconds : int, optional
        How many seconds to request, default=0.
    gigabytes : int, optional
        How much memory to request, default: 2.
    num_procs : int, optional
        How many processes to request (threaded cores or MPI), default: 1.
    launcher : str, optional
        How to launch the script, default: ``'python'``. But could for example
        be ``'mpiexec python'`` for a MPI program.
    setup : str, optional
        Python script to run before growing, for things that shouldnt't be put
        in the crop function itself, e.g. one-time imports with side-effects
        like: ``"import tensorflow as tf; tf.enable_eager_execution()``".
    shell_setup : str, optional
        Commands to be run by the shell before the python script is executed.
        E.g. ``conda activate my_env``.
    mpi : bool, optional
        Request MPI processes not threaded processes.
    temp_gigabytes : int, optional
        How much temporary on-disk memory.
    output_directory : str, optional
        What directory to write output to. Defaults to "$HOME/Scratch/output".
    extra_resources : str, optional
        Extra "#$ -l" resources, e.g. 'gpu=1'
    debugging : bool, optional
        Set the python log level to debugging.
    scheduler : {'sge', 'pbs'}, optional
        Whether to use a SGE or PBS submission script template.

    Returns
    -------
    str
    """
    if hours is minutes is seconds is None:
        hours, minutes, seconds = 1, 0, 0
    else:
        hours = 0 if hours is None else int(hours)
        minutes = 0 if minutes is None else int(minutes)
        seconds = 0 if seconds is None else int(seconds)

    if output_directory is None:
        from os.path import expanduser
        home = expanduser("~")
        output_directory = os.path.join(home, 'Scratch', 'output')

    crop.calc_progress()

    if extra_resources is None:
        extra_resources = ""
    else:
        extra_resources = "#$ -l {}".format(extra_resources)

    if num_threads is None:
        if mpi:
            num_threads = 1
        else:
            num_threads = num_procs

    opts = {
        'hours': hours,
        'minutes': minutes,
        'seconds': seconds,
        'gigabytes': gigabytes,
        'name': crop.name,
        'parent_dir': os.path.abspath(crop.parent_dir),
        'num_procs': num_procs,
        'num_threads': num_threads,
        'num_nodes': num_nodes,
        'run_start': 1,
        'launcher': launcher,
        'setup': setup,
        'shell_setup': shell_setup,
        'pe': 'mpi' if mpi else 'smp',
        'temp_gigabytes': temp_gigabytes,
        'output_directory': output_directory,
        'working_directory': crop.parent_dir,
        'extra_resources': extra_resources,
        'debugging': debugging,
    }

    if scheduler not in {'sge', 'pbs'}:
        raise ValueError

    if scheduler == 'sge':
        script = _SGE_HEADER
    elif scheduler == 'pbs':
        script = _PBS_HEADER

    script += _BASE

    # grow specific ids
    if batch_ids is not None:
        if scheduler == 'sge':
            script += _QSUB_SGE_GROW_PARTIAL_SCRIPT
        elif scheduler == 'pbs':
            script += _QSUB_PBS_GROW_PARTIAL_SCRIPT
        batch_ids = tuple(batch_ids)
        opts['run_stop'] = len(batch_ids)
        opts['batch_ids'] = batch_ids

    # grow all ids
    elif crop.num_results == 0:
        batch_ids = tuple(range(crop.num_batches))
        if scheduler == 'sge':
            script += _QSUB_SGE_GROW_ALL_SCRIPT
        elif scheduler == 'pbs':
            script += _QSUB_PBS_GROW_ALL_SCRIPT
        opts['run_stop'] = crop.num_batches

    # grow missing ids only
    else:
        if scheduler == 'sge':
            script += _QSUB_SGE_GROW_PARTIAL_SCRIPT
        elif scheduler == 'pbs':
            script += _QSUB_PBS_GROW_PARTIAL_SCRIPT
        batch_ids = crop.missing_results()
        opts['run_stop'] = len(batch_ids)
        opts['batch_ids'] = batch_ids

    script += _BASE_QSUB_SCRIPT_END
    script = script.format(**opts)

    if (scheduler == 'pbs') and len(batch_ids) == 1:
        # PBS can't handle arrays jobs of size 1...
        script = (script.replace('#PBS -J 1-1\n', "")
                        .replace("$PBS_ARRAY_INDEX", '1'))

    return script


def qsub_grow(
    crop, batch_ids=None, *,
    hours=None,
    minutes=None,
    seconds=None,
    gigabytes=2,
    num_procs=1,
    num_threads=None,
    num_nodes=1,
    launcher='python',
    setup="#",
    shell_setup="",
    mpi=False,
    temp_gigabytes=1,
    output_directory=None,
    extra_resources=None,
    debugging=False,
    scheduler='sge',
):  # pragma: no cover
    """Automagically submit SGE jobs to grow all missing results.

    Parameters
    ----------
    crop : Crop
        The crop to grow.
    batch_ids : int or tuple[int]
        Which batch numbers to grow, defaults to all missing batches.
    hours : int
        How many hours to request, default=0.
    minutes : int, optional
        How many minutes to request, default=20.
    seconds : int, optional
        How many seconds to request, default=0.
    gigabytes : int, optional
        How much memory to request, default: 2.
    num_procs : int, optional
        How many processes to request (threaded cores or MPI), default: 1.
    launcher : str, optional
        How to launch the script, default: ``'python'``. But could for example
        be ``'mpiexec python'`` for a MPI program.
    setup : str, optional
        Python script to run before growing, for things that shouldnt't be put
        in the crop function itself, e.g. one-time imports with side-effects
        like: ``"import tensorflow as tf; tf.enable_eager_execution()``".
    shell_setup : str, optional
        Commands to be run by the shell before the python script is executed.
        E.g. ``conda activate my_env``.
    mpi : bool, optional
        Request MPI processes not threaded processes.
    temp_gigabytes : int, optional
        How much temporary on-disk memory.
    output_directory : str, optional
        What directory to write output to. Defaults to "$HOME/Scratch/output".
    extra_resources : str, optional
        Extra "#$ -l" resources, e.g. 'gpu=1'
    debugging : bool, optional
        Set the python log level to debugging.
    scheduler : {'sge', 'pbs'}, optional
        Whether to use a SGE or PBS submission script template.
    """
    if crop.is_ready_to_reap():
        print("Crop ready to reap: nothing to submit.")
        return

    import subprocess

    script = gen_qsub_script(
        crop,
        batch_ids=batch_ids,
        hours=hours,
        minutes=minutes,
        seconds=seconds,
        gigabytes=gigabytes,
        temp_gigabytes=temp_gigabytes,
        output_directory=output_directory,
        num_procs=num_procs,
        num_threads=num_threads,
        num_nodes=num_nodes,
        launcher=launcher,
        setup=setup,
        shell_setup=shell_setup,
        mpi=mpi,
        extra_resources=extra_resources,
        debugging=debugging,
        scheduler=scheduler,
    )

    script_file = os.path.join(crop.location, "__qsub_script__.sh")

    with open(script_file, mode='w') as f:
        f.write(script)

    subprocess.run(['qsub', script_file])

    os.remove(script_file)


Crop.gen_qsub_script = gen_qsub_script
Crop.qsub_grow = qsub_grow
