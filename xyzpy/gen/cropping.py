import os
import re
import copy
import math
import time
import glob
import shutil
import pickle
import pathlib
import warnings
import functools
import importlib
import itertools

from ..utils import _get_fn_name, prod, progbar
from .combo_runner import (
    nan_like_result,
    combo_runner_core,
    combo_runner_to_ds,
    get_reusable_executor,
)
from .case_runner import (
    case_runner,
)
from .prepare import (
    parse_combos,
    parse_constants,
    parse_attrs,
    parse_fn_args,
    parse_cases,
)
from .farming import Runner, Harvester, Sampler, XYZError


BTCH_NM = "xyz-batch-{}.jbdmp"
RSLT_NM = "xyz-result-{}.jbdmp"
FNCT_NM = "xyz-function.clpkl"
INFO_NM = "xyz-settings.jbdmp"


def write_to_disk(obj, fname):
    with open(fname, "wb") as file:
        pickle.dump(obj, file)


def read_from_disk(fname):
    with open(fname, "rb") as file:
        return pickle.load(file)


@functools.lru_cache(8)
def get_picklelib(picklelib="joblib.externals.cloudpickle"):
    return importlib.import_module(picklelib)


def to_pickle(obj, picklelib="joblib.externals.cloudpickle"):
    plib = get_picklelib(picklelib)
    s = plib.dumps(obj)
    return s


def from_pickle(s, picklelib="joblib.externals.cloudpickle"):
    plib = get_picklelib(picklelib)
    obj = plib.loads(s)
    return obj


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


def parse_fn_farmer(fn, farmer):
    if farmer is not None:
        if fn is not None:
            warnings.warn(
                "'fn' is ignored if a 'Runner', 'Harvester', or "
                "'Sampler' is supplied as the 'farmer' kwarg."
            )
        fn = farmer.fn

    return fn, farmer


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
        raise XYZError(
            "This crop is not ready to reap yet - results are "
            "missing. You can reap only finished batches by setting"
            " ``allow_incomplete=True``, but be aware this will "
            "represent all missing batches with ``np.nan`` and thus"
            " might effect data-types."
        )


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
        to saving a version of `fn` to disk for `cropping.grow` to use.
    name : str, optional
        Custom name for this set of runs - must be given if `fn`
        is not.
    parent_dir : str, optional
        If given, alternative directory to put the ".xyz-{name}/"
        folder in with all the cases and results.
    save_fn : bool, optional
        Whether to save the function to disk for `cropping.grow` to use.
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

    def __init__(
        self,
        *,
        fn=None,
        name=None,
        parent_dir=None,
        save_fn=None,
        batchsize=None,
        num_batches=None,
        shuffle=False,
        farmer=None,
        autoload=True,
    ):
        self._fn, self.farmer = parse_fn_farmer(fn, farmer)

        self.name = name
        self.parent_dir = parent_dir
        self.save_fn = save_fn
        self.batchsize = batchsize
        self.num_batches = num_batches
        self.shuffle = shuffle
        self._batch_remainder = None
        self._all_nan_result = None

        # Work out the full directory for the crop
        self.location, self.name, self.parent_dir = parse_crop_details(
            self._fn, self.name, self.parent_dir
        )

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
        if combos:
            n_combos = prod(len(x) for _, x in combos)
        else:
            n_combos = 1

        if cases:
            n_cases = len(cases)
        else:
            n_cases = 1

        # for each case every combination is run
        n = n_cases * n_combos

        if (self.batchsize is not None) and (self.num_batches is not None):
            # Check that they are set correctly
            pos_tot = self.batchsize * self.num_batches
            if self._batch_remainder is not None:
                pos_tot += self._batch_remainder
            if not (n <= pos_tot < n + self.batchsize):
                raise ValueError(
                    "`batchsize` and `num_batches` cannot both"
                    "be specified if they do not not multiply"
                    "to the correct number of total cases."
                )

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
            # cap at the total number of cases
            self.num_batches = min(n, self.num_batches)

            if not isinstance(self.num_batches, int):
                raise TypeError("`num_batches` must be an integer.")
            if self.num_batches < 1:
                raise ValueError("`num_batches` must be >= 1.")

            self.batchsize, self._batch_remainder = divmod(n, self.num_batches)

    def ensure_dirs_exists(self):
        """Make sure the directory structure for this crop exists."""
        os.makedirs(os.path.join(self.location, "batches"), exist_ok=True)
        os.makedirs(os.path.join(self.location, "results"), exist_ok=True)

    def save_info(self, combos=None, cases=None, fn_args=None):
        """Save information about the sowed cases."""
        # If saving Harvester or Runner, strip out function information so
        #   as just to use pickle.
        if self.farmer is not None:
            farmer_copy = copy.deepcopy(self.farmer)
            farmer_copy.fn = None
            farmer_pkl = to_pickle(farmer_copy)
        else:
            farmer_pkl = None

        write_to_disk(
            {
                "combos": combos,
                "cases": cases,
                "fn_args": fn_args,
                "batchsize": self.batchsize,
                "num_batches": self.num_batches,
                "_batch_remainder": self._batch_remainder,
                "shuffle": self.shuffle,
                "farmer": farmer_pkl,
            },
            os.path.join(self.location, INFO_NM),
        )

    def load_info(self):
        """Load the full settings from disk."""
        sfile = os.path.join(self.location, INFO_NM)

        if not os.path.isfile(sfile):
            raise XYZError("Settings can't be found at {}.".format(sfile))
        else:
            return read_from_disk(sfile)

    def _sync_info_from_disk(self, only_missing=True):
        """Load information about the saved cases."""
        settings = self.load_info()
        self.batchsize = settings["batchsize"]
        self.num_batches = settings["num_batches"]
        self._batch_remainder = settings["_batch_remainder"]

        farmer_pkl = settings["farmer"]
        farmer = None if farmer_pkl is None else from_pickle(farmer_pkl)

        fn, farmer = parse_fn_farmer(None, farmer)

        # if crop already has a harvester/runner. (e.g. was instantiated from
        # one) by default don't overwrite from disk
        if (self.farmer) is None or (not only_missing):
            self.farmer = farmer

        if self.fn is None:
            self.load_function()

    def save_function_to_disk(self):
        """Save the base function to disk using cloudpickle"""
        write_to_disk(
            to_pickle(self._fn), os.path.join(self.location, FNCT_NM)
        )

    def load_function(self):
        """Load the saved function from disk, and try to re-insert it back into
        Harvester or Runner if present.
        """
        self._fn = from_pickle(
            read_from_disk(os.path.join(self.location, FNCT_NM))
        )

        if self.farmer is not None:
            if self.farmer.fn is None:
                self.farmer.fn = self._fn
            else:
                # TODO: check equality?
                raise XYZError(
                    "Trying to load this Crop's function, {}, from "
                    "disk but its farmer already has a function "
                    "set: {}.".format(self._fn, self.farmer.fn)
                )

    def prepare(self, combos=None, cases=None, fn_args=None):
        """Write information about this crop and the supplied combos to disk.
        Typically done at start of sow, not when Crop instantiated.
        """
        self.ensure_dirs_exists()
        if self.save_fn:
            self.save_function_to_disk()
        self.save_info(combos=combos, cases=cases, fn_args=fn_args)

    def is_prepared(self):
        """Check whether this crop has been written to disk."""
        return os.path.exists(os.path.join(self.location, INFO_NM))

    def calc_progress(self):
        """Calculate how much progressed has been made in growing the batches."""
        if self.is_prepared():
            self._sync_info_from_disk()
            self._num_sown_batches = len(
                glob.glob(
                    os.path.join(self.location, "batches", BTCH_NM.format("*"))
                )
            )
            self._num_results = len(
                glob.glob(
                    os.path.join(self.location, "results", RSLT_NM.format("*"))
                )
            )
        else:
            self._num_sown_batches = -1
            self._num_results = -1

    def is_ready_to_reap(self):
        """Have all batches been grown?"""
        self.calc_progress()
        return self._num_results > 0 and (
            self._num_results == self.num_sown_batches
        )

    def missing_results(self):
        """Return tuple of batches which haven't been grown yet."""
        self.calc_progress()

        def no_result_exists(x):
            return not os.path.isfile(
                os.path.join(self.location, "results", RSLT_NM.format(x))
            )

        return tuple(filter(no_result_exists, range(1, self.num_batches + 1)))

    def delete_all(self):
        """Delete the crop directory and all its contents."""
        # delete everything
        shutil.rmtree(self.location)

    @property
    def all_nan_result(self):
        """Get a stand-in result for cases which are missing still."""
        if self._all_nan_result is None:
            result_files = glob.glob(
                os.path.join(self.location, "results", RSLT_NM.format("*"))
            )
            if not result_files:
                raise XYZError(
                    "To infer an all-nan result requires at least "
                    "one finished result."
                )
            reference_result = read_from_disk(result_files[0])[0]
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

        return (
            "\n"
            "{location}\n"
            "{under_crop_dir}{under_crop_name}\n"
            "{num_results} / {total} batches of size {bsz} completed\n"
            "[{done_bars}{not_done_spaces}] : {percentage:.1f}%"
            "\n"
        ).format(
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
        constants = parse_constants(constants)

        if self.runner is not None:
            constants = {**self.runner._constants, **constants}
            constants = {**self.runner._resources, **constants}

        return constants

    def sow_combos(
        self,
        combos,
        cases=None,
        constants=None,
        shuffle=False,
        verbosity=1,
        batchsize=None,
        num_batches=None,
    ):
        """Sow combos to disk to be later grown, potentially in batches. Note
        if you have already sown this `Crop`, as long as the number of batches
        hasn't changed (e.g. you have just tweaked the function or a constant
        argument), you can safely resow and only the batches will be
        overwritten, i.e. the results will remain.

        Parameters
        ----------
        combos : dict_like[str, iterable]
            The combinations to sow for all or some function arguments.
        cases : iterable or mappings, optional
            Optionally provide an sequence of individual cases to sow for some
            or all function arguments.
        constants : mapping, optional
            Provide additional constant function values to use when sowing.
        shuffle : bool or int, optional
            If given, sow the combos in a random order (using ``random.seed``
            and ``random.shuffle``), which can be helpful for distributing
            resources when not all cases are computationally equal.
        verbosity : int, optional
            How much information to show when sowing.
        batchsize : int, optional
            If specified, set a new batchsize for the crop.
        num_batches : int, optional
            If specified, set a new num_batches for the crop.
        """
        if batchsize is not None:
            self.batchsize = batchsize
        if num_batches is not None:
            self.num_batches = num_batches
        if shuffle is not None:
            self.shuffle = shuffle

        combos = parse_combos(combos)
        cases = parse_cases(cases)
        constants = self.parse_constants(constants)

        # Sort to ensure order remains same for reaping results
        #   (don't want to hash kwargs)
        combos = sorted(combos, key=lambda x: x[0])

        self.choose_batch_settings(combos=combos, cases=cases)
        self.prepare(combos=combos, cases=cases)

        with Sower(self) as sow_fn:
            combo_runner_core(
                fn=sow_fn,
                combos=combos,
                cases=cases,
                constants=constants,
                shuffle=shuffle,
                verbosity=verbosity,
            )

    def sow_cases(
        self,
        fn_args,
        cases,
        combos=None,
        constants=None,
        verbosity=1,
        batchsize=None,
        num_batches=None,
    ):
        """Sow cases to disk to be later grown, potentially in batches.

        Parameters
        ----------
        fn_args : iterable[str] or str
            The names and order of the function arguments, can be ``None`` if
            each case is supplied as a ``dict``.
        cases : iterable or mappings, optional
            Sequence of individual cases to sow for all or some function
            arguments.
        combos : dict_like[str, iterable]
            Combinations to sow for some or all function arguments.
        constants : mapping, optional
            Provide additional constant function values to use when sowing.
        verbosity : int, optional
            How much information to show when sowing.
        batchsize : int, optional
            If specified, set a new batchsize for the crop.
        num_batches : int, optional
            If specified, set a new num_batches for the crop.
        """
        if batchsize is not None:
            self.batchsize = batchsize
        if num_batches is not None:
            self.num_batches = num_batches

        fn_args = parse_fn_args(self._fn, fn_args)
        cases = parse_cases(cases, fn_args)
        constants = self.parse_constants(constants)

        self.choose_batch_settings(combos=combos, cases=cases)
        self.prepare(fn_args=fn_args, combos=combos, cases=cases)

        with Sower(self) as sow_fn:
            case_runner(
                fn=sow_fn,
                fn_args=fn_args,
                cases=cases,
                combos=combos,
                constants=constants,
                verbosity=verbosity,
                parse=False,
            )

    def sow_samples(self, n, combos=None, constants=None, verbosity=1):
        """Sow ``n`` samples to disk."""
        fn_args, cases = self.farmer.gen_cases_fnargs(n, combos)
        self.sow_cases(
            fn_args, cases, constants=constants, verbosity=verbosity
        )

    def grow(self, batch_ids, **combo_runner_opts):
        """Grow specific batch numbers using this process."""
        if isinstance(batch_ids, int):
            batch_ids = (batch_ids,)

        combo_runner_core(
            grow,
            combos=(("batch_number", batch_ids),),
            constants={"verbosity": 0, "crop": self},
            **combo_runner_opts,
        )

    def grow_missing(self, **combo_runner_opts):
        """Grow any missing results using this process."""
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
        settings = self.load_info()

        with Reaper(
            self,
            num_batches=settings["num_batches"],
            wait=wait,
            default_result=default_result,
        ) as reap_fn:
            results = combo_runner_core(
                fn=reap_fn,
                combos=settings["combos"],
                cases=settings["cases"],
                constants={},
                shuffle=settings.get("shuffle", False),
            )

        if clean_up:
            self.delete_all()

        return results

    def reap_combos_to_ds(
        self,
        var_names=None,
        var_dims=None,
        var_coords=None,
        constants=None,
        attrs=None,
        parse=True,
        wait=False,
        clean_up=None,
        allow_incomplete=False,
        to_df=False,
    ):
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
        to_df : bool, optional
            Whether to reap to a ``xarray.Dataset`` or a ``pandas.DataFrame``.

        Returns
        -------
        xarray.Dataset or pandas.Dataframe
            Multidimensional labelled dataset contatining all the results.
        """
        check_ready_to_reap(self, allow_incomplete, wait)

        clean_up, default_result = calc_clean_up_default_res(
            self, clean_up, allow_incomplete
        )

        # load exact same combinations as cases saved with
        settings = self.load_info()

        if parse:
            constants = parse_constants(constants)
            attrs = parse_attrs(attrs)

        with Reaper(
            self,
            num_batches=settings["num_batches"],
            wait=wait,
            default_result=default_result,
        ) as reap_fn:
            # move constants into attrs, so as not to pass them to the Reaper
            #   when if fact they were meant for the original function.
            data = combo_runner_to_ds(
                fn=reap_fn,
                combos=settings["combos"],
                cases=settings["cases"],
                var_names=var_names,
                var_dims=var_dims,
                var_coords=var_coords,
                constants={},
                resources={},
                attrs={**constants, **attrs},
                shuffle=settings.get("shuffle", False),
                parse=parse,
                to_df=to_df,
            )

        if clean_up:
            self.delete_all()

        return data

    def reap_runner(
        self,
        runner,
        wait=False,
        clean_up=None,
        allow_incomplete=False,
        to_df=False,
    ):
        """Reap a Crop over sowed combos and save to a dataset defined by a
        :class:`~xyzpy.Runner`.
        """
        # Can ignore `Runner.resources` as they play no part in desecribing the
        #   output, though they should be supplied to sow and thus grow.
        data = self.reap_combos_to_ds(
            var_names=runner._var_names,
            var_dims=runner._var_dims,
            var_coords=runner._var_coords,
            constants=runner._constants,
            attrs=runner._attrs,
            parse=False,
            wait=wait,
            clean_up=clean_up,
            allow_incomplete=allow_incomplete,
            to_df=to_df,
        )

        if to_df:
            runner._last_df = data
        else:
            runner._last_ds = data

        return data

    def reap_harvest(
        self,
        harvester,
        wait=False,
        sync=True,
        overwrite=None,
        clean_up=None,
        allow_incomplete=False,
    ):
        """Reap a Crop over sowed combos and merge with the dataset defined by
        a :class:`~xyzpy.Harvester`.
        """
        if harvester is None:
            raise ValueError("Cannot reap and harvest if no Harvester is set.")

        ds = self.reap_runner(
            harvester.runner,
            wait=wait,
            clean_up=False,
            allow_incomplete=allow_incomplete,
            to_df=False,
        )

        if sync:
            harvester.add_ds(ds, sync=sync, overwrite=overwrite)

        # defer cleaning up until we have sucessfully synced new dataset
        if clean_up is None:
            clean_up = not allow_incomplete
        if clean_up:
            self.delete_all()

        return ds

    def reap_samples(
        self,
        sampler,
        wait=False,
        sync=True,
        clean_up=None,
        allow_incomplete=False,
    ):
        """Reap a Crop over sowed combos and merge with the dataframe defined
        by a :class:`~xyzpy.Sampler`.
        """
        if sampler is None:
            raise ValueError("Cannot reap samples without a 'Sampler'.")

        df = self.reap_runner(
            sampler.runner,
            wait=wait,
            clean_up=clean_up,
            allow_incomplete=allow_incomplete,
            to_df=True,
        )

        if sync:
            sampler._last_df = df
            sampler.add_df(df, sync=sync)

        return df

    def reap(
        self,
        wait=False,
        sync=True,
        overwrite=None,
        clean_up=None,
        allow_incomplete=False,
    ):
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
        opts = dict(
            clean_up=clean_up, wait=wait, allow_incomplete=allow_incomplete
        )

        if isinstance(self.farmer, Runner):
            return self.reap_runner(self.farmer, **opts)

        if isinstance(self.farmer, Harvester):
            opts["overwrite"] = overwrite
            return self.reap_harvest(self.farmer, sync=sync, **opts)

        if isinstance(self.farmer, Sampler):
            return self.reap_samples(self.farmer, sync=sync, **opts)

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
        result_files = glob.glob(
            os.path.join(self.location, "results", RSLT_NM.format("*"))
        )

        bad_ids = []

        for result_file in result_files:
            # load corresponding batch file to check length.
            result_num = (
                os.path.split(result_file)[-1]
                .strip("xyz-result-")
                .strip(".jbdmp")
            )
            batch_file = os.path.join(
                self.location, "batches", BTCH_NM.format(result_num)
            )

            batch = read_from_disk(batch_file)

            try:
                result = read_from_disk(result_file)
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

    fn = property(
        _get_fn,
        _set_fn,
        _del_fn,
        "Function to save with the Crop for automatic loading and "
        "running. Default crop name will be inferred from this if"
        "not given explicitly as well.",
    )

    @property
    def num_sown_batches(self):
        """Total number of batches to be run/grown."""
        self.calc_progress()
        return self._num_sown_batches

    @property
    def num_results(self):
        self.calc_progress()
        return self._num_results


def load_crops(directory="."):
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
    crop_rgx = re.compile(r"^\.xyz-(.+)")

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
        crop : xyzpy.Crop
            Description of where and how to store the cases and results.
        """
        self.crop = crop
        # Internal:
        self._batch_cases = []  # collects cases to be written in single batch
        self._counter = 0  # counts how many cases are in batch so far
        self._batch_counter = 0  # counts how many batches have been written

    def save_batch(self):
        """Save the current batch of cases to disk and start the next batch."""
        self._batch_counter += 1
        write_to_disk(
            self._batch_cases,
            os.path.join(
                self.crop.location,
                "batches",
                BTCH_NM.format(self._batch_counter),
            ),
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


def grow(
    batch_number,
    crop=None,
    fn=None,
    num_workers=None,
    check_mpi=True,
    verbosity=2,
    debugging=False,
):
    """Automatically process a batch of cases into results. Should be run in an
    ".xyz-{fn_name}" folder.

    Parameters
    ----------
    batch_number : int
        Which batch to 'grow' into a set of results.
    crop : xyzpy.Crop
        Description of where and how to store the cases and results.
    fn : callable, optional
        If specified, the function used to generate the results, otherwise
        the function will be loaded from disk.
    num_workers : int, optional
        If specified, grow using a pool of this many workers. This uses
        ``joblib.externals.loky`` to spawn processes.
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

    # first we load the function and batch of cases to run

    if crop is None:
        current_folder = os.path.relpath(".", "..")
        if current_folder[:5] != ".xyz-":
            raise XYZError(
                "`grow` should be run in a "
                '"{crop_parent}/.xyz-{crop_name}" folder, else '
                "`crop_parent` and `crop_name` (or `fn`) should be "
                "specified."
            )
        crop_name = current_folder[5:]
        crop_location = os.getcwd()
    else:
        crop_name = crop.name
        crop_location = crop.location

    fn_file = os.path.join(crop_location, FNCT_NM)
    cases_file = os.path.join(
        crop_location, "batches", BTCH_NM.format(batch_number)
    )
    results_file = os.path.join(
        crop_location, "results", RSLT_NM.format(batch_number)
    )

    # load function
    if fn is None:
        fn = from_pickle(read_from_disk(fn_file))

    # load cases to evaluate
    cases = read_from_disk(cases_file)

    if len(cases) == 0:
        raise ValueError(
            "Something has gone wrong with the loading of "
            "batch {} ".format(BTCH_NM.format(batch_number))
            + "for the crop at {}.".format(crop.location)
        )
    if (verbosity >= 1):
        print(f"xyzpy: loaded batch {batch_number} of {crop_name}.")

    # maybe want to run grow as mpiexec (i.e. `fn` itself in parallel),
    # so only save and delete on rank 0
    if check_mpi and "OMPI_COMM_WORLD_RANK" in os.environ:  # pragma: no cover
        mpi = True
        rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
    elif check_mpi and "PMI_RANK" in os.environ:  # pragma: no cover
        mpi = True
        rank = int(os.environ["PMI_RANK"])
    else:
        mpi = False
        rank = 0
    if mpi and (verbosity >= 1):
        print(f"xyzpy: detected mpi rank {rank}.")

    # create a lazy iterator over the results
    if num_workers is None:
        # sequential
        results_it = (fn(**case) for case in cases)
    else:
        # parallel
        executor = get_reusable_executor(max_workers=num_workers)
        fs = [executor.submit(fn, **case) for case in cases]
        results_it = (f.result() for f in fs)

    if verbosity >= 1:
        results_it = progbar(results_it, total=len(cases))

    # get the actual results!
    results = []
    for i, r in enumerate(results_it):
        if (verbosity >= 2):
            results_it.set_description(f"{cases[i]}")
        results.append(r)

    if rank == 0:
        # only save to results file if the main worker
        write_to_disk(tuple(results), results_file)

    if (verbosity >= 1):
        print(f"xyzpy: success - batch {batch_number} completed.")


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
        crop : xyzpy.Crop
            Description of where and how to store the cases and results.
        """
        self.crop = crop

        files = (
            os.path.join(self.crop.location, "results", RSLT_NM.format(i + 1))
            for i in range(num_batches)
        )

        def _load(x):
            use_default = (
                (default_result is not None)
                and (not wait)
                and (not os.path.isfile(x))
            )

            # actual result doesn't exist yet - use the default if specified
            if use_default:
                i = int(re.findall(RSLT_NM.format(r"(\d+)"), x)[0])
                size = crop.batchsize + int(i < crop._batch_remainder)
                res = (default_result,) * size
            else:
                res = read_from_disk(x)

            if (res is None) or len(res) == 0:
                raise ValueError(
                    "Something not right: result {} contains "
                    "no data upon read from disk.".format(x)
                )
            return res

        def wait_to_load(x):
            while not os.path.exists(x):
                time.sleep(0.2)

            if os.path.isfile(x):
                return _load(x)
            else:
                raise ValueError("{} is not a file.".format(x))

        self.results = itertools.chain.from_iterable(
            map(wait_to_load if wait else _load, files)
        )

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

_SGE_HEADER = (
    "#!/bin/bash -l\n"
    "#$ -S /bin/bash\n"
    "#$ -l h_rt={hours}:{minutes}:{seconds},mem={gigabytes}G\n"
    "#$ -l tmpfs={temp_gigabytes}G\n"
    "#$ -N {name}\n"
    "mkdir -p {output_directory}\n"
    "#$ -wd {output_directory}\n"
    "#$ -pe {pe} {num_procs}\n"
    "{extra_resources}\n"
)
_SGE_ARRAY_HEADER = (
    "#$ -t {run_start}-{run_stop}\n"
)

_PBS_HEADER = (
    "#!/bin/bash -l\n"
    "#PBS -lselect={num_nodes}:ncpus={num_procs}:mem={gigabytes}gb\n"
    "#PBS -lwalltime={hours:02}:{minutes:02}:{seconds:02}\n"
    "#PBS -N {name}\n"
    "{extra_resources}\n"
)
_PBS_ARRAY_HEADER = (
    "#PBS -J {run_start}-{run_stop}\n"
)

_SLURM_HEADER = (
    "#!/bin/bash -l\n"
    "#SBATCH --nodes={num_nodes}\n"
    "#SBATCH --cpus-per-task={num_procs}\n"
    "#SBATCH --time={hours:02}:{minutes:02}:{seconds:02}\n"
    "#SBATCH --mem={gigabytes}G\n"
    "#SBATCH --job-name={name}\n"
    "{extra_resources}\n"
)
_SLURM_ARRAY_HEADER = (
    "#SBATCH --array={run_start}-{run_stop}\n"
)

_BASE = (
    "cd {working_directory}\n"
    "export OMP_NUM_THREADS={num_threads}\n"
    "export MKL_NUM_THREADS={num_threads}\n"
    "export OPENBLAS_NUM_THREADS={num_threads}\n"
    "export NUMBA_NUM_THREADS={num_threads}\n"
    "{shell_setup}\n"
    "read -r -d '' SCRIPT << EOM\n"
    "{setup}\n"
    "from xyzpy.gen.cropping import grow, Crop\n"
    "if __name__ == '__main__':\n"
    "    crop = Crop(name='{name}', parent_dir='{parent_dir}')\n"
)
_ARRAY_GROW_KWARGS = (
    "    grow_kwargs = dict(crop=crop, debugging={debugging}, "
    "num_workers={num_workers})\n"
)

_CLUSTER_SGE_GROW_ALL_SCRIPT = (
    _ARRAY_GROW_KWARGS +
    "    grow($SGE_TASK_ID, **grow_kwargs)\n"
)

_CLUSTER_PBS_GROW_ALL_SCRIPT = (
    _ARRAY_GROW_KWARGS +
    "    grow($PBS_ARRAY_INDEX, **grow_kwargs)\n"
)

_CLUSTER_SLURM_GROW_ALL_SCRIPT = (
    _ARRAY_GROW_KWARGS +
    "    grow($SLURM_ARRAY_TASK_ID, **grow_kwargs)\n"
)

_CLUSTER_SGE_GROW_PARTIAL_SCRIPT = (
    _ARRAY_GROW_KWARGS +
    "    batch_ids = {batch_ids}]\n"
    "    grow(batch_ids[$SGE_TASK_ID - 1], **grow_kwargs)\n"
)

_CLUSTER_PBS_GROW_PARTIAL_SCRIPT = (
    _ARRAY_GROW_KWARGS +
    "    batch_ids = {batch_ids}\n"
    "    grow(batch_ids[$PBS_ARRAY_INDEX - 1], **grow_kwargs)\n"
)

_CLUSTER_SLURM_GROW_PARTIAL_SCRIPT = (
    _ARRAY_GROW_KWARGS +
    "    batch_ids = {batch_ids}\n"
    "    grow(batch_ids[$SLURM_ARRAY_TASK_ID - 1], **grow_kwargs)\n"
)

_BASE_CLUSTER_GROW_SINGLE = (
    "    batch_ids = {batch_ids}\n"
    "    crop.grow(batch_ids, num_workers={num_workers})\n"
)

_BASE_CLUSTER_SCRIPT_END = (
    "EOM\n"
    '{launcher} -c "$SCRIPT"\n'
    "echo 'XYZPY script finished'\n"
)


def gen_cluster_script(
    crop,
    scheduler,
    batch_ids=None,
    *,
    mode="array",
    hours=None,
    minutes=None,
    seconds=None,
    gigabytes=2,
    num_procs=1,
    num_threads=None,
    num_nodes=1,
    num_workers=None,
    conda_env=True,
    launcher="python",
    setup="#",
    shell_setup="",
    mpi=False,
    temp_gigabytes=1,
    output_directory=None,
    debugging=False,
    **kwargs,
):
    """Generate a cluster script to grow a Crop.

    Parameters
    ----------
    crop : Crop
        The crop to grow.
    scheduler : {'sge', 'pbs', 'slurm'}
        Whether to use a SGE, PBS or slurm submission script template.
    batch_ids : int or tuple[int]
        Which batch numbers to grow, defaults to all missing batches.
    mode : {'array', 'single'}
        How to distribute the batches, either as an array job with a single
        batch per job, or as a single job processing batches in parallel.
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
    num_threads : int, optional
        How many threads to use per process. Will be computed automatically
        based on ``num_procs`` and ``num_workers`` if not specified.
    num_workers : int, optional
        How many workers to use for parallel growing, default is sequential. If
        specified, then generally ``num_workers * num_threads == num_procs``.
    num_nodes : int, optional
        How many nodes to request, default: 1.
    conda_env : bool or str, optional
        Whether to activate a conda environment before running the script.
        If ``True``, the environment will be the same as the one used to
        launch the script. If a string, the environment will be the one
        specified by the string.
    launcher : str, optional
        How to launch the script, default: ``'python'``. But could for example
        be ``'mpiexec python'`` for a MPI program.
    setup : str, optional
        Python script to run before growing, for things that shouldnt't be put
        in the crop function itself, e.g. one-time imports with side-effects
        like: ``"import tensorflow as tf; tf.enable_eager_execution()``".
    shell_setup : str, optional
        Commands to be run by the shell before the python script is executed.
    mpi : bool, optional
        Request MPI processes not threaded processes
    temp_gigabytes : int, optional
        How much temporary on-disk memory.
    output_directory : str, optional
        What directory to write output to. Defaults to "$HOME/Scratch/output".
    debugging : bool, optional
        Set the python log level to debugging.
    kwargs : dict, optional
        Extra keyword arguments are taken to be extra resources to request
        in the header of the submission script, e.g. ``{'gpu': 1}`` will
        add ``"#SBATCH --gpu=1"`` to the header if using slurm. If you supply
        literal ``True`` or ``None`` as the value, then the key will be treated
        as a flag. E.g. ``{'requeue': None}`` will add ``"#SBATCH --requeue"``
        to the header.

    Returns
    -------
    str
    """

    scheduler = scheduler.lower()  # be case-insensitive for scheduler

    if scheduler not in ("sge", "pbs", "slurm"):
        raise ValueError("scheduler must be one of 'sge', 'pbs', or 'slurm'.")

    if mode not in ("array", "single"):
        raise ValueError("mode must be one of 'array' or 'single'.")

    if hours is minutes is seconds is None:
        hours, minutes, seconds = 1, 0, 0
    else:
        hours = 0 if hours is None else int(hours)
        minutes = 0 if minutes is None else int(minutes)
        seconds = 0 if seconds is None else int(seconds)

    if output_directory is None:
        from os.path import expanduser

        home = expanduser("~")
        output_directory = os.path.join(home, "Scratch", "output")

    if conda_env is True:
        # automatically set conda environment to be the
        # same as the one that's running this function
        conda_env = os.environ.get("CONDA_DEFAULT_ENV", False)
        if conda_env:
            # but only if we are in a conda environment
            if ("conda activate" in shell_setup) or (
                "mamba activate" in shell_setup
            ):
                # and user is not already explicitly activating
                conda_env = False

    if isinstance(conda_env, str):
        # should now be a string
        shell_setup += f"\nconda activate {conda_env}"
    elif conda_env is not False:
        raise ValueError(
            "conda_env must be either ``False``, "
            f"``True`` or a string, not {conda_env}"
        )

    crop.calc_progress()

    if kwargs:
        if scheduler == "slurm":
            extra_resources = "\n".join([
                f"#SBATCH --{k}={v}"
                if v not in (True, None) else
                f"#SBATCH --{k}"
                for k, v in kwargs.items()
            ])
        elif scheduler == "pbs":
            extra_resources = "\n".join([
                f"#PBS -l {k}={v}"
                if v not in (True, None) else
                f"#PBS -l {k}"
                for k, v in kwargs.items()
            ])
        elif scheduler == "sge":
            extra_resources = "\n".join([
                f"#$ -l {k}={v}"
                if v not in (True, None) else
                f"#$ -l {k}"
                for k, v in kwargs.items()
            ])
    else:
        extra_resources = ""

    if num_threads is None:
        if mpi:
            # assume single thread per rank
            num_threads = 1
        else:
            if num_workers is None:
                # assume all multithreading over all cores
                num_threads = num_procs
            else:
                # assume each worker has equal number of threads
                num_threads = max(1, num_procs // num_workers)

    if num_workers is not None:
        if num_workers * num_threads != num_procs:
            warnings.warn(
                f"num_workers * num_threads ({num_workers} * {num_threads}) "
                f"!= num_procs ({num_procs}), may not be computationally "
                "efficient."
            )

    # get absolute path
    full_parent_dir = str(pathlib.Path(crop.parent_dir).expanduser().resolve())

    opts = {
        "hours": hours,
        "minutes": minutes,
        "seconds": seconds,
        "gigabytes": gigabytes,
        "name": crop.name,
        "parent_dir": full_parent_dir,
        "num_procs": num_procs,
        "num_threads": num_threads,
        "num_nodes": num_nodes,
        "num_workers": num_workers,
        "launcher": launcher,
        "setup": setup,
        "shell_setup": shell_setup,
        "pe": "mpi" if mpi else "smp",
        "temp_gigabytes": temp_gigabytes,
        "output_directory": output_directory,
        "working_directory": full_parent_dir,
        "extra_resources": extra_resources,
        "debugging": debugging,
    }

    if batch_ids is not None:
        # grow specific ids
        opts["batch_ids"] = tuple(batch_ids)
        array_mode = "partial"
    elif crop.num_results == 0:
        # grow all ids
        opts["batch_ids"] = range(1, crop.num_batches + 1)
        array_mode = "all"
    else:
        # find missing ids and grow them
        opts["batch_ids"] = crop.missing_results()
        array_mode = "partial"

    # build the script!

    if scheduler == "sge":
        script = _SGE_HEADER
        if mode == "array":
            script += _SGE_ARRAY_HEADER
    elif scheduler == "pbs":
        script = _PBS_HEADER
        if mode == "array":
            script += _PBS_ARRAY_HEADER
    elif scheduler == "slurm":
        script = _SLURM_HEADER
        if mode == "array":
            script += _SLURM_ARRAY_HEADER

    script += _BASE

    if mode == "array":
        opts["run_start"] = 1

        if array_mode == "all":
            opts["run_stop"] = crop.num_batches
            if scheduler == "sge":
                script += _CLUSTER_SGE_GROW_ALL_SCRIPT
            elif scheduler == "pbs":
                script += _CLUSTER_PBS_GROW_ALL_SCRIPT
            elif scheduler == "slurm":
                script += _CLUSTER_SLURM_GROW_ALL_SCRIPT

        elif array_mode == "partial":
            opts["run_stop"] = len(opts["batch_ids"])
            if scheduler == "sge":
                script += _CLUSTER_SGE_GROW_PARTIAL_SCRIPT
            elif scheduler == "pbs":
                script += _CLUSTER_PBS_GROW_PARTIAL_SCRIPT
            elif scheduler == "slurm":
                script += _CLUSTER_SLURM_GROW_PARTIAL_SCRIPT

    elif mode == "single":
        if batch_ids is None:
            # grow all missing, but compute the list dynamically
            # this allows the job to be restarted
            opts["batch_ids"] = "crop.missing_results()"
        script += _BASE_CLUSTER_GROW_SINGLE

    script += _BASE_CLUSTER_SCRIPT_END
    script = script.format(**opts)

    if (scheduler == "pbs") and len(opts["batch_ids"]) == 1:
        # PBS can't handle arrays jobs of size 1...
        script = script.replace("#PBS -J 1-1\n", "").replace(
            "$PBS_ARRAY_INDEX", "1"
        )

    return script


def grow_cluster(
    crop,
    scheduler,
    batch_ids=None,
    *,
    hours=None,
    minutes=None,
    seconds=None,
    gigabytes=2,
    num_procs=1,
    num_threads=None,
    num_nodes=1,
    num_workers=None,
    conda_env=True,
    launcher="python",
    setup="#",
    shell_setup="",
    mpi=False,
    temp_gigabytes=1,
    output_directory=None,
    debugging=False,
    **kwargs,
):  # pragma: no cover
    """Automagically submit SGE, PBS, or slurm jobs to grow all missing
    results.

    Parameters
    ----------
    crop : Crop
        The crop to grow.
    scheduler : {'sge', 'pbs', 'slurm'}
        Whether to use a SGE, PBS or slurm submission script template.
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
    debugging : bool, optional
        Set the python log level to debugging.
    """
    if crop.is_ready_to_reap():
        print("Crop ready to reap: nothing to submit.")
        return

    import subprocess

    script = gen_cluster_script(
        crop,
        scheduler,
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
        num_workers=num_workers,
        conda_env=conda_env,
        launcher=launcher,
        setup=setup,
        shell_setup=shell_setup,
        mpi=mpi,
        debugging=debugging,
        **kwargs,
    )

    script_file = os.path.join(crop.location, "__qsub_script__.sh")

    with open(script_file, mode="w") as f:
        f.write(script)

    if scheduler in {"sge", "pbs"}:
        result = subprocess.run(["qsub", script_file], capture_output=True)
    elif scheduler == "slurm":
        result = subprocess.run(["sbatch", script_file], capture_output=True)

    print(result.stderr.decode())
    print(result.stdout.decode())

    os.remove(script_file)


def gen_qsub_script(
    crop, batch_ids=None, *, scheduler="sge", **kwargs
):  # pragma: no cover
    """Generate a qsub script to grow a Crop. Deprecated in favour of
    `gen_cluster_script` and will be removed in the future.

    Parameters
    ----------
    crop : Crop
        The crop to grow.
    batch_ids : int or tuple[int]
        Which batch numbers to grow, defaults to all missing batches.
    scheduler : {'sge', 'pbs'}, optional
        Whether to use a SGE or PBS submission script template.
    kwargs
        See `gen_cluster_script` for all other parameters.
    """
    warnings.warn(
        "'gen_qsub_script' is deprecated in favour of "
        "`gen_cluster_script` and will be removed in the future",
        FutureWarning,
    )
    return gen_cluster_script(crop, scheduler, batch_ids=batch_ids, **kwargs)


def qsub_grow(
    crop, batch_ids=None, *, scheduler="sge", **kwargs
):  # pragma: no cover
    """Automagically submit SGE or PBS jobs to grow all missing results.
    Deprecated in favour of `grow_cluster` and will be removed in the future.

    Parameters
    ----------
    crop : Crop
        The crop to grow.
    batch_ids : int or tuple[int]
        Which batch numbers to grow, defaults to all missing batches.
    scheduler : {'sge', 'pbs'}, optional
        Whether to use a SGE or PBS submission script template.
    kwargs
        See `grow_cluster` for all other parameters.
    """
    warnings.warn(
        "'qsub_grow' is deprecated in favour of "
        "`grow_cluster` and will be removed in the future",
        FutureWarning,
    )
    grow_cluster(crop, scheduler, batch_ids=batch_ids, **kwargs)


Crop.gen_qsub_script = gen_qsub_script
Crop.qsub_grow = qsub_grow
Crop.gen_cluster_script = gen_cluster_script
Crop.grow_cluster = grow_cluster

Crop.gen_sge_script = functools.partialmethod(
    Crop.gen_cluster_script, scheduler="sge"
)
Crop.grow_sge = functools.partialmethod(Crop.grow_cluster, scheduler="sge")

Crop.gen_pbs_script = functools.partialmethod(
    Crop.gen_cluster_script, scheduler="pbs"
)
Crop.grow_pbs = functools.partialmethod(Crop.grow_cluster, scheduler="pbs")

Crop.gen_slurm_script = functools.partialmethod(
    Crop.gen_cluster_script, scheduler="slurm"
)
Crop.grow_slurm = functools.partialmethod(Crop.grow_cluster, scheduler="slurm")


def clean_slurm_outputs(job, directory=".", cancel_if_finished=True):
    """ """
    import re
    import pathlib
    import subprocess

    job = str(job)

    files = list(pathlib.Path(directory).glob(f"slurm-{job}_*.out"))

    for file in files:
        jobid = int(re.match("slurm-\d+_(\d+).out", str(file)).groups()[0])

        with open(file, "r") as f:
            contents = f.read()

        jname = f"{job}_{jobid}"
        print(jname, end=" ")

        if f"batch {jobid} completed" in contents:
            print("xyzpy finished!", end=" ")

            if cancel_if_finished:
                # check if job queued still
                status = subprocess.run(
                    ["scontrol", "show", "job", jname], capture_output=True
                ).stdout.decode()

                running = "JobState=RUNNING" in status
                if running:
                    print("slurm cancelling,", end=" ")
                    subprocess.run(["scancel", jname])
                else:
                    print("slurm finished,", end=" ")

            # delete the output file
            print("deleting output.", end=" ")
            file.unlink()
        else:
            print("xyzpy running...", end=" ")

        print()

    return len(files)


def manage_slurm_outputs(crop, job, wait_time=60):
    import time
    from IPython.display import clear_output

    try:
        while True:
            clear_output(wait=True)
            print(crop)
            clean_slurm_outputs(job)

            if crop.is_ready_to_reap():
                break

            time.sleep(wait_time)
    except KeyboardInterrupt:
        pass
