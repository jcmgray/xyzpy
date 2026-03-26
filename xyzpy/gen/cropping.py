import copy
import functools
import glob
import importlib
import itertools
import math
import os
import pickle
import re
import shutil
import sys
import time
import warnings
from pathlib import Path

from ..utils import _get_fn_name, prod, progbar
from .case_runner import (
    case_runner,
)
from .combo_runner import (
    combo_runner_core,
    combo_runner_to_ds,
    get_reusable_executor,
    nan_like_result,
)
from .farming import Harvester, Runner, Sampler, XYZError
from .prepare import (
    parse_attrs,
    parse_cases,
    parse_combos,
    parse_constants,
    parse_fn_args,
)

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

    crop_parent_path = Path.cwd() if crop_parent is None else Path(crop_parent)
    crop_location = crop_parent_path / f".xyz-{crop_name}"

    return str(crop_location), crop_name, str(crop_parent_path)


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


# ----------- subprocess resource pooling ----------- #


def _parse_resource_ids(raw):
    """Normalize an int, list, tuple, range, or comma-separated string
    into a list of integer resource IDs.
    """
    if isinstance(raw, int):
        return [raw]
    elif isinstance(raw, (list, tuple, range)):
        return list(map(int, raw))
    else:
        return list(map(int, raw.split(",")))


def _acquire_affinity(rid, pargs, env):
    """Prepend ``taskset -c <cpu>`` to pin to a CPU core."""
    pargs[0:0] = ["taskset", "-c", str(rid)]


def _acquire_gpu(rid, pargs, env):
    """Set ``CUDA_VISIBLE_DEVICES`` to pin to a GPU."""
    env["CUDA_VISIBLE_DEVICES"] = str(rid)


class _ResourcePool:
    """A pool of reusable resource IDs (CPUs, GPUs, etc.) that can be
    acquired and released once per batch subprocess.

    Parameters
    ----------
    ids : list of int
        The resource IDs available to hand out.
    acquire_fn : callable
        ``fn(rid, pargs, env)`` — mutate *pargs* (the command prefix
        list) and/or *env* (the environment dict) to apply *rid*.
    """

    def __init__(self, ids, acquire_fn):
        self.free = list(ids)
        self.used = {}
        self.acquire_fn = acquire_fn

    @classmethod
    def from_raw(cls, raw, acquire_fn):
        """Create a pool from a raw user value, or return ``None``."""
        if raw is None:
            return None
        return cls(_parse_resource_ids(raw), acquire_fn)

    def available(self):
        """Whether there is at least one free resource."""
        return bool(self.free)

    def acquire(self, batch_id, pargs, env):
        """Pop a resource, apply it, and track it against *batch_id*."""
        rid = self.free.pop()
        self.used[batch_id] = rid
        self.acquire_fn(rid, pargs, env)

    def release(self, batch_id):
        """Return the resource used by *batch_id* to the free pool."""
        self.free.append(self.used.pop(batch_id))


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
        self._num_sown_batches = -1
        self._num_results = -1

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

        if cases is not None:
            n_cases = len(cases)
        else:
            n_cases = 1

        # for each case every combination is run
        n = n_cases * n_combos

        if (self.batchsize is not None) and (self.num_batches is not None):
            if not isinstance(self.batchsize, int):
                raise TypeError("`batchsize` must be an integer.")
            if self.batchsize < 1:
                raise ValueError("`batchsize` must be >= 1.")
            if not isinstance(self.num_batches, int):
                raise TypeError("`num_batches` must be an integer.")
            if self.num_batches < 1:
                raise ValueError("`num_batches` must be >= 1.")

            # check that the supplied settings cover all cases directly,
            # independent of any stale remainder from a previous layout.
            pos_tot = self.batchsize * self.num_batches
            if not (n <= pos_tot < n + self.batchsize):
                raise ValueError(
                    "`batchsize` and `num_batches` cannot both"
                    "be specified if they do not not multiply"
                    "to the correct number of total cases."
                )
            self._batch_remainder = 0

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
        root = Path(self.location)
        (root / "batches").mkdir(parents=True, exist_ok=True)
        (root / "results").mkdir(parents=True, exist_ok=True)

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
            Path(self.location) / INFO_NM,
        )

    def load_info(self):
        """Load the full settings from disk."""
        sfile = Path(self.location) / INFO_NM

        if not sfile.is_file():
            raise XYZError(f"Settings can't be found at {sfile}.")
        else:
            return read_from_disk(sfile)

    def load_batch(self, batch_number):
        """Load a specific batch from disk."""
        return read_from_disk(
            Path(self.location) / "batches" / BTCH_NM.format(batch_number)
        )

    def load_result(self, batch_number):
        """Load a specific result from disk."""
        return read_from_disk(
            Path(self.location) / "results" / RSLT_NM.format(batch_number)
        )

    def save_result(self, batch_number, result):
        """Save a specific result to disk."""
        write_to_disk(
            result,
            Path(self.location) / "results" / RSLT_NM.format(batch_number),
        )

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
        write_to_disk(to_pickle(self._fn), Path(self.location) / FNCT_NM)

    def load_function(self):
        """Load the saved function from disk, and try to re-insert it back into
        Harvester or Runner if present.
        """
        self._fn = from_pickle(read_from_disk(Path(self.location) / FNCT_NM))

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
        return (Path(self.location) / INFO_NM).exists()

    def calc_progress(self):
        """Calculate how much progressed has been made in growing the batches."""
        if self.is_prepared():
            self._sync_info_from_disk()
            self._num_sown_batches = len(
                list(
                    (Path(self.location) / "batches").glob(BTCH_NM.format("*"))
                )
            )
            self._num_results = len(
                list(
                    (Path(self.location) / "results").glob(RSLT_NM.format("*"))
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

    def completed_results(self) -> tuple[int, ...]:
        """Return tuple of batches which have been grown already."""
        self.calc_progress()

        def result_exists(x):
            return (
                Path(self.location) / "results" / RSLT_NM.format(x)
            ).is_file()

        return tuple(filter(result_exists, range(1, self.num_batches + 1)))

    def missing_results(self) -> tuple[int, ...]:
        """Return tuple of batches which haven't been grown yet."""
        self.calc_progress()

        def no_result_exists(x):
            return not (
                Path(self.location) / "results" / RSLT_NM.format(x)
            ).is_file()

        return tuple(filter(no_result_exists, range(1, self.num_batches + 1)))

    def delete_all(self):
        """Delete the crop directory and all its contents, and reset
        any loaded information on this Crop object.
        """
        crop_path = Path(self.location)
        if crop_path.exists():
            shutil.rmtree(crop_path)

        # reset in-memory state so the object reflects a fresh crop
        self.batchsize = None
        self.num_batches = None
        self._batch_remainder = None
        self._all_nan_result = None
        self._num_sown_batches = -1
        self._num_results = -1

    def handle_existing(self, action="ask", msg=None, e=None, overwrite=False):
        """Handle an already prepared crop.

        Parameters
        ----------
        action : {'ask', 'reap', 'delete', 'skip', 'raise'}
            What to do with the existing crop. If ``'ask'`` (default),
            interactively prompt the user. Otherwise execute the
            specified action directly.
        msg : str, optional
            Message to display when prompting.
        e : Exception, optional
            Exception to re-raise if action is ``'raise'``.
        overwrite : bool, optional
            Whether to overwrite existing data when reaping.
        """
        valid = {"ask", "reap", "delete", "skip", "raise"}
        if action not in valid:
            raise ValueError(
                f"Invalid action {action!r}, expected one of {valid}."
            )

        if msg is None:
            msg = f"Crop {self} has already been prepared."

        if action == "ask":
            question = (
                f"{msg}\n"
                f"Try and reap existing {self.num_results}?\n"
                "y (yes)/ n (no) / d (delete) / r (raise)."
            )
            while True:
                answer = input(question).lower()
                if answer in {"y", "yes"}:
                    action = "reap"
                    break
                elif answer in {"n", "no"}:
                    action = "skip"
                    break
                elif answer in {"d", "delete"}:
                    action = "delete"
                    break
                elif answer in {"r", "raise"}:
                    action = "raise"
                    break
                else:
                    print(f"Didn't understand answer {answer}")

        if action == "reap":
            self.reap(
                allow_incomplete=True,
                overwrite=overwrite,
                clean_up=True,
            )
        elif action == "delete":
            self.delete_all()
        elif action == "raise":
            if e is None:
                e = XYZError(f"Existing crop {self}, aborting.")
            raise e
        # action == "skip": do nothing

    @property
    def all_nan_result(self):
        """Get a stand-in result for cases which are missing still."""
        if self._all_nan_result is None:
            result_files = list(
                (Path(self.location) / "results").glob(RSLT_NM.format("*"))
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
        if not Path(self.location).exists():
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
        msg = "<Crop(name='{}', batchsize={}, num_batches={})>"
        return msg.format(self.name, self.batchsize, self.num_batches)

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
        desc="Sow",
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
            How much information to show when sowing. 0: no output, 1:
            progress bar, 2: progress bar with each setting being sown.
        desc : str, optional
            Description to show in the progress bar when sowing.
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
                desc=desc,
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
            How much information to show when sowing. 0: no output, 1:
            progress bar, 2: progress bar with each setting being sown.
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

    def grow_subprocess(
        self,
        batch_ids=None,
        num_workers=1,
        num_threads=1,
        gpus=None,
        affinities=None,
        raise_errors=False,
        log=False,
        min_wait=1e-6,
        max_wait=1e-1,
        verbosity=1,
        verbosity_grow=0,
        desc="Grow",
    ):
        """Grow particular or missing batches using a single fresh subprocess
        per batch. This has a higher overhead for starting each process, but is
        more robust memory wise, and allows controlling the number of threads
        used, CPU affinity and GPU assignment.

        Parameters
        ----------
        batch_ids : int or sequence of int, optional
            Which batch numbers to grow, defaults to all missing.
        num_workers : int, optional
            The maximum number of concurrent subprocesses.
        num_threads : int, optional
            The number of threads per subprocess.
        gpus : int, str, or sequence of int, optional
            GPU device IDs to assign to subprocesses via
            ``CUDA_VISIBLE_DEVICES``. Each subprocess gets a single GPU from
            this pool; the pool also limits concurrency to the number of GPUs
            provided. You can oversubscribe GPUs by repeating device IDs, e.g.
            ``0,0,1,1`` to allow 2 subprocesses to share each GPU.
        affinities : int, str, or sequence of int, optional
            CPU core IDs to pin subprocesses to via ``taskset``.
            Also limits concurrency to the number of affinities.
        raise_errors : bool, optional
            Whether to raise errors encountered during growing.
        log : bool, optional
            Whether to save subprocess stdout and stderr to log files in the
            crop directory under ``logs/batch-{batch_id}.log``. Default is
            False, which discards stdout and only prints stderr on error.
        min_wait : float, optional
            Minimum polling interval in seconds.
        max_wait : float, optional
            Maximum polling interval in seconds.
        verbosity : int, optional
            How much information to show when growing. 0: no output, 1:
            progress bar, 2: progress bar with each setting being grown.
        verbosity_grow : int, optional
            Verbosity within each batch grow.
        desc : str, optional
            Description to show in the progress bar when sowing.
        """
        from subprocess import DEVNULL, PIPE, Popen

        if batch_ids is None:
            batch_ids = self.missing_results()
        elif isinstance(batch_ids, int):
            batch_ids = (batch_ids,)

        if log:
            from pathlib import Path

            log_dir = Path(self.location) / "logs"
            log_dir.mkdir(exist_ok=True)

        # queue is reversed so that we can pop from right
        queue = list(reversed(batch_ids))
        # map each batch id to the process
        processing = {}
        # track open log file handles
        log_files = {}

        if verbosity:
            pbar = progbar(total=len(queue), desc=desc)
        else:
            pbar = None

        pargs = [
            sys.executable,
            "-m",
            "xyzpy.gen.xyzpy_grow_cli",
            self.name,
            "--parent-dir",
            self.parent_dir,
            "--num-threads",
            str(num_threads),
            "--verbosity-grow",
            str(verbosity_grow),
        ]
        if raise_errors:
            pargs.append("--raise-errors")

        pools = []
        for raw, acquire_fn in (
            (affinities, _acquire_affinity),
            (gpus, _acquire_gpu),
        ):
            pool = _ResourcePool.from_raw(raw, acquire_fn)
            if pool is not None:
                pools.append(pool)

        try:
            while queue or processing:
                # still work to do!
                while (
                    # there are batches still
                    bool(queue)
                    and
                    # and there are free workers
                    (len(processing) < num_workers)
                    and
                    # and all resource pools have availability
                    all(pool.available() for pool in pools)
                ):
                    # can submit more work!
                    batch_id = queue.pop()
                    these_pargs = []
                    env = os.environ.copy() if pools else None

                    for pool in pools:
                        pool.acquire(batch_id, these_pargs, env)

                    these_pargs.extend(pargs)
                    these_pargs.append("--batch-id")
                    these_pargs.append(str(batch_id))

                    if log:
                        log_f = open(log_dir / f"batch-{batch_id}.log", "w")
                        log_files[batch_id] = log_f
                        stdout = log_f
                        stderr = log_f
                    else:
                        stdout = DEVNULL
                        stderr = PIPE

                    processing[batch_id] = Popen(
                        these_pargs,
                        stdout=stdout,
                        stderr=stderr,
                        text=True,
                        env=env,
                    )

                all_running = True
                # reset wait time
                dt = min_wait

                while processing and all_running:
                    # check for finished work
                    for batch_id in tuple(processing):
                        p = processing[batch_id]
                        retcode = p.poll()
                        if retcode is not None:
                            # batch finished!
                            del processing[batch_id]
                            all_running = False

                            for pool in pools:
                                pool.release(batch_id)

                            if log:
                                log_files.pop(batch_id).close()
                                if retcode != 0:
                                    # flag to user the batch log location
                                    log_path = (
                                        log_dir / f"batch-{batch_id}.log"
                                    )
                                    print(
                                        f"batch {batch_id} failed "
                                        f"(retcode={retcode}), see {log_path}"
                                    )
                            else:
                                # drain stderr to avoid resource leaks
                                _, stderr = p.communicate()
                                if retcode != 0:
                                    print(retcode, stderr)

                            if pbar is not None:
                                pbar.update()

                    if all_running:
                        # exponential backoff
                        time.sleep(dt)
                        dt = min(1.2 * dt, max_wait)

        except KeyboardInterrupt:
            # kill all processes
            for p in processing.values():
                p.kill()
            for f in log_files.values():
                f.close()
            raise
        finally:
            if pbar is not None:
                pbar.close()

    def grow(
        self,
        batch_ids=None,
        raise_errors=False,
        subprocess=False,
        debugging=False,
        verbosity=1,
        verbosity_grow=0,
        log=False,
        desc="Grow",
        **combo_runner_opts,
    ):
        """Grow specific batch numbers using this process.

        Parameters
        ----------
        batch_ids : int or sequence of ints, optional
            Which batch numbers to grow, by default all missing results.
        raise_errors : bool, optional
            Whether to raise errors if they occur during growing.
        subprocess : bool, optional
            Whether to grow each batch in a fresh subprocess. This adds about
            1 second of overhead per batch, but allows the number of threads,
            cpu affinity and gpu assignment to be controlled. See
            `grow_subprocess` for more details.
        debugging : bool, optional
            Whether to set the logging level to debug.
        verbosity : int, optional
            How much information to show when growing. 0: no output, 1:
            progress bar, 2: progress bar with each setting being grown.
        verbosity_grow : int, optional
            How much information to show when growing each batch.
        log : bool, optional
            Whether to save subprocess output to log files. Only used
            when ``subprocess=True``.
        desc : str, optional
            Description to show in the progress bar when growing.
        **combo_runner_opts
            Additional options to pass to the `combo_runner_core` function.
            Only if `subprocess`` is False.
        """
        if batch_ids is None:
            batch_ids = self.missing_results()
        elif isinstance(batch_ids, int):
            batch_ids = (batch_ids,)

        if subprocess:
            self.grow_subprocess(
                batch_ids=batch_ids,
                raise_errors=raise_errors,
                verbosity=verbosity,
                verbosity_grow=verbosity_grow,
                log=log,
                **combo_runner_opts,
            )
        else:
            combo_runner_core(
                grow,
                combos=(("batch_number", batch_ids),),
                constants={
                    "verbosity": verbosity_grow,
                    "crop": (self.name, self.location),
                    "raise_errors": raise_errors,
                    "debugging": debugging,
                },
                verbosity=verbosity,
                desc=desc,
                **combo_runner_opts,
            )

    def grow_missing(self, **combo_runner_opts):
        """Grow any missing results using this process."""
        self.grow(batch_ids=None, **combo_runner_opts)

    def reap_combos(
        self,
        wait=False,
        clean_up=None,
        allow_incomplete=False,
        verbosity=1,
        desc="Reap",
    ):
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
        verbosity : int, optional
            How much information to show when reaping. 0: no output, 1:
            progress bar, 2: progress bar with each setting being reaped.
        desc : str, optional
            Description to show in the progress bar when reaping.

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

        reaper = Reaper(
            self,
            num_batches=settings["num_batches"],
            wait=wait,
            default_result=default_result,
        )

        results = combo_runner_core(
            fn=reaper,
            combos=settings["combos"],
            cases=settings["cases"],
            constants={},
            shuffle=settings.get("shuffle", False),
            verbosity=verbosity,
            desc=desc,
        )

        reaper.check_finished()

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
        verbosity=1,
        desc="Reap",
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
        verbosity : int, optional
            How much information to show when reaping. 0: no output, 1:
            progress bar, 2: progress bar with each setting being reaped.
        desc : str, optional
            Description to show in the progress bar when reaping.

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

        reaper = Reaper(
            self,
            num_batches=settings["num_batches"],
            wait=wait,
            default_result=default_result,
        )

        # move constants into attrs, so as not to pass them to the Reaper
        #   when if fact they were meant for the original function.
        data = combo_runner_to_ds(
            fn=reaper,
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
            verbosity=verbosity,
            desc=desc,
        )

        reaper.check_finished()

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
        verbosity=1,
        desc="Reap",
        **kwargs,
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
            verbosity=verbosity,
            desc=desc,
            **kwargs,
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
        verbosity=1,
        desc="Reap",
        **kwargs,
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
            verbosity=verbosity,
            desc=desc,
            **kwargs,
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
        verbosity=1,
        desc="Reap",
        **kwargs,
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
            verbosity=verbosity,
            desc=desc,
            **kwargs,
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
        verbosity=1,
        desc="Reap",
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
        verbosity : int, optional
            How much information to show when reaping. 0: no output, 1:
            progress bar, 2: progress bar with each setting being reaped.
        desc : str, optional
            Description to show in the progress bar when reaping.

        Returns
        -------
        nested tuple or xarray.Dataset
        """
        opts = dict(
            clean_up=clean_up,
            wait=wait,
            allow_incomplete=allow_incomplete,
            verbosity=verbosity,
            desc=desc,
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
            str(Path(self.location) / "results" / RSLT_NM.format("*"))
        )

        bad_ids = []

        for result_file in result_files:
            # load corresponding batch file to check length.
            result_name = Path(result_file).name
            result_num = result_name.strip("xyz-result-").strip(".jbdmp")
            batch = self.load_batch(result_num)

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
                    Path(result_file).unlink()

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
    import re

    folders = [
        path.name for path in Path(directory).iterdir() if path.is_dir()
    ]
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
            Path(self.crop.location)
            / "batches"
            / BTCH_NM.format(self._batch_counter),
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
    raise_errors=True,
):
    """Automatically process a batch of cases into results. Should be run in an
    ".xyz-{fn_name}" folder, or `crop` should be specified.

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
    raise_errors : bool, optional
        Whether to raise errors that occur during the computation. If growing
        many batches in parallel, it can be useful to set this to False so
        a single error doesn't crash the whole process.
    """
    if debugging:
        import logging

        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)

    # first we load the function and batch of cases to run

    if crop is None:
        current_folder = Path.cwd().name
        if current_folder[:5] != ".xyz-":
            raise XYZError(
                "`grow` should be run in a '{crop_parent}/.xyz-{crop_name}' "
                "folder, else `crop_parent` and `crop_name` (or `fn`) should "
                "be specified."
            )
        crop_name = current_folder[5:]
        crop_location = str(Path.cwd())
    elif isinstance(crop, tuple):
        # only need location, helpful to avoid pickling issues
        crop_name, crop_location = crop
    else:
        crop_name = crop.name
        crop_location = crop.location

    crop_root = Path(crop_location)
    fn_file = crop_root / FNCT_NM
    cases_file = crop_root / "batches" / BTCH_NM.format(batch_number)
    results_file = crop_root / "results" / RSLT_NM.format(batch_number)

    # load function
    if fn is None:
        fn = from_pickle(read_from_disk(fn_file))

    # load cases to evaluate
    cases = read_from_disk(cases_file)

    if len(cases) == 0:
        raise ValueError(
            "Something has gone wrong with the loading of batch {} ".format(
                BTCH_NM.format(batch_number)
            )
            + "for the crop at {}.".format(crop.location)
        )
    if verbosity >= 1:
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

    try:
        # get the actual results!
        results = []
        for i, r in enumerate(results_it):
            if verbosity >= 2:
                results_it.set_description(f"{cases[i]}")
            results.append(r)

        if rank == 0:
            # only save to results file if the main worker
            write_to_disk(tuple(results), results_file)

        if verbosity >= 1:
            print(f"xyzpy: success - batch {batch_number} completed.")

    except Exception as e:
        # possibly catch errors, so they don't crash the whole process
        print(f"xyzpy: error - batch {batch_number} failed with error: {e}")
        if raise_errors:
            raise e

        # allow ctrl-c to stop the process
        if isinstance(e, KeyboardInterrupt):
            raise e

        print(f"xyzpy: ... continuing since raise_errors={raise_errors}.")

    finally:
        if verbosity:
            results_it.close()


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
            Path(self.crop.location) / "results" / RSLT_NM.format(i + 1)
            for i in range(num_batches)
        )

        def _load(x):
            x = Path(x)
            use_default = (
                (default_result is not None)
                and (not wait)
                and (not x.is_file())
            )

            # actual result doesn't exist yet - use the default if specified
            if use_default:
                i = int(re.findall(RSLT_NM.format(r"(\d+)"), str(x))[0])
                size = len(crop.load_batch(i))
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
            x = Path(x)
            while not x.exists():
                time.sleep(0.2)

            if x.is_file():
                return _load(x)
            else:
                raise ValueError(f"{x} is not a file.")

        self.results = itertools.chain.from_iterable(
            map(wait_to_load if wait else _load, files)
        )

    def __call__(self, **kwargs):
        return next(self.results)

    def check_finished(self):
        # Check everything gone acccording to plan
        remaining_results = tuple(self.results)
        if remaining_results:
            raise XYZError("Not all results reaped!")


# --------------------------------------------------------------------------- #
#                     Automatic Batch Submission Scripts                      #
# --------------------------------------------------------------------------- #

_SGE_HEADER = (
    "#!/bin/bash -l\n"
    "#$ -S /bin/bash\n"
    "#$ -N {name}\n"
    "#$ -l h_rt={hours}:{minutes}:{seconds},mem={gigabytes}G\n"
    "#$ -l tmpfs={temp_gigabytes}G\n"
    "mkdir -p {output_directory}\n"
    "#$ -wd {output_directory}\n"
    "#$ -pe {pe} {num_procs}\n"
    "{header_options}\n"
)
_SGE_ARRAY_HEADER = "#$ -t {run_start}-{run_stop}\n"

_PBS_HEADER = (
    "#!/bin/bash -l\n"
    "#PBS -N {name}\n"
    "#PBS -lselect={num_nodes}:ncpus={num_procs}:mem={gigabytes}gb\n"
    "#PBS -lwalltime={hours:02}:{minutes:02}:{seconds:02}\n"
    "{header_options}\n"
)
_PBS_ARRAY_HEADER = "#PBS -J {run_start}-{run_stop}\n"

_SLURM_HEADER = (
    "#!/bin/bash -l\n"
    "#SBATCH --job-name={name}\n"
    "#SBATCH --time={hours:02}:{minutes:02}:{seconds:02}\n"
    "{header_options}\n"
)
_SLURM_ARRAY_HEADER = "#SBATCH --array={run_start}-{run_stop}\n"

# _BASE = (
#     "echo 'XYZPY script starting...'\n"
#     "cd {working_directory}\n"
#     "{shell_setup}\n"
#     "xyzpy-grow {name}"
#     " --parent-dir {parent_dir}"
#     " --batch-ids $SLURM_ARRAY_TASK_ID"
#     " --num-threads {num_threads}"
#     " --num_workers {num_workers}"
#     " --subprocess {subprocess}"
#     "\n"
# )

_BASE = (
    "echo 'XYZPY script starting...'\n"
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
    "    print('Growing:', repr(crop))\n"
    "    grow_kwargs = dict(\n"
    "        num_workers={num_workers},\n"
    "        subprocess={subprocess},\n"
    "        debugging={debugging},\n"
    "        verbosity_grow=2,\n"
    "    )\n"
)

_CLUSTER_SGE_GROW_ALL_SCRIPT = "    crop.grow($SGE_TASK_ID, **grow_kwargs)\n"

_CLUSTER_PBS_GROW_ALL_SCRIPT = (
    "    crop.grow($PBS_ARRAY_INDEX, **grow_kwargs)\n"
)

_CLUSTER_SLURM_GROW_ALL_SCRIPT = (
    "    crop.grow($SLURM_ARRAY_TASK_ID, **grow_kwargs)\n"
)

_CLUSTER_SGE_GROW_PARTIAL_SCRIPT = (
    "    batch_ids = {batch_ids}]\n"
    "    crop.grow(batch_ids[$SGE_TASK_ID - 1], **grow_kwargs)\n"
)

_CLUSTER_PBS_GROW_PARTIAL_SCRIPT = (
    "    batch_ids = {batch_ids}\n"
    "    crop.grow(batch_ids[$PBS_ARRAY_INDEX - 1], **grow_kwargs)\n"
)

_CLUSTER_SLURM_GROW_PARTIAL_SCRIPT = (
    "    batch_ids = {batch_ids}\n"
    "    crop.grow(batch_ids[$SLURM_ARRAY_TASK_ID - 1], **grow_kwargs)\n"
)

_BASE_CLUSTER_GROW_SINGLE = (
    "    grow_kwargs['verbosity_grow'] = 0\n"
    "    batch_ids = {batch_ids}\n"
    "    crop.grow(batch_ids, **grow_kwargs)\n"
)

_BASE_CLUSTER_SCRIPT_END = (
    "EOM\n{launcher} -c \"$SCRIPT\"\necho 'XYZPY script finished'\n"
)


def gen_cluster_script(
    crop,
    scheduler,
    batch_ids=None,
    *,
    mode="array",
    num_procs=None,
    num_threads=None,
    num_nodes=None,
    num_workers=None,
    subprocess=False,
    mem=None,
    mem_per_cpu=None,
    gigabytes=None,
    time=None,
    hours=None,
    minutes=None,
    seconds=None,
    conda_env=True,
    launcher=None,
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
    subprocess : bool, optional
        Whether to use a fresh subprocess for each batch, default: False.
    num_nodes : int, optional
        How many nodes to request, default: 1.
    conda_env : bool or str, optional
        Whether to activate a conda environment before running the script.
        If ``True``, the environment will be the same as the one used to
        launch the script. If a string, the environment will be the one
        specified by the string.
    launcher : str, optional
        How to launch the script, default: the current Python interpreter. But
        could for example be ``'mpiexec python'`` for a MPI program.
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

    # parse the number of threads
    if num_threads is None:
        if num_workers is None:
            # default to 1 thread per core for no workers
            num_threads = num_procs
        else:
            # default to 1 thread per worker
            num_threads = round(num_procs / num_workers)

    # parse the time requirement
    if hours is minutes is seconds is None:
        if time is not None:
            if isinstance(time, (int, float)):
                hours = time
                minutes, seconds = 0, 0
            elif isinstance(time, str):
                hours, minutes, seconds = time.split(":")
        else:
            hours, minutes, seconds = 1, 0, 0
    else:
        if time is not None:
            raise ValueError(
                "Cannot specify both time and hours, minutes, seconds."
            )
        hours = 0 if hours is None else int(hours)
        minutes = 0 if minutes is None else int(minutes)
        seconds = 0 if seconds is None else int(seconds)

    if scheduler == "slurm":
        # only supply specified header options
        # TODO: same with PBS and SGE

        if num_nodes is not None:
            kwargs["nodes"] = num_nodes
        if num_procs is not None:
            kwargs["cpus-per-task"] = num_procs

        if gigabytes is not None:
            if mem is not None:
                raise ValueError("Cannot specify both gigabytes and mem.")
            mem = gigabytes

        if mem is not None:
            if isinstance(mem, int):
                mem = f"{mem}G"
            kwargs["mem"] = mem

        if mem_per_cpu is not None:
            if isinstance(mem_per_cpu, int):
                mem_per_cpu = f"{mem_per_cpu}G"
            kwargs["mem-per-cpu"] = mem_per_cpu

    else:
        # pbs, sge
        # parse memory to gigabytes
        if (gigabytes is not None) and (mem is not None):
            raise ValueError("Cannot specify both gigabytes and mem.")

        if mem is not None:
            # take gigabytes from mem
            gigabytes = int(mem)

    if output_directory is None:
        output_directory = str(Path.home() / "Scratch" / "output")

    if launcher is None:
        launcher = sys.executable

    if conda_env is True:
        # automatically set conda environment to be the
        # same as the one that's running this function
        conda_env = os.environ.get("CONDA_DEFAULT_ENV", False)
        if conda_env:
            # but only if we are in a conda environment
            if (
                ("conda activate" in shell_setup)
                or ("mamba activate" in shell_setup)
                or ("micromamba activate" in shell_setup)
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
            header_options = "\n".join(
                [
                    f"#SBATCH --{k}"
                    if (v is None or v is True)
                    else f"#SBATCH --{k}={v}"
                    for k, v in kwargs.items()
                ]
            )
        elif scheduler == "pbs":
            header_options = "\n".join(
                [
                    f"#PBS -l {k}"
                    if (v is None or v is True)
                    else f"#PBS -l {k}={v}"
                    for k, v in kwargs.items()
                ]
            )
        elif scheduler == "sge":
            header_options = "\n".join(
                [
                    f"#$ -l {k}"
                    if (v is None or v is True)
                    else f"#$ -l {k}={v}"
                    for k, v in kwargs.items()
                ]
            )
    else:
        header_options = ""

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
    full_parent_dir = str(Path(crop.parent_dir).expanduser().resolve())

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
        "subprocess": subprocess,
        "launcher": launcher,
        "setup": setup,
        "shell_setup": shell_setup,
        "pe": "mpi" if mpi else "smp",
        "temp_gigabytes": temp_gigabytes,
        "output_directory": output_directory,
        "working_directory": full_parent_dir,
        "header_options": header_options,
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
            opts["batch_ids"] = "None"
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
    num_nodes=1,
    num_procs=1,
    num_threads=None,
    num_workers=None,
    subprocess=False,
    conda_env=True,
    launcher=None,
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
    num_nodes : int, optional
        How many nodes to request, default: 1.
    num_procs : int, optional
        How many processes to request (threaded cores or MPI), default: 1.
    num_threads : int, optional
        How many threads to use per process. Will be computed automatically
        based on ``num_procs`` and ``num_workers`` if not specified.
    num_workers : int, optional
        How many workers to use for parallel growing, default is sequential. If
        specified, then generally ``num_workers * num_threads == num_procs``.
    subprocess : bool, optional
        Whether to use a fresh subprocess for each batch, default: False.
    conda_env : bool or str, optional
        Whether to activate a conda environment before running the script.
        If ``True``, the environment will be the same as the one used to
        launch the script. If a string, the environment will be the one
        specified by the string.
    launcher : str, optional
        How to launch the script, default: the current Python interpreter. But
        could for example be ``'mpiexec python'`` for a MPI program.
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
    from subprocess import run

    if crop.is_ready_to_reap():
        print("Crop ready to reap: nothing to submit.")
        return

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
        subprocess=subprocess,
        conda_env=conda_env,
        launcher=launcher,
        setup=setup,
        shell_setup=shell_setup,
        mpi=mpi,
        debugging=debugging,
        **kwargs,
    )

    script_file = Path(crop.location) / "__qsub_script__.sh"

    with open(script_file, mode="w") as f:
        f.write(script)

    if scheduler in {"sge", "pbs"}:
        result = run(["qsub", str(script_file)], capture_output=True)
    elif scheduler == "slurm":
        result = run(["sbatch", str(script_file)], capture_output=True)

    print(result.stderr.decode())
    print(result.stdout.decode())

    script_file.unlink()


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
    import subprocess

    job = str(job)

    files = list(Path(directory).glob(f"slurm-{job}_*.out"))

    for file in files:
        jobid = int(re.match(r"slurm-\d+_(\d+).out", str(file)).groups()[0])

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
        elif "error" in contents.lower():
            # check if any captilization of 'error' in file
            print("appears to have an error!", end=" ")
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
