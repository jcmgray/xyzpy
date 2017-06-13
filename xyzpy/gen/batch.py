import os
import shutil
from itertools import chain
from time import sleep

try:
    import joblib
except ImportError:  # pragma: no cover
    pass

try:
    import cloudpickle
except ImportError:  # pragma: no cover
    pass

from ..utils import _get_fn_name, prod
from .prepare import _parse_combos, _parse_constants, _parse_attrs
from .combo_runner import _combo_runner, combo_runner_to_ds


BTCH_NM = "xyz-batch-{}.jbdmp"
RSLT_NM = "xyz-result-{}.jbdmp"
FNCT_NM = "xyz-function.clpkl"
INFO_NM = "xyz-settings.jbdmp"


class XYZError(Exception):
    pass


# --------------------------------- parsing --------------------------------- #

def parse_crop_details(fn, crop_name, crop_dir):
    """Work out how to structure the sowed data.

    Parameters
    ----------
        fn : callable (optional)
            Function to infer name crop_name from, if not given.
        crop_name : str (optional)
            Specific name to give this set of runs.
        crop_dir : str (optional)
            Specific directory to put the ".xyz-{crop_name}/" folder in
            with all the cases and results.

    Returns
    -------
        crop_folder : str
            Full path to the crop-folder.
    """
    if crop_name is None:
        if fn is None:
            raise ValueError("Either `fn` or `crop_name` must be give.")
        crop_name = _get_fn_name(fn)

    crop_folder = os.path.join(
        crop_dir if crop_dir is not None else os.getcwd(),
        ".xyz-{}".format(crop_name))

    return crop_folder


class Crop(object):
    """Encapsulates all the details describing a single 'crop', that is,
    its location, name, and batch size/number.
    """

    def __init__(self, *,
                 fn=None,
                 name=None,
                 folder=None,
                 save_fn=None,
                 batchsize=None,
                 num_batches=None):
        """

        Parameters
        ----------
            fn : callable (optional)
                Target function - Crop `name` will be inferred from this if
                not given explicitly. If given, `Sower` will also default
                to saving a version of `fn` to disk for `batch.grow` to use.
            name : str (optional)
                Custom name for this set of runs - must be given if `fn`
                is not.
            folder : str (optional)
                If given, alternative directory to put the ".xyz-{name}/"
                folder in with all the cases and results.
            save_fn : bool (optional)
                Whether to save the function to disk for `batch.grow` to use.
                Will default to True if `fn` is given.
            batchsize : int (optional)
                How many cases to group into a single batch per worker.
                By default, batchsize=1. Cannot be specified if `num_batches`
                is.
            num_batches : int (optional)
                How many total batches to aim for, cannot be specified if
                `batchsize` is.
        """

        self._fn = fn
        self.name = name
        self.folder = folder
        self.save_fn = save_fn
        self.batchsize = batchsize
        self.num_batches = num_batches

        # Work out the full directory for the crop
        self.location = parse_crop_details(fn, name, folder)

        # Save function so it can be automatically loaded with all deps?
        if (fn is None) and (save_fn is True):
            raise ValueError("Must specify a function for it to be saved!")
        self.save_fn = save_fn is not False

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

    def choose_batch_settings(self, combos):
        """Work out how to divide all cases into batches, i.e. ensure
        that ``batchsize * num_batches >= num_cases``.
        """
        n = prod(len(x) for _, x in combos)

        if (self.batchsize is not None) and (self.num_batches is not None):
            raise ValueError("`batchsize` and `num_batches` cannot both be "
                             "specified.")

        # Decide based on batchsize
        elif self.num_batches is None:
            if self.batchsize is None:
                self.batchsize = 1

            if not isinstance(self.batchsize, int):
                raise TypeError("`batchsize` must be an integer.")
            if self.batchsize < 1:
                raise ValueError("`batchsize` must be >= 1.")

            self.num_batches = ((n // self.batchsize) +
                                int(n % self.batchsize != 0))

        # Decide based on num_batches:
        else:
            if not isinstance(self.num_batches, int):
                raise TypeError("`num_batches` must be an integer.")
            if self.num_batches < 1:
                raise ValueError("`num_batches` must be >= 1.")

            self.batchsize = ((n // self.num_batches) +
                              int(n % self.num_batches != 0))

    def ensure_dirs_exists(self):
        """Make sure the directory structure for this crop exists.
        """
        os.makedirs(os.path.join(self.location, "cases"), exist_ok=True)
        os.makedirs(os.path.join(self.location, "results"), exist_ok=True)

    def save_function_to_disk(self):
        """Save the base function to disk using cloudpickle
        """
        import cloudpickle

        joblib.dump(cloudpickle.dumps(self._fn),
                    os.path.join(self.location, FNCT_NM))

    def save_info(self, combos):
        """Save information about the sowed cases.
        """
        joblib.dump({
            'combos': combos,
            'batchsize': self.batchsize,
            'num_batches': self.num_batches,
        }, os.path.join(self.location, INFO_NM))

    def prepare(self, combos):
        self.ensure_dirs_exists()
        if self.save_fn:
            self.save_function_to_disk()
        self.save_info(combos)


class Sower(object):
    """Class for sowing a 'crop' of batched combos to then 'grow' (on any
    number of workers sharing the filesystem) and then reap.
    """

    def __init__(self, crop, combos):
        """
        Parameters
        ----------
            crop : xyzpy.batch.Crop instance
                Description of where and how to store the cases and results.
            combos : mapping_like
                Description of combinations from which to sow cases from.

        """
        self.combos = combos
        self.crop = crop
        self.crop.choose_batch_settings(combos)
        # Internal
        self._batch_cases = []
        self._counter = 0
        self._batch_counter = 0

    def save_batch(self):
        """Save the current batch of cases to disk using joblib.dump
         and start the next batch.
        """
        self._batch_counter += 1
        joblib.dump(
            self._batch_cases,
            os.path.join(self.crop.location,
                         "cases",
                         BTCH_NM.format(self._batch_counter)))

    # Context manager #

    def __enter__(self):
        self.crop.prepare(self.combos)
        return self

    def __call__(self, **kwargs):
        self._batch_cases.append(kwargs)
        self._counter += 1

        if self._counter == self.crop.batchsize:
            self.save_batch()
            self._batch_cases = []
            self._counter = 0

    def __exit__(self, exception_type, exception_value, traceback):
        # Make sure any overfill also saved
        if self._batch_cases:
            self.save_batch()


def combos_sow(crop, combos, constants=None, hide_progbar=False):

    combos = _parse_combos(combos)
    constants = _parse_constants(constants)

    # Sort to ensure order remains same for reaping results (don't hash kwargs)
    combos = sorted(combos, key=lambda x: x[0])

    with Sower(crop, combos) as sow_fn:
        _combo_runner(fn=sow_fn, combos=combos, constants=constants,
                      hide_progbar=hide_progbar)


def grow(batch_number, crop=None, fn=None, check_mpi=True):
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
    """
    if crop is None:
        if os.path.relpath('.', '..')[:5] != ".xyz-":
            raise XYZError("`grow` should be run in a "
                           "\"{crop_dir}/.xyz-{crop_name}\" folder, else "
                           "`crop_dir` and `crop_name` (or `fn`) should be "
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
        os.path.join(crop_location, "cases", BTCH_NM.format(batch_number)))

    # process each case
    results = tuple(fn(**kws) for kws in cases)

    # maybe want to run grow as mpiexec (i.e. `fn` itself in parallel),
    # so only save and delete on rank 0
    if check_mpi and 'OMPI_COMM_WORLD_RANK' in os.environ:  # pragma: no cover
        rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
    elif check_mpi and 'PMI_RANK' in os.environ:  # pragma: no cover
        rank = int(os.environ['PMI_RANK'])
    else:
        rank = 0

    if rank == 0:
        # save to results
        joblib.dump(results, os.path.join(crop_location,
                                          "results",
                                          RSLT_NM.format(batch_number)))

        # delete set of runs
        os.remove(os.path.join(crop_location,
                               "cases",
                               BTCH_NM.format(batch_number)))


# --------------------------------------------------------------------------- #
#                              Gathering results                              #
# --------------------------------------------------------------------------- #

class Reaper(object):
    """Class that acts as a stateful function to retrieve already sown and
    grow results.
    """

    def __init__(self, crop, num_batches):
        """Class for retrieving the batched, flat, 'grown' results.

        Parameters
        ----------
            crop : xyzpy.batch.Crop instance
                Description of where and how to store the cases and results.
        """
        self.crop = crop

        files = (os.path.join(self.crop.location, "results", RSLT_NM.format(i))
                 for i in range(1, num_batches + 1))

        def wait_to_load(x):
            while not os.path.exists(x):
                sleep(0.2)

            if os.path.isfile(x):
                return joblib.load(x)
            else:
                raise ValueError("{} is not a file.".format(x))

        self.results = chain.from_iterable(map(wait_to_load, files))

    def __enter__(self):
        return self

    def __call__(self, **kwargs):
        return next(self.results)

    def __exit__(self, exception_type, exception_value, traceback):
        if tuple(*self.results):
            raise XYZError("Not all results reaped!")
        else:
            shutil.rmtree(self.crop.location)


def combos_reap(crop):
    """Reap already sown and grow results from specified crop.

    Parameters
    ----------
        crop : xyzpy.batch.Crop instance
            Description of where and how to store the cases and results.
    """
    # Load same combinations as cases saved with
    settings = joblib.load(os.path.join(crop.location, INFO_NM))

    with Reaper(crop, num_batches=settings['num_batches']) as reap_fn:
        results = _combo_runner(fn=reap_fn,
                                combos=settings['combos'],
                                constants={})

    return results


def combos_reap_to_ds(crop,
                      var_names=None,
                      var_dims=None,
                      var_coords=None,
                      constants=None,
                      attrs=None,
                      parse=True):
    """Reap a function over sowed combinations and output to a Dataset.

    Parameters
    ----------
        crop : xyzpy.batch.Crop instance
            Description of where and how to store the cases and results.
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

    Returns
    -------
        xarray.Dataset
            Multidimensional labelled dataset contatining all the results.
    """
    # Load same combinations as cases saved with
    settings = joblib.load(os.path.join(crop.location, INFO_NM))

    constants = _parse_constants(constants)
    attrs = _parse_attrs(attrs)

    with Reaper(crop, num_batches=settings['num_batches']) as reap_fn:
        ds = combo_runner_to_ds(fn=reap_fn,
                                combos=settings['combos'],
                                var_names=var_names,
                                var_dims=var_dims,
                                var_coords=var_coords,
                                constants={},
                                resources={},
                                attrs={**attrs, **constants},
                                parse=parse)

    return ds


def combos_sow_and_reap(crop, combos, constants=None):
    """Sow combos and immediately (wait to) reap the results.

    Parameters
    ----------
        crop : xyzpy.batch.Crop instance
            Description of where and how to store the cases and results.
        combos : mapping_like
            Description of combinations from which to sow cases from.
    """
    combos_sow(crop, combos, constants=constants, hide_progbar=True)
    return combos_reap(crop)


def combos_sow_and_reap_to_ds(crop, combos, constants=None,
                              var_names=None,
                              var_dims=None,
                              var_coords=None,
                              attrs=None,
                              parse=True):
    """Sow combos and immediately (wait to) reap the results to a dataset.

    Parameters
    ----------
        crop : xyzpy.batch.Crop instance
            Description of where and how to store the cases and results.
        combos : mapping_like
            Description of combinations from which to sow cases from.

    """
    combos_sow(crop, combos, constants=constants, hide_progbar=True)
    return combos_reap_to_ds(crop,
                             var_names=var_names,
                             var_dims=var_dims,
                             var_coords=var_coords,
                             constants=constants,
                             attrs=attrs,
                             parse=parse)
