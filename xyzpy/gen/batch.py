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
        field_folder : str
            Full path to the field-folder.
    """
    if crop_name is None:
        if fn is None:
            raise ValueError("Either `fn` or `crop_name` must be give.")
        crop_name = _get_fn_name(fn)

    field_folder = os.path.join(
        crop_dir if crop_dir is not None else os.getcwd(),
        ".xyz-{}".format(crop_name))

    return field_folder


class Crop(object):
    """Encapsulates all the details describing a single 'crop', that is,
    its location, name, and batch size/number.
    """

    def __init__(self, *,
                 fn=None,
                 crop_name=None,
                 crop_dir=None,
                 save_fn=None,
                 batchsize=None,
                 num_batches=None):

        self.fn = fn
        self.crop_name = crop_name
        self.crop_dir = crop_dir
        self.save_fn = save_fn
        self.batchsize = batchsize
        self.num_batches = num_batches

        self.field_folder = parse_crop_details(fn, crop_name, crop_dir)


class Sower(object):
    """Class for sowing a 'field' of batched combos to then 'grow' (on any
    number of workers sharing the filesystem) and then reap.
    """

    def __init__(self, *,
                 fn=None,
                 crop_name=None,
                 crop_dir=None,
                 save_fn=None,
                 batchsize=None,
                 num_batches=None,
                 combos=None):
        """
        Parameters
        ----------
            fn : callable (optional)
                Target function - `crop_name` will be inferred from this if
                not given explicitly. If given, `Sower` will also default
                to saving a version of `fn` to disk for `batch.grow` to use.
            crop_name : str (optional)
                Custom name for this set of runs - must be given if `fn`
                is not.
            crop_dir : str (optional)
                If given, alternative directory to put the ".xyz-{crop_name}/"
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
        self.fn = fn
        self.field_folder = parse_crop_details(fn, crop_name, crop_dir)
        self.combos = combos
        self.choose_batch_settings(batchsize, num_batches)

        # Save function so it can be automatically loaded with all deps?
        if (fn is None) and (save_fn is True):
            raise ValueError("Must specify a function for it to be saved!")
        self.save_fn = save_fn is not False

        # Internal
        self._batch_cases = []
        self._counter = 0
        self._batch_counter = 0

    def choose_batch_settings(self, batchsize, num_batches):
        """Work out how to divide all cases into batches, i.e. ensure
        that ``batchsize * num_batches >= num_cases``.
        """
        n = prod(len(x) for _, x in self.combos)

        if (batchsize is not None) and (num_batches is not None):
            raise ValueError("`batchsize` and `num_batches` cannot both be "
                             "specified.")

        # Decide based on batchsize
        elif num_batches is None:
            batchsize = batchsize if batchsize is not None else 1

            if not isinstance(batchsize, int):
                raise TypeError("`batchsize` must be an integer.")
            if batchsize < 1:
                raise ValueError("`batchsize` must be >= 1.")

            num_batches = (n // batchsize) + int(n % batchsize != 0)

        # Decide based on num_batches:
        else:
            if not isinstance(num_batches, int):
                raise TypeError("`num_batches` must be an integer.")
            if num_batches < 1:
                raise ValueError("`num_batches` must be >= 1.")

            batchsize = (n // num_batches) + int(n % num_batches != 0)

        self.batchsize, self.num_batches = batchsize, num_batches

    def ensure_dirs_exists(self):
        """Make sure the directory structure for this field exists.
        """
        os.makedirs(os.path.join(self.field_folder, "cases"), exist_ok=True)
        os.makedirs(os.path.join(self.field_folder, "results"), exist_ok=True)

    def save_info(self):
        """Save information about the sowed cases.
        """
        joblib.dump({
            'combos': self.combos,
            'batchsize': self.batchsize,
            'num_batches': self.num_batches,
        }, os.path.join(self.field_folder, INFO_NM))

    def save_function_to_disk(self):
        """Save the base function to disk using cloudpickle
        """
        import cloudpickle

        joblib.dump(cloudpickle.dumps(self.fn),
                    os.path.join(self.field_folder, FNCT_NM))

    def save_batch(self):
        """Save the current batch of cases to disk using joblib.dump
         and start the next batch.
        """
        self._batch_counter += 1
        joblib.dump(self._batch_cases,
                    os.path.join(self.field_folder, "cases",
                                 BTCH_NM.format(self._batch_counter)))

    # Context manager

    def __enter__(self):
        self.ensure_dirs_exists()
        self.save_info()
        if self.save_fn:
            self.save_function_to_disk()
        return self

    def __call__(self, **kwargs):
        self._batch_cases.append(kwargs)
        self._counter += 1

        if self._counter == self.batchsize:
            self.save_batch()
            self._batch_cases = []
            self._counter = 0

    def __exit__(self, exception_type, exception_value, traceback):
        # Make sure any overfill also saved
        if self._batch_cases:
            self.save_batch()


def combos_sow(combos, *, constants=None,
               fn=None,
               crop_name=None,
               crop_dir=None,
               save_fn=None,
               batchsize=None,
               num_batches=None,
               hide_progbar=False):

    combos = _parse_combos(combos)
    constants = _parse_constants(constants)

    # Sort to ensure order remains same for reaping results (don't hash kwargs)
    combos = sorted(combos, key=lambda x: x[0])

    sow_opts = {
        'fn': fn,
        'crop_name': crop_name,
        'crop_dir': crop_dir,
        'batchsize': batchsize,
        'num_batches': num_batches,
        'save_fn': save_fn,
        'combos': combos,
    }

    with Sower(**sow_opts) as s:
        _combo_runner(fn=s, combos=combos, constants=constants,
                      hide_progbar=hide_progbar)


def grow(batch_number, fn=None, crop_name=None, crop_dir=None,
         check_mpi=True):
    """Automatically process a batch of cases into results. Should be run in an
    ".xyz-{fn_name}" folder.

    Parameters
    ----------
        batch_number : int
            Which batch to 'grow' into a set of results.
        fn : callable, optional
            If specified, the function used to generate the results, otherwise
            the function will be loaded from disk.
        crop_name : str, optional
            Name of the set of results, will be taken as the name of ``fn``
            if not given.
        crop_dir : str, optional
            Directory within which the 'field' is, taken as current working
            directory if not given.
        check_mpi : bool, optional
            Whether to check if the process is rank 0 and only save results if
            so - allows mpi functions to be simply used. Defaults to true,
            this should only be turned off if e.g. a pool of workers is being
            used to run different ``grow`` instances.
    """
    if fn is crop_name is crop_dir is None:
        field_folder = os.getcwd()
        if os.path.relpath('.', '..')[:5] != ".xyz-":
            raise XYZError("`grow` should be run in a "
                           "\"{crop_dir}/.xyz-{crop_name}\" folder, else "
                           "`crop_dir` and `crop_name` (or `fn`) should be "
                           "specified.")
    else:
        field_folder = parse_crop_details(fn, crop_name, crop_dir)

    # load function
    if fn is None:
        fn = cloudpickle.loads(
            joblib.load(os.path.join(field_folder, FNCT_NM)))

    # load cases to evaluate
    cases = joblib.load(
        os.path.join(field_folder, "cases", BTCH_NM.format(batch_number)))

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
        joblib.dump(results, os.path.join(field_folder, "results",
                                          RSLT_NM.format(batch_number)))

        # delete set of runs
        os.remove(os.path.join(field_folder, "cases",
                               BTCH_NM.format(batch_number)))


# --------------------------------------------------------------------------- #
#                              Gathering results                              #
# --------------------------------------------------------------------------- #

class Reaper(object):
    """
    """

    def __init__(self, fn=None, crop_name=None, crop_dir=None,
                 num_batches=None):
        """Class for retrieving the batched, flat, 'grown' results.

        Parameters
        ----------
            fn : callable (optional)
                Target function - `crop_name` will be inferred from this if
                not given explicitly.
            crop_name : str (optional)
                Custom name for the set of batches - must be given if `fn`
                is not.
            crop_dir : str (optional)
                If given, alternative to current working directory for results
                to be reaped from.
        """
        self.field_folder = parse_crop_details(fn, crop_name, crop_dir)

        files = (os.path.join(self.field_folder,
                              "results",
                              RSLT_NM.format(i))
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
            shutil.rmtree(self.field_folder)


def combos_reap(fn=None, crop_name=None, crop_dir=None):
    """
    """
    field_folder = parse_crop_details(fn, crop_name, crop_dir)
    # Load same combinations as cases saved with
    settings = joblib.load(os.path.join(field_folder, INFO_NM))

    with Reaper(fn, crop_name, crop_dir,
                num_batches=settings['num_batches']) as r:
        results = _combo_runner(fn=r, combos=settings['combos'], constants={})

    return results


def combos_reap_to_ds(fn=None, crop_name=None, crop_dir=None, *,
                      var_names=None,
                      var_dims=None,
                      var_coords=None,
                      constants=None,
                      attrs=None,
                      parse=True):
    """Reap a function over sowed combinations and output to a Dataset.

    Parameters
    ----------
        fn : callable (optional)
            Target function - `crop_name` will be inferred from this if
            not given explicitly.
        crop_name : str (optional)
            Custom name for the set of batches - must be given if `fn`
            is not.
        crop_dir : str (optional)
            If given, alternative to current working directory for results
            to be reaped from.
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
        **combo_runner_settings: dict-like, optional
            Arguments supplied to `combo_runner`.

    Returns
    -------
        xarray.Dataset
            Multidimensional labelled dataset contatining all the results.
    """
    field_folder = parse_crop_details(fn, crop_name, crop_dir)
    # Load same combinations as cases saved with
    settings = joblib.load(os.path.join(field_folder, INFO_NM))

    constants = _parse_constants(constants)
    attrs = _parse_attrs(attrs)

    with Reaper(fn, crop_name, crop_dir,
                num_batches=settings['num_batches']) as r:
        ds = combo_runner_to_ds(fn=r,
                                combos=settings['combos'],
                                var_names=var_names,
                                var_dims=var_dims,
                                var_coords=var_coords,
                                constants={},
                                resources={},
                                attrs={**attrs, **constants},
                                parse=parse)

    return ds


def combos_sow_and_reap(combos, *, constants=None,
                        fn=None,
                        crop_name=None,
                        crop_dir=None,
                        save_fn=None,
                        batchsize=None,
                        num_batches=None):
    """Sow combos and immediately (wait to) reap the results.
    """

    combos_sow(combos,
               constants=constants,
               fn=fn,
               crop_name=crop_name,
               crop_dir=crop_dir,
               save_fn=save_fn,
               batchsize=batchsize,
               num_batches=num_batches,
               hide_progbar=True)

    return combos_reap(fn=fn, crop_name=crop_name, crop_dir=crop_dir)


def combos_sow_and_reap_to_ds(combos, *, constants=None,
                              fn=None,
                              crop_name=None,
                              crop_dir=None,
                              save_fn=None,
                              batchsize=None,
                              num_batches=None,
                              var_names=None,
                              var_dims=None,
                              var_coords=None,
                              attrs=None,
                              parse=True):
    """
    """

    combos_sow(combos,
               constants=constants,
               fn=fn,
               crop_name=crop_name,
               crop_dir=crop_dir,
               save_fn=save_fn,
               batchsize=batchsize,
               num_batches=num_batches,
               hide_progbar=True)

    return combos_reap_to_ds(fn=fn,
                             crop_name=crop_name,
                             crop_dir=crop_dir,
                             var_names=var_names,
                             var_dims=var_dims,
                             var_coords=var_coords,
                             constants=constants,
                             attrs=attrs,
                             parse=parse)
