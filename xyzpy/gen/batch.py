import os
import shutil
from glob import glob
from itertools import chain

try:
    import joblib
except ImportError:
    pass

try:
    import cloudpickle
except ImportError:
    pass

from ..utils import _get_fn_name
from .prepare import _parse_combos, _parse_constants
from .combo_runner import _combo_runner


BTCH_NM = "xyz-batch-{}.jbdmp"
RSLT_NM = "xyz-result-{}.jbdmp"
FNCT_NM = "xyz-function.clpkl"
CMBS_NM = "xyz-combos.jbdmp"


class XYZError(Exception):
    pass


def parse_field_details(fn, field_name, field_dir):
    """Work out how to structure the sowed data.

    Parameters
    ----------
        fn : callable (optional)
            Function to infer name field_name from, if not given.
        field_name : str (optional)
            Specific name to give this set of runs.
        field_dir : str (optional)
            Specific directory to put the ".xyz-{field_name}/" folder in
            with all the cases and results.

    Returns
    -------
        field_folder : str
            Full path to the field-folder.
    """
    if field_name is None:
        if fn is None:
            raise ValueError("Either `fn` or `field_name` must be give.")
        field_name = _get_fn_name(fn)

    field_folder = os.path.join(
        field_dir if field_dir is not None else os.getcwd(),
        ".xyz-{}".format(field_name))

    return field_folder


class Sower(object):
    """Class for sowing a 'field' of batched combos to then 'grow' (on any
    number of workers sharing the filesystem) and then reap.
    """

    def __init__(self, *,
                 fn=None,
                 field_name=None,
                 field_dir=None,
                 save_fn=None,
                 batchsize=None):
        """
        Parameters
        ----------
            fn : callable (optional)
                Target function - `field_name` will be inferred from this if
                not given explicitly. If given, `Sower` will also default
                to saving a version of `fn` to disk for `batch.grow` to use.
            field_name : str (optional)
                Custom name for this set of runs - must be given if `fn`
                is not.
            field_dir : str (optional)
                If given, alternative directory to put the ".xyz-{field_name}/"
                folder in with all the cases and results.
            save_fn : bool (optional)
                Whether to save the function to disk for `batch.grow` to use.
                Will default to True if `fn` is given.
            batchsize : int (optional)
                How many cases to group into a single batch per worker.
                By default, batchsize=1.
        """
        self.fn = fn
        self.field_folder = parse_field_details(fn, field_name, field_dir)

        # Create directory structure
        # XXX: only allow fresh / new folders? option to clear/overwrite?
        os.makedirs(os.path.join(self.field_folder, "cases"), exist_ok=True)
        os.makedirs(os.path.join(self.field_folder, "results"), exist_ok=True)

        # Save function so it can be automatically loaded with all deps?
        if fn is None:
            if save_fn is True:
                raise ValueError("Must specify a function for it to be saved!")
        elif save_fn is not False:
            self.save_function_to_disk()

        self.batchsize = batchsize if batchsize is not None else 1
        if not isinstance(self.batchsize, int):
            raise TypeError("`batchsize` must be an integer.")
        if self.batchsize < 1:
            raise ValueError("`batchsize` must be >= 1.")

        # Internal
        self._batch_cases = []
        self._counter = 0
        self._batch_counter = 0

    def save_function_to_disk(self):
        """Save the base function to disk using cloudpickle
        """
        import cloudpickle
        joblib.dump(cloudpickle.dumps(self.fn),
                    os.path.join(self.field_folder, FNCT_NM))

    def save_batch_cases(self):
        """Save the current batch of cases to disk using joblib.dump
         and start the next batch.
        """
        self._batch_counter += 1
        joblib.dump(self._batch_cases,
                    os.path.join(self.field_folder, "cases",
                                 BTCH_NM.format(self._batch_counter)))

    def __enter__(self):
        return self

    def __call__(self, **kwargs):
        self._batch_cases.append(kwargs)
        self._counter += 1

        if self._counter == self.batchsize:
            self.save_batch_cases()
            self._batch_cases = []
            self._counter = 0

    def __exit__(self, exception_type, exception_value, traceback):
        # Make sure any overfill also saved
        if self._batch_cases:
            self.save_batch_cases()


def sow_combos(combos, *, constants=None,
               fn=None,
               field_name=None,
               field_dir=None,
               save_fn=None,
               batchsize=None):

    combos = _parse_combos(combos)
    constants = _parse_constants(constants)

    field_folder = parse_field_details(fn, field_name, field_dir)

    # Sort to ensure order remains same for reaping results (don't hash kwargs)
    combos = sorted(combos, key=lambda x: x[0])

    sow_opts = {
        'fn': fn,
        'field_name': field_name,
        'field_dir': field_dir,
        'batchsize': batchsize,
        'save_fn': save_fn,
    }

    with Sower(**sow_opts) as s:
        _combo_runner(fn=s, combos=combos, constants=constants)

    # Save the combos for automatic reaping
    joblib.dump(combos, os.path.join(field_folder, CMBS_NM))


def grow(batch_number, fn=None, field_name=None, field_dir=None,
         check_mpi=True):
    """Automatically process a batch of cases into results. Should be run in an
    ".xyz-{fn_name}" folder.
    """
    # TODO: warn if batch can't be found

    if fn is field_name is field_dir is None:
        field_folder = os.getcwd()
        if os.path.relpath('.', '..')[:5] != ".xyz-":
            raise XYZError("`grow` should be run in a "
                           "\"{field_dir}/.xyz-{field_name}\" folder, else "
                           "`field_dir` and `field_name` (or `fn`) should be "
                           "specified.")
    else:
        field_folder = parse_field_details(fn, field_name, field_dir)

    # load function
    if fn is None:
        fn = cloudpickle.loads(
            joblib.load(os.path.join(field_folder, FNCT_NM)))

    # load cases to evaluate
    cases = joblib.load(
        os.path.join(field_folder, "cases", BTCH_NM.format(batch_number)))

    # process each case
    results = tuple(fn(**kws) for kws in cases)

    # maybe want to run grow as mpiexec, but only save and delete once...
    if check_mpi:
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()
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

    def __init__(self, fn=None, field_name=None, field_dir=None):
        """Class for retrieving the batched, flat, 'grown' results.

        Parameters
        ----------
            fn : callable (optional)
                Target function - `field_name` will be inferred from this if
                not given explicitly.
            field_name : str (optional)
                Custom name for the set of batches - must be given if `fn`
                is not.
            field_dir : str (optional)
                If given, alternative to current working directory for results
                to be reaped from.
        """
        self.field_folder = parse_field_details(fn, field_name, field_dir)

        if glob(os.path.join(self.field_folder, "cases", BTCH_NM.format("*"))):
            raise XYZError("Not all cases have been successfully run yet!")

        # Lazily load each batch in order and iterate through its results
        self.results = chain.from_iterable(
            map(lambda x: joblib.load(x),
                sorted(glob(
                    os.path.join(self.field_folder,
                                 "results",
                                 RSLT_NM.format("*"))))))

    def __enter__(self):
        return self

    def __call__(self, **kwargs):
        return next(self.results)

    def __exit__(self, exception_type, exception_value, traceback):
        if tuple(*self.results):
            raise XYZError("Not all results reaped! Reaper must be called"
                           "with exactly the same combinations as the Sower"
                           "used to generate the cases run.")
        else:
            shutil.rmtree(self.field_folder)


def reap_combos(fn=None, field_name=None, field_dir=None):
    """
    """
    field_folder = parse_field_details(fn, field_name, field_dir)
    combos = joblib.load(os.path.join(field_folder, CMBS_NM))

    with Reaper(fn, field_name, field_dir) as r:
        results = _combo_runner(r, combos, constants={})

    return results
