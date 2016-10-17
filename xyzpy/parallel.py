"""Parallel work and scheduling.
"""
import functools

from dask.callbacks import Callback

from .utils import progbar


# --------------------------------------------------------------------------- #
# MULTIPROCESSING                                                             #
# --------------------------------------------------------------------------- #

@functools.lru_cache(1)
def _start_multiprocessing_client(n=None):
    import multiprocessing
    p = multiprocessing.Pool(n)

    def _submit(fn, *args, **kwargs):
        future = p.apply_async(fn, args, kwargs)
        future.result = future.get
        return future

    p.submit = _submit
    return p


# --------------------------------------------------------------------------- #
# DASK.DISTRIBUTED                                                            #
# --------------------------------------------------------------------------- #


class DaskTqdmProgbar(Callback):
    def __init__(self, fn_name=None, **kwargs):
        """Callbacks for tracking dask compute progress with tqdm.

        Parameters
        ----------
            fn_name: str
                if given, only count progress on tasks containing this name.
            **kwargs: dict-like
                passed to progbar
        """
        self.fn_name = fn_name
        self.kwargs = kwargs

    def _start(self, dsk):
        if self.fn_name is None:
            total = len(dsk.keys())
        else:
            total = sum(self.fn_name in k for k in dsk.keys())
        self.pbar = progbar(total=total, **self.kwargs)

    def _start_state(self, dsk, state):
        pass

    def _pretask(self, key, dsk, state):
        pass

    def _posttask(self, key, result, dsk, state, id):
        if self.fn_name is not None:
            if self.fn_name in key:
                self.pbar.update()
        else:
            self.pbar.update()

    def _finish(self, dsk, state, errored):
        self.pbar.close()


@functools.lru_cache(1)
def _distributed_client(n=None):
    """Return a dask.distributed client, but cache it.
    """
    import distributed
    cluster = distributed.LocalCluster(n, scheduler_port=0)
    client = distributed.Client(cluster)
    return client


def _dask_get(get_mode, num_workers=None):
    """
    """
    if get_mode.upper() in {'T', 'THREADED'}:
        from dask.threaded import get
        return get
    elif get_mode.upper() in {'M', 'MULTIPROCESS'}:
        from dask.multiprocessing import get
        return get
    elif get_mode.upper() in {'D', 'DISTRIBUTED'}:
        client = _distributed_client(num_workers)
        return client.get
    else:
        raise ValueError("\'" + get_mode + " \' is not a valid scheduler.")
