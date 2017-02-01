"""Parallel work and scheduling.
"""
import time
import functools
from dask.callbacks import Callback
from ..utils import progbar


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
        if self.fn_name is None:  # pragma: no cover
            #  if function name not defined, just use total graph nodes
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


def dask_scheduler_get(get_mode, num_workers=None):
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
        raise ValueError("\'{}\' is not a valid scheduler.".format(get_mode))


# --------------------------- DASK.DISTRIBUTED ------------------------------ #

@functools.lru_cache(1)
def _distributed_client(n=None):
    """Return a dask.distributed client, but cache it.
    """
    import distributed
    cluster = distributed.LocalCluster(n, scheduler_port=0)
    client = distributed.Client(cluster)
    return client


def distributed_getter(future):
    res = future.result()
    future.release()
    return res


def distributed_getter_stored(future):
    while True:
        if hasattr(future, '_stored_result'):
            return future._stored_result
        else:
            time.sleep(0.1)


def try_stored_then_result(future):
    if hasattr(future, '_stored_result'):
        return future._stored_result
    else:
        return future.result()


def try_stored_then_result_and_release(future):
    if hasattr(future, '_stored_result'):
        res = future._stored_result
    else:
        res = future.result()
    future.release()
    return res


def make_distributed_submit_with_callback(pbar):

    def releaser(future):
        future._stored_result = future.result()
        future.release()
        pbar.update()
        return

    def submitter(pool, fn, *args, **kwargs):
        future = pool.submit(fn, *args, **kwargs)
        future.add_done_callback(releaser)
        return future

    return submitter


def make_distributed_submit_with_callback_replicate(pbar, client):

    def replicator(future):
        client.replicate(future)
        pbar.update()
        return

    def submitter(pool, fn, *args, **kwargs):
        future = pool.submit(fn, *args, **kwargs)
        future.add_done_callback(replicator)
        return future

    return submitter
