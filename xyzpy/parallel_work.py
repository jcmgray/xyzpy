import functools


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
# IPYPARALLEL                                                                 #
# --------------------------------------------------------------------------- #

def _start_ipcluster(n=None):
    """Start the local ipyparallel cluster with `n` workers.
    """
    from subprocess import Popen, DEVNULL
    cmd = ["ipcluster", "start"]
    if n is not None:
        cmd += ["-n", str(n)]
    Popen(cmd, stderr=DEVNULL, stdout=DEVNULL)


def _stop_ipcluster():
    """Stop the local ipyparallel cluster,
    """
    from subprocess import Popen, DEVNULL
    cmd = ["ipcluster", "stop"]
    Popen(cmd, stderr=DEVNULL, stdout=DEVNULL)


class _IpyparallelLocalCluster(object):
    """A class to manage the automatic starting and closing of the
    ipyparallel local cluster.
    """
    def __init__(self, n=None):
        self.process = _start_ipcluster(n)

    def __del__(self):
        _stop_ipcluster()


@functools.lru_cache(1)
def _ipyparallel_client(n=None):
    """Return a load balanced ipyparllel client, but cache it.
    """
    import ipyparallel
    cluster = _IpyparallelLocalCluster(n=n)
    rc = ipyparallel.Client()
    rc._local_cluster = cluster  # keep reference to cluster
    lbv = rc.load_balanced_view()
    lbv.submit = lbv.apply_async
    return lbv


# --------------------------------------------------------------------------- #
# DASK.DISTRIBUTED                                                            #
# --------------------------------------------------------------------------- #

@functools.lru_cache(1)
def _distributed_client(n=None):
    """Return a dask.distributed client, but cache it.
    """
    import distributed
    cluster = distributed.LocalCluster(n, scheduler_port=0)
    client = distributed.Client(cluster)
    return client


# --------------------------------------------------------------------------- #
# WORKER POOL                                                                 #
# --------------------------------------------------------------------------- #

class Pool(object):
    def __init__(self, n, backend='MULTIPROCESSING'):
        self.n = n
        self.backend = backend

    # --------------- #
    # Context Manager #
    # --------------- #
    def __enter__(self):
        if self.backend == 'MULTIPROCESSING':
            return _start_multiprocessing_client(self.n)
        elif self.backend == 'IPYPARALLEL':
            return _ipyparallel_client(self.n)
        elif self.backend == 'DISTRIBUTED':
            return _distributed_client(self.n)
        else:
            raise ValueError("Invalid backend specified.")

    def __exit__(self, type, value, traceback):
        pass
        # if self.backend == 'MULTIPROCESSING':
        #     self.workers.close()
        #     del self.workers
        # elif self.backend == 'IPYPARALLEL':
        #     self.workers.shutdown()
        #     del self.workers
        # elif self.backend == 'DISTRIBUTED':
        #     pass
        #     # self.workers.shutdown()
        #     # del self.workers

    # --------------- #
    # Direct          #
    # --------------- #
    def _submit_mp(self, fn, *args, **kwargs):
        future = self.workers.apply_async(fn, args, kwargs)
        future.result = future.get
        return future

    def _submit_ipy(self, fn, *args, **kwargs):
        return self.workers.apply_async(fn, *args, **kwargs)

    def _submit_dist(self, fn, *args, **kwargs):
            return self.workers.submit(fn, *args, **kwargs)
