import collections
import functools
import inspect
import operator
import warnings


@functools.lru_cache(None)
def get_ray():
    """Import and return the ``ray`` module (cached)."""
    import ray

    return ray


class RayFuture:
    """Basic ``concurrent.futures`` like future wrapping a ray ``ObjectRef``."""

    __slots__ = ("_obj", "_cancelled")

    def __init__(self, obj):
        self._obj = obj
        self._cancelled = False

    def result(self, timeout=None):
        return get_ray().get(self._obj, timeout=timeout)

    def done(self):
        return self._cancelled or bool(
            get_ray().wait([self._obj], timeout=0)[0]
        )

    def cancel(self):
        get_ray().cancel(self._obj)
        self._cancelled = True


def _unpack_futures_tuple(x):
    return tuple(map(_unpack_futures, x))


def _unpack_futures_list(x):
    return list(map(_unpack_futures, x))


def _unpack_futures_dict(x):
    return {k: _unpack_futures(v) for k, v in x.items()}


def _unpack_futures_identity(x):
    return x


_unpack_dispatch = collections.defaultdict(
    lambda: _unpack_futures_identity,
    {
        RayFuture: operator.attrgetter("_obj"),
        tuple: _unpack_futures_tuple,
        list: _unpack_futures_list,
        dict: _unpack_futures_dict,
    },
)


def _unpack_futures(x):
    """Allows passing futures by reference - takes e.g. args and kwargs and
    replaces all ``RayFuture`` objects with their underyling ``ObjectRef``
    within all nested tuples, lists and dicts.

    [Subclassing ``ObjectRef`` might avoid needing this.]
    """
    return _unpack_dispatch[x.__class__](x)


@functools.lru_cache(2**14)
def get_remote_fn(fn, **remote_opts):
    """Cached retrieval of remote function."""
    ray = get_ray()
    if remote_opts:
        return ray.remote(**remote_opts)(fn)
    return ray.remote(fn)


@functools.lru_cache(2**14)
def get_fn_as_remote_object(fn):
    """Store ``fn`` in the Ray object store and return an ``ObjectRef``."""
    ray = get_ray()
    return ray.put(fn)


@functools.lru_cache(None)
def get_deploy(**remote_opts):
    """Alternative for 'non-function' callables - e.g. partial
    functions - pass the callable object too.
    """
    ray = get_ray()

    def deploy(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    if remote_opts:
        return ray.remote(**remote_opts)(deploy)
    return ray.remote(deploy)


class RayExecutor:
    """Basic ``concurrent.futures`` like interface using ``ray``.

    Example usage::

        from xyzpy import RayExecutor

        # create a pool that by default requests a single gpu per task
        pool = RayExecutor(
            num_cpus=4,
            num_gpus=4,
            default_remote_opts={"num_gpus": 1},
        )

    """

    def __init__(self, *args, default_remote_opts=None, **kwargs):
        ray = get_ray()
        if not ray.is_initialized():
            ray.init(*args, **kwargs)
        elif args or kwargs:
            warnings.warn(
                "Ignoring arguments to RayExecutor constructor, "
                "ray is already initialized."
            )

        self.default_remote_opts = (
            {} if default_remote_opts is None else dict(default_remote_opts)
        )

    def _maybe_inject_remote_opts(self, remote_opts=None):
        """Return the default remote options, possibly overriding some with
        those supplied by a ``submit call``.
        """
        ropts = self.default_remote_opts
        if remote_opts is not None:
            ropts = {**ropts, **remote_opts}
        return ropts

    def submit(self, fn, *args, pure=False, remote_opts=None, **kwargs):
        """Remotely run ``fn(*args, **kwargs)``, returning a ``RayFuture``."""
        # want to pass futures by reference
        args = _unpack_futures_tuple(args)
        kwargs = _unpack_futures_dict(kwargs)

        ropts = self._maybe_inject_remote_opts(remote_opts)

        # this is the same test ray uses to accept functions
        if inspect.isfunction(fn):
            # can use the faster cached remote function
            obj = get_remote_fn(fn, **ropts).remote(*args, **kwargs)
        else:
            fn_obj = get_fn_as_remote_object(fn)
            obj = get_deploy(**ropts).remote(fn_obj, *args, **kwargs)

        return RayFuture(obj)

    def map(self, func, *iterables, remote_opts=None):
        """Remote map ``func`` over arguments ``iterables``."""
        ropts = self._maybe_inject_remote_opts(remote_opts)
        remote_fn = get_remote_fn(func, **ropts)
        objs = tuple(map(remote_fn.remote, *iterables))
        ray = get_ray()
        return map(ray.get, objs)

    def scatter(self, data):
        """Push ``data`` into the distributed store, returning an ``ObjectRef``
        that can be supplied to ``submit`` calls for example.
        """
        ray = get_ray()
        return ray.put(data)

    def shutdown(self):
        """Shutdown the parent ray cluster, this ``RayExecutor`` instance
        itself does not need any cleanup.
        """
        get_ray().shutdown()


class RayGPUExecutor(RayExecutor):
    """A ``RayExecutor`` that by default requests a single gpu per task."""

    def __init__(self, *args, gpus_per_task=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.default_remote_opts.setdefault("num_gpus", gpus_per_task)
