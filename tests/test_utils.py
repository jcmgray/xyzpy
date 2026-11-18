import functools
import os
import sys
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest
from dask import delayed
from numpy.testing import assert_allclose

import xyzpy as xyz
from xyzpy.utils import _get_fn_name


class TestGetFnName:
    def test_normal(self):
        def foo(a, b):
            pass

        assert _get_fn_name(foo) == "foo"

    def test_partial(self):
        def foo(a, b):
            pass

        pfoo = functools.partial(foo, b=2)
        assert _get_fn_name(pfoo) == "foo"

    def test_delayed(self):
        @delayed
        def dfoo(a, b):
            pass

        assert _get_fn_name(dfoo) == "dfoo"


class TestProgbar:
    def test_normal(self):
        for i in xyz.progbar(range(10)):
            pass

    def test_overide_ascii(self):
        for i in xyz.progbar(range(10), ascii=False):
            pass


class TestTimer:
    def test_simple(self):
        import time

        with xyz.Timer() as timer:
            time.sleep(0.01)
        assert timer.t > 0.0


class TestMemoryMonitorGpu:
    def test_peak_total_across_devices(self, monkeypatch):
        pid = os.getpid()
        events = []
        used = [4, 1]

        def processes(device):
            return [
                SimpleNamespace(pid=pid, usedGpuMemory=used[device] << 30),
                SimpleNamespace(pid=pid + 1, usedGpuMemory=100 << 30),
            ]

        nvml = SimpleNamespace(
            nvmlInit=lambda: events.append("init"),
            nvmlShutdown=lambda: events.append("shutdown"),
            nvmlDeviceGetCount=lambda: 2,
            nvmlDeviceGetHandleByIndex=lambda index: index,
            nvmlDeviceGetComputeRunningProcesses=processes,
        )
        monkeypatch.setitem(sys.modules, "pynvml", nvml)

        with xyz.MemoryMonitorGpu(interval=60) as mem:
            assert mem.peak == pytest.approx(5.0)
            used[:] = [1, 5]
        assert mem.peak == pytest.approx(6.0)
        assert events == ["init", "shutdown"]

        used[:] = [0, 1]
        with mem:
            pass
        assert mem.peak == pytest.approx(1.0)

    def test_background_sample(self, monkeypatch):
        sampled = Event()
        used = 0
        pid = os.getpid()

        def processes(device):
            process = SimpleNamespace(pid=pid, usedGpuMemory=used)
            if process.usedGpuMemory:
                sampled.set()
            return [process]

        nvml = SimpleNamespace(
            nvmlInit=lambda: None,
            nvmlShutdown=lambda: None,
            nvmlDeviceGetCount=lambda: 1,
            nvmlDeviceGetHandleByIndex=lambda index: index,
            nvmlDeviceGetComputeRunningProcesses=processes,
        )
        monkeypatch.setitem(sys.modules, "pynvml", nvml)

        with xyz.MemoryMonitorGpu(interval=0.001) as mem:
            used = 3 << 30
            assert sampled.wait(timeout=5)
            used = 0
        assert mem.peak == pytest.approx(3.0)

    @pytest.mark.parametrize("body_fails", [False, True])
    def test_cleanup_after_failed_sample(self, monkeypatch, body_fails):
        events = []
        processes = []
        nvml = SimpleNamespace(
            nvmlInit=lambda: events.append("init"),
            nvmlShutdown=lambda: events.append("shutdown"),
            nvmlDeviceGetCount=lambda: 1,
            nvmlDeviceGetHandleByIndex=lambda index: index,
            nvmlDeviceGetComputeRunningProcesses=lambda device: processes,
        )
        monkeypatch.setitem(sys.modules, "pynvml", nvml)

        error = ValueError if body_fails else RuntimeError
        match = "work failed" if body_fails else "GPU memory is unavailable"
        with (
            pytest.raises(error, match=match),
            xyz.MemoryMonitorGpu(interval=60) as mem,
        ):
            assert mem.peak == pytest.approx(0.0)
            processes.append(
                SimpleNamespace(pid=os.getpid(), usedGpuMemory=None)
            )
            if body_fails:
                raise ValueError("work failed")
        assert events == ["init", "shutdown"]

    def test_requires_nvml(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "pynvml", None)
        with (
            pytest.raises(ImportError, match="requires nvidia-ml-py"),
            xyz.MemoryMonitorGpu(),
        ):
            pass


class TestMemoryMonitorGpuTorch:
    @pytest.mark.parametrize(
        ("device", "index"), [(None, 1), ("cuda", 1), (0, 0), ("cuda:0", 0)]
    )
    def test_peak_stats(self, monkeypatch, device, index):
        current = 1
        events = []

        def result(name, device, gib):
            events.append((name, device))
            return int(gib * (1 << 30))

        cuda = SimpleNamespace(
            is_available=lambda: True,
            current_device=lambda: current,
            device_count=lambda: 3,
            init=lambda: events.append(("init", None)),
            reset_peak_memory_stats=lambda i: events.append(("reset", i)),
            max_memory_allocated=lambda i: result("allocated", i, 1.5),
            max_memory_reserved=lambda i: result("reserved", i, 2.0),
        )
        torch = SimpleNamespace(
            cuda=cuda,
            device=lambda spec: SimpleNamespace(
                type="cuda", index=0 if spec == "cuda:0" else None
            ),
        )
        monkeypatch.setitem(sys.modules, "torch", torch)

        with xyz.MemoryMonitorGpuTorch(device=device) as mem:
            assert events == [("init", None), ("reset", index)]
            current = 2

        assert events == [
            ("init", None),
            ("reset", index),
            ("allocated", index),
            ("reserved", index),
        ]
        assert mem.peak == pytest.approx(1.5)
        assert mem.peak_reserved == pytest.approx(2.0)

    def test_requires_torch(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        with (
            pytest.raises(ImportError, match="requires torch"),
            xyz.MemoryMonitorGpuTorch(),
        ):
            pass

    def test_requires_cuda(self, monkeypatch):
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False)
        )
        monkeypatch.setitem(sys.modules, "torch", torch)
        with (
            pytest.raises(RuntimeError, match="torch CUDA is not available"),
            xyz.MemoryMonitorGpuTorch(),
        ):
            pass


class TestBenchmark:
    def test_no_setup_no_size(self):
        t = xyz.benchmark(lambda: np.linalg.eig(np.random.randn(100, 100)))
        assert t > 0

    def test_no_setup_with_size(self):
        n = 10
        t = xyz.benchmark(lambda n: np.linalg.eig(np.random.randn(n, n)), n=n)
        assert t > 0

    def test_with_setup_no_size(self):
        t = xyz.benchmark(
            fn=lambda X: np.linalg.eig(X),
            setup=lambda: np.random.randn(10, 10),
        )
        assert t > 0

    def test_with_setup_with_size(self):
        t = xyz.benchmark(
            fn=lambda X: np.linalg.eig(X),
            setup=lambda n: np.random.randn(n, n),
            n=10,
        )
        assert t > 0

    def test_torch_cuda_sync(self, monkeypatch):
        events = []
        torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                synchronize=lambda: events.append("sync"),
            )
        )
        monkeypatch.setitem(sys.modules, "torch", torch)

        def setup():
            events.append("setup")

        def fn(_):
            events.append("fn")

        t = xyz.benchmark(
            fn,
            setup=setup,
            min_t=0.0,
            repeats=1,
            torch_cuda_sync=True,
        )

        assert t > 0
        assert events == ["setup", "sync", "fn", "sync"]

    def test_torch_cuda_sync_requires_torch(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)

        with pytest.raises(
            ImportError,
            match="torch is required when torch_cuda_sync=True",
        ):
            xyz.benchmark(lambda: None, torch_cuda_sync=True)

    def test_torch_cuda_sync_requires_cuda(self, monkeypatch):
        torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False)
        )
        monkeypatch.setitem(sys.modules, "torch", torch)

        with pytest.raises(RuntimeError, match="torch CUDA is not available"):
            xyz.benchmark(lambda: None, torch_cuda_sync=True)


class TestBenchmarker:
    def test_basic(self):

        def add1(a, b):
            return a + b

        def add2(a, b):
            c = np.zeros_like(a)
            c += b
            return c

        kernels = [add1, add2]

        def setup(n):
            a = np.random.randn(n)
            b = np.random.randn(n)
            return a, b

        benchmark_opts = {"starmap": True, "min_t": 0.01, "repeats": 3}

        b = xyz.Benchmarker(kernels, setup, benchmark_opts=benchmark_opts)

        ns = [2**i for i in range(1, 6)]

        b.run(ns, verbosity=2)
        b.lineplot()
        b.ilineplot()


class TestRunningStatistics:
    def test_basic(self):
        rs = xyz.RunningStatistics()

        rs.update(42.0)
        rs.update(42.0)
        rs.update(42.0)
        rs.update(42.0)

        assert rs.mean == pytest.approx(42.0)
        assert rs.var == pytest.approx(0.0)
        assert rs.std == pytest.approx(0.0)
        assert rs.err == pytest.approx(0.0)

        rs.update_from_it([44.0, 44.0, 44.0, 44.0])

        assert rs.mean == pytest.approx(43.0)
        assert rs.std == pytest.approx(1.0)
        assert rs.rel_err == pytest.approx(1.0 / (43 * 8**0.5))


class TestRunningCovariance:
    def test_matches_numpy(self):
        xs = np.random.randn(100)
        ys = np.random.randn(100)
        rc = xyz.RunningCovariance()
        rc.update(xs[0], ys[0])
        rc.update(xs[1], ys[1])
        rc.update(xs[2], ys[2])
        assert rc.sample_covar == pytest.approx(np.cov(xs[:3], ys[:3])[0, 1])
        rc.update_from_it(xs[3:], ys[3:])
        assert rc.sample_covar == pytest.approx(np.cov(xs, ys)[0, 1])
        assert rc.covar == pytest.approx(np.cov(xs, ys, bias=True)[0, 1])

    def test_matrix_form(self):
        xs = np.random.rand(1000)
        ys = 0.2 * xs * np.random.rand(1000)
        zs = 0.4 * ys * np.random.rand(1000)
        rcm = xyz.RunningCovarianceMatrix(n=3)
        rcm.update_from_it(xs, ys, zs)
        assert_allclose(np.cov([xs, ys, zs]), rcm.sample_covar_matrix)
        assert_allclose(np.cov([xs, ys, zs], bias=True), rcm.covar_matrix)


class TestFromRepeats:
    def test_basic(self):

        def fn(n):
            return np.random.rand(n).sum()

        rs = xyz.estimate_from_repeats(fn, 10, verbosity=2)
        assert rs.mean == pytest.approx(5, rel=0.2)


class TestGetSizeOf:
    def test_nested_list(self):
        import sys

        obj1 = [[1]]
        obj2 = [[1 << 100 - 1]]
        assert sys.getsizeof(obj1) == sys.getsizeof(obj2)
        assert xyz.getsizeof(obj1) != xyz.getsizeof(obj2)
