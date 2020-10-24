import functools

import pytest
import numpy as np
from dask import delayed
from numpy.testing import assert_allclose

import xyzpy as xyz
from xyzpy.utils import _get_fn_name


class TestGetFnName:
    def test_normal(self):
        def foo(a, b):
            pass

        assert _get_fn_name(foo) == 'foo'

    def test_partial(self):
        def foo(a, b):
            pass

        pfoo = functools.partial(foo, b=2)
        assert _get_fn_name(pfoo) == 'foo'

    def test_delayed(self):
        @delayed
        def dfoo(a, b):
            pass

        assert _get_fn_name(dfoo) == 'dfoo'


class TestProgbar:
    def test_normal(self):
        for i in xyz.progbar(range(10)):
            pass

    def test_overide_ascii(self):
        for i in xyz.progbar(range(10), ascii=False):
            pass


class TestTimer:

    def test_simple(self):
        with xyz.Timer() as timer:
            pass
        assert timer.t > 0.0


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

        benchmark_opts = {'starmap': True, 'min_t': 0.01, 'repeats': 3}

        b = xyz.Benchmarker(kernels, setup, benchmark_opts=benchmark_opts)

        ns = [2**i for i in range(1, 6)]

        b.run(ns, verbosity=2)
        b.lineplot()
        b.ilineplot()


class TestRunningStatistics:

    def test_basic(self):
        rs = xyz.RunningStatistics()

        rs.update(42.)
        rs.update(42.)
        rs.update(42.)
        rs.update(42.)

        assert rs.mean == pytest.approx(42.0)
        assert rs.var == pytest.approx(0.0)
        assert rs.std == pytest.approx(0.0)
        assert rs.err == pytest.approx(0.0)

        rs.update_from_it([44., 44., 44., 44.])

        assert rs.mean == pytest.approx(43.0)
        assert rs.std == pytest.approx(1.0)
        assert rs.rel_err == pytest.approx(1. / (43 * 8**0.5))


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
        assert rs.mean == pytest.approx(5, rel=0.1)


class TestGetSizeOf:

    def test_nested_list(self):
        import sys
        obj1 = [[1]]
        obj2 = [[1 << 100 - 1]]
        assert sys.getsizeof(obj1) == sys.getsizeof(obj2)
        assert xyz.getsizeof(obj1) != xyz.getsizeof(obj2)
