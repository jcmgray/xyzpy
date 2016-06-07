from collections import OrderedDict
import numpy as np
from numpy.testing import assert_allclose
from ..generate import (
    sub_split,
    case_runner,
    parallel_case_runner,
    xr_case_runner,
)


def foo1(a, b, c):
    return a + b + c


def foo2(a, b, c):
    return a + b + c, a % 2 == 0


def foo1s(abc):
    a, b, c = abc
    return a + b + c


def foo2s(abc):
        a, b, c = abc
        return a + b + c, a % 2 == 0


class TestSubSplit:
    def test_2res(self):
        a = [[[('a', 1), ('b', 2)],
              [('c', 3), ('d', 4)]],
             [[('e', 5), ('f', 6)],
              [('g', 7), ('h', 8)]]]
        c, d = sub_split(a)
        assert c.tolist() == [[['a', 'b'],
                               ['c', 'd']],
                              [['e', 'f'],
                               ['g', 'h']]]
        assert d.tolist() == [[[1, 2],
                               [3, 4]],
                              [[5, 6],
                               [7, 8]]]


class TestCaseRunner:
    def test_simple(self):
        params = [('a', [1, 2]),
                  ('b', [10, 20, 30]),
                  ('c', [100, 200, 300, 400])]
        x = case_runner(foo1, params)
        xn = (np.array([1, 2]).reshape((2, 1, 1)) +
              np.array([10, 20, 30]).reshape((1, 3, 1)) +
              np.array([100, 200, 300, 400]).reshape((1, 1, 4)))
        assert_allclose(x, xn)

    def test_dict(self):
        params = OrderedDict((('a', [1, 2]),
                              ('b', [10, 20, 30]),
                              ('c', [100, 200, 300, 400])))
        x = case_runner(foo1, params)
        xn = (np.array([1, 2]).reshape((2, 1, 1)) +
              np.array([10, 20, 30]).reshape((1, 3, 1)) +
              np.array([100, 200, 300, 400]).reshape((1, 1, 4)))
        assert_allclose(x, xn)

    def test_multires(self):
        params = [('a', [1, 2]),
                  ('b', [10, 20, 30]),
                  ('c', [100, 200, 300, 400])]
        x, y = case_runner(foo2, params, split=True)
        xn = (np.array([1, 2]).reshape((2, 1, 1)) +
              np.array([10, 20, 30]).reshape((1, 3, 1)) +
              np.array([100, 200, 300, 400]).reshape((1, 1, 4)))
        yn = (np.array([1, 2]).reshape((2, 1, 1)) %
              np.array([2]*24).reshape((2, 3, 4))) == 0
        assert_allclose(x, xn)
        assert_allclose(y, yn)


class TestParallelCaseRunner:
    def test_basic(self):
        params = (('a', [1, 2]),
                  ('b', [10, 20, 30]),
                  ('c', [100, 200, 300, 400]))
        x = parallel_case_runner(foo1s, params)
        xn = (np.array([1, 2]).reshape((2, 1, 1)) +
              np.array([10, 20, 30]).reshape((1, 3, 1)) +
              np.array([100, 200, 300, 400]).reshape((1, 1, 4)))
        assert_allclose(x, xn)

    def test_multires(self):
        params = (('a', [1, 2]),
                  ('b', [10, 20, 30]),
                  ('c', [100, 200, 300, 400]))
        x = parallel_case_runner(foo2s, params)
        xn = (np.array([1, 2]).reshape((2, 1, 1)) +
              np.array([10, 20, 30]).reshape((1, 3, 1)) +
              np.array([100, 200, 300, 400]).reshape((1, 1, 4)))
        assert_allclose(x[0], xn)
        assert np.all(x[1][1, ...])

    def test_dict(self):
        params = OrderedDict((('a', [1, 2]),
                              ('b', [10, 20, 30]),
                              ('c', [100, 200, 300, 400])))
        x = [*parallel_case_runner(foo1s, params)]
        xn = (np.array([1, 2]).reshape((2, 1, 1)) +
              np.array([10, 20, 30]).reshape((1, 3, 1)) +
              np.array([100, 200, 300, 400]).reshape((1, 1, 4)))
        assert_allclose(x, xn)


class TestXRCaseRunner:
    def test_basic(self):
        def foo(a, b, c):
            return a + b + c
        params = (('a', [1, 2]),
                  ('b', [10, 20, 30]),
                  ('c', [100, 200, 300, 400]))
        ds = xr_case_runner(foo, params, 'bananas')
        assert ds.sel(a=2, b=30, c=400)['bananas'].data == 432

    def test_multiresult(self):
        def foo(a, b, c):
            return a + b + c, a % 2 == 0
        params = (('a', [1, 2]),
                  ('b', [10, 20, 30]),
                  ('c', [100, 200, 300, 400]))
        ds = xr_case_runner(foo, params, ['bananas', 'cakes'])
        assert ds.bananas.data.dtype == int
        assert ds.cakes.data.dtype == bool
        assert ds.sel(a=2, b=30, c=400)['bananas'].data == 432
        assert ds.sel(a=1, b=10, c=100)['bananas'].data == 111
        assert ds.sel(a=2, b=30, c=400)['cakes'].data
        assert not ds.sel(a=1, b=10, c=100)['cakes'].data
