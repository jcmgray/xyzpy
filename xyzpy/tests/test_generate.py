from collections import OrderedDict
import numpy as np
from numpy.testing import assert_allclose
from ..generate import (
    sub_split,
    case_runner,
    xr_case_runner,
)


def foo1(a, b, c):
    assert a >= 0
    assert a < 10
    assert b >= 10
    assert b < 100
    assert c >= 100
    assert c < 1000
    return a + b + c


def foo2(a, b, c):
    assert a >= 0
    assert a < 10
    assert b >= 10
    assert b < 100
    assert c >= 100
    assert c < 1000
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
        cases = [('a', [1, 2]),
                 ('b', [10, 20, 30]),
                 ('c', [100, 200, 300, 400])]
        x = case_runner(foo1, cases)
        xn = (np.array([1, 2]).reshape((2, 1, 1)) +
              np.array([10, 20, 30]).reshape((1, 3, 1)) +
              np.array([100, 200, 300, 400]).reshape((1, 1, 4)))
        assert_allclose(x, xn)

    def test_dict(self):
        cases = OrderedDict((('a', [1, 2]),
                             ('b', [10, 20, 30]),
                             ('c', [100, 200, 300, 400])))
        x = case_runner(foo1, cases)
        xn = (np.array([1, 2]).reshape((2, 1, 1)) +
              np.array([10, 20, 30]).reshape((1, 3, 1)) +
              np.array([100, 200, 300, 400]).reshape((1, 1, 4)))
        assert_allclose(x, xn)

    def test_multires(self):
        cases = [('a', [1, 2]),
                 ('b', [10, 20, 30]),
                 ('c', [100, 200, 300, 400])]
        x, y = case_runner(foo2, cases, output_dims=[1, 1])
        xn = (np.array([1, 2]).reshape((2, 1, 1)) +
              np.array([10, 20, 30]).reshape((1, 3, 1)) +
              np.array([100, 200, 300, 400]).reshape((1, 1, 4)))
        yn = (np.array([1, 2]).reshape((2, 1, 1)) %
              np.array([2]*24).reshape((2, 3, 4))) == 0
        assert_allclose(x, xn)
        assert_allclose(y, yn)


class TestParallelCaseRunner:
    def test_basic(self):
        cases = (('a', [1, 2]),
                 ('b', [10, 20, 30]),
                 ('c', [100, 200, 300, 400]))
        x = case_runner(foo1, cases, processes=2)
        xn = (np.array([1, 2]).reshape((2, 1, 1)) +
              np.array([10, 20, 30]).reshape((1, 3, 1)) +
              np.array([100, 200, 300, 400]).reshape((1, 1, 4)))
        assert_allclose(x, xn)

    def test_multires(self):
        cases = (('a', [1, 2]),
                 ('b', [10, 20, 30]),
                 ('c', [100, 200, 300, 400]))
        x = case_runner(foo2, cases, processes=2, output_dims=[1, 1])
        xn = (np.array([1, 2]).reshape((2, 1, 1)) +
              np.array([10, 20, 30]).reshape((1, 3, 1)) +
              np.array([100, 200, 300, 400]).reshape((1, 1, 4)))
        assert_allclose(x[0], xn)
        assert np.all(np.asarray(x[1])[1, ...])

    def test_dict(self):
        cases = OrderedDict((('a', [1, 2]),
                             ('b', [10, 20, 30]),
                             ('c', [100, 200, 300, 400])))
        x = [*case_runner(foo1, cases, processes=2)]
        xn = (np.array([1, 2]).reshape((2, 1, 1)) +
              np.array([10, 20, 30]).reshape((1, 3, 1)) +
              np.array([100, 200, 300, 400]).reshape((1, 1, 4)))
        assert_allclose(x, xn)


class TestXRCaseRunner:
    def test_basic(self):
        def foo(a, b, c):
            return a + b + c
        cases = (('a', [1, 2]),
                 ('b', [10, 20, 30]),
                 ('c', [100, 200, 300, 400]))
        ds = xr_case_runner(foo, cases, 'bananas')
        assert ds.sel(a=2, b=30, c=400)['bananas'].data == 432

    def test_multiresult(self):
        def foo(a, b, c):
            return a + b + c, a % 2 == 0
        cases = (('a', [1, 2]),
                 ('b', [10, 20, 30]),
                 ('c', [100, 200, 300, 400]))
        ds = xr_case_runner(foo, cases, ['bananas', 'cakes'])
        assert ds.bananas.data.dtype == int
        assert ds.cakes.data.dtype == bool
        assert ds.sel(a=2, b=30, c=400)['bananas'].data == 432
        assert ds.sel(a=1, b=10, c=100)['bananas'].data == 111
        assert ds.sel(a=2, b=30, c=400)['cakes'].data
        assert not ds.sel(a=1, b=10, c=100)['cakes'].data
