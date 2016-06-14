from collections import OrderedDict
from functools import partial

from pytest import raises
import numpy as np
from numpy.testing import assert_allclose

from ..generate import (
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


def foo_t(a, b):
    assert a >= 0
    assert a < 10
    assert b >= 10
    assert b < 100
    return [b + a + 0.1*i for i in range(10)]


def foo_t2(a, b):
    assert a >= 0
    assert a < 10
    assert b >= 10
    assert b < 100
    return [b + a + 0.1*i for i in range(10)], a % 2 == 0


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

    def test_single_case(self):
        cases = [('a', [1, 2])]
        x = case_runner(partial(foo1, b=20, c=300), cases)
        assert_allclose(x, [321, 322])

    def test_single_case_single_tuple(self):
        cases = ('a', [1, 2])
        constants = {'b': 20, 'c': 300}
        x = case_runner(foo1, cases, constants=constants)
        assert_allclose(x, [321, 322])

    def test_multires(self):
        cases = [('a', [1, 2]),
                 ('b', [10, 20, 30]),
                 ('c', [100, 200, 300, 400])]
        x, y = case_runner(foo2, cases, split=True)
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
        x = case_runner(foo2, cases, processes=2, split=True)
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
        cases = (('a', [1, 2]),
                 ('b', [10, 20, 30]),
                 ('c', [100, 200, 300, 400]))
        ds = xr_case_runner(foo1, cases, var_names=['bananas'])
        assert ds.sel(a=2, b=30, c=400)['bananas'].data == 432

    def test_multiresult(self):
        cases = (('a', [1, 2]),
                 ('b', [10, 20, 30]),
                 ('c', [100, 200, 300, 400]))
        ds = xr_case_runner(foo2, cases, var_names=['bananas', 'cakes'])
        assert ds.bananas.data.dtype == int
        assert ds.cakes.data.dtype == bool
        assert ds.sel(a=2, b=30, c=400)['bananas'].data == 432
        assert ds.sel(a=1, b=10, c=100)['bananas'].data == 111
        assert ds.sel(a=2, b=30, c=400)['cakes'].data
        assert not ds.sel(a=1, b=10, c=100)['cakes'].data

    def test_arrayresult(self):
        cases = (('a', [1, 2]),
                 ('b', [10, 20, 30]))
        ds = xr_case_runner(foo_t, cases,
                            var_names='bananas',
                            var_dims=(['sugar']),
                            var_coords={'sugar': [*range(10)]})
        assert ds.bananas.data.dtype == float
        assert_allclose(ds.sel(a=2, b=30)['bananas'].data,
                        [32.0, 32.1, 32.2, 32.3, 32.4,
                         32.5, 32.6, 32.7, 32.8, 32.9,])

    def test_array_and_single_result(self):
        cases = (('a', [1, 2]),
                 ('b', [10, 20, 30]))
        ds = xr_case_runner(foo_t2, cases,
                            var_names=['bananas', 'ripe'],
                            var_dims=(['sugar'], []),
                            var_coords={'sugar': [*range(10, 20)]})
        assert ds.ripe.data.dtype == bool
        assert ds.sel(a=2, b=30, sugar=14)['bananas'].data == 32.4
        with raises(KeyError):
            ds['ripe'].sel(sugar=12)
