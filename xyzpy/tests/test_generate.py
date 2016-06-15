from collections import OrderedDict
from functools import partial

from pytest import raises
import numpy as np
from numpy.testing import assert_allclose

from ..generate import (
    progbar,
    combo_runner,
    combo_runner_to_ds,
    case_runner,
    cases_to_ds
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


class TestProgbar:
    def test_normal(self):
        for i in progbar(range(10)):
            pass

    def test_overide_ascii(self):
        for i in progbar(range(10), ascii=False):
            pass


class TestcomboRunner:
    def test_simple(self):
        combos = [('a', [1, 2]),
                  ('b', [10, 20, 30]),
                  ('c', [100, 200, 300, 400])]
        x = combo_runner(foo1, combos)
        xn = (np.array([1, 2]).reshape((2, 1, 1)) +
              np.array([10, 20, 30]).reshape((1, 3, 1)) +
              np.array([100, 200, 300, 400]).reshape((1, 1, 4)))
        assert_allclose(x, xn)

    def test_progbars(self):
        combos = [('a', [1, 2]),
                  ('b', [10, 20, 30]),
                  ('c', [100, 200, 300, 400])]
        combo_runner(foo1, combos, progbars=3)

    def test_dict(self):
        combos = OrderedDict((('a', [1, 2]),
                             ('b', [10, 20, 30]),
                             ('c', [100, 200, 300, 400])))
        x = combo_runner(foo1, combos)
        xn = (np.array([1, 2]).reshape((2, 1, 1)) +
              np.array([10, 20, 30]).reshape((1, 3, 1)) +
              np.array([100, 200, 300, 400]).reshape((1, 1, 4)))
        assert_allclose(x, xn)

    def test_single_combo(self):
        combos = [('a', [1, 2])]
        x = combo_runner(partial(foo1, b=20, c=300), combos)
        assert_allclose(x, [321, 322])

    def test_single_combo_single_tuple(self):
        combos = ('a', [1, 2])
        constants = {'b': 20, 'c': 300}
        x = combo_runner(foo1, combos, constants=constants)
        assert_allclose(x, [321, 322])

    def test_multires(self):
        combos = [('a', [1, 2]),
                  ('b', [10, 20, 30]),
                  ('c', [100, 200, 300, 400])]
        x, y = combo_runner(foo2, combos, split=True)
        xn = (np.array([1, 2]).reshape((2, 1, 1)) +
              np.array([10, 20, 30]).reshape((1, 3, 1)) +
              np.array([100, 200, 300, 400]).reshape((1, 1, 4)))
        yn = (np.array([1, 2]).reshape((2, 1, 1)) %
              np.array([2]*24).reshape((2, 3, 4))) == 0
        assert_allclose(x, xn)
        assert_allclose(y, yn)


class TestParallelcomboRunner:
    def test_basic(self):
        combos = (('a', [1, 2]),
                  ('b', [10, 20, 30]),
                  ('c', [100, 200, 300, 400]))
        x = combo_runner(foo1, combos, processes=2)
        xn = (np.array([1, 2]).reshape((2, 1, 1)) +
              np.array([10, 20, 30]).reshape((1, 3, 1)) +
              np.array([100, 200, 300, 400]).reshape((1, 1, 4)))
        assert_allclose(x, xn)

    def test_multires(self):
        combos = (('a', [1, 2]),
                  ('b', [10, 20, 30]),
                  ('c', [100, 200, 300, 400]))
        x = combo_runner(foo2, combos, processes=2, split=True)
        xn = (np.array([1, 2]).reshape((2, 1, 1)) +
              np.array([10, 20, 30]).reshape((1, 3, 1)) +
              np.array([100, 200, 300, 400]).reshape((1, 1, 4)))
        assert_allclose(x[0], xn)
        assert np.all(np.asarray(x[1])[1, ...])

    def test_dict(self):
        combos = OrderedDict((('a', [1, 2]),
                             ('b', [10, 20, 30]),
                             ('c', [100, 200, 300, 400])))
        x = [*combo_runner(foo1, combos, processes=2)]
        xn = (np.array([1, 2]).reshape((2, 1, 1)) +
              np.array([10, 20, 30]).reshape((1, 3, 1)) +
              np.array([100, 200, 300, 400]).reshape((1, 1, 4)))
        assert_allclose(x, xn)


class TestXRcomboRunner:
    def test_basic(self):
        combos = (('a', [1, 2]),
                  ('b', [10, 20, 30]),
                  ('c', [100, 200, 300, 400]))
        ds = combo_runner_to_ds(foo1, combos, var_names=['bananas'])
        assert ds.sel(a=2, b=30, c=400)['bananas'].data == 432

    def test_multiresult(self):
        combos = (('a', [1, 2]),
                  ('b', [10, 20, 30]),
                  ('c', [100, 200, 300, 400]))
        ds = combo_runner_to_ds(foo2, combos, var_names=['bananas', 'cakes'])
        assert ds.bananas.data.dtype == int
        assert ds.cakes.data.dtype == bool
        assert ds.sel(a=2, b=30, c=400)['bananas'].data == 432
        assert ds.sel(a=1, b=10, c=100)['bananas'].data == 111
        assert ds.sel(a=2, b=30, c=400)['cakes'].data
        assert not ds.sel(a=1, b=10, c=100)['cakes'].data

    def test_arrayresult(self):
        combos = (('a', [1, 2]),
                  ('b', [10, 20, 30]))
        ds = combo_runner_to_ds(foo_t, combos,
                                var_names='bananas',
                                var_dims=(['sugar']),
                                var_coords={'sugar': [*range(10)]})
        assert ds.bananas.data.dtype == float
        assert_allclose(ds.sel(a=2, b=30)['bananas'].data,
                        [32.0, 32.1, 32.2, 32.3, 32.4,
                         32.5, 32.6, 32.7, 32.8, 32.9])

    def test_array_and_single_result(self):
        combos = (('a', [1, 2]),
                  ('b', [10, 20, 30]))
        ds = combo_runner_to_ds(foo_t2, combos,
                                var_names=['bananas', 'ripe'],
                                var_dims=(['sugar'], []),
                                var_coords={'sugar': [*range(10, 20)]})
        assert ds.ripe.data.dtype == bool
        assert ds.sel(a=2, b=30, sugar=14)['bananas'].data == 32.4
        with raises(KeyError):
            ds['ripe'].sel(sugar=12)

    def test_double_array_return_with_same_dimensions(self):
        def foo(a, b):
            return [b + i*a for i in range(5)], [b - i*a for i in range(5)]
        combos = (('a', [1, 2]),
                  ('b', [10, 20, 30]))
        ds = combo_runner_to_ds(foo, combos,
                                var_names=['apples', 'oranges'],
                                var_dims=(['seeds'],),
                                var_coords={'seeds': [*range(5)]})
        assert ds.oranges.data.dtype == int
        assert_allclose(ds.sel(a=2, b=30).apples.data, [30, 32, 34, 36, 38])
        assert_allclose(ds.sel(a=2, b=30).oranges.data, [30, 28, 26, 24, 22])
        assert 'seeds' in ds.apples.coords
        assert 'seeds' in ds.oranges.coords

        # TODO: test single string var_names and no given var_dims


class TestConfigRunner:
    def test_seq(self):
        configs = ((1, 10, 100),
                   (2, 20, 200),
                   (3, 30, 300))
        xs = case_runner(foo1, ('a', 'b', 'c'), configs)
        assert xs == (111, 222, 333)

    def test_progbar(self):
        configs = ((1, 10, 100),
                   (2, 20, 200),
                   (3, 30, 300))
        xs = case_runner(foo1, ('a', 'b', 'c'), configs, progbars=1)
        assert xs == (111, 222, 333)

    def test_constants(self):
        configs = ((1,),
                   (2,),
                   (3,))
        xs = case_runner(foo1, ('a', 'b', 'c'), configs,
                         constants={'b': 10, 'c': 100})
        assert xs == (111, 112, 113)

    def test_parallel(self):
        configs = ((1, 10, 100),
                   (2, 20, 200),
                   (3, 30, 300))
        xs = case_runner(foo1, ('a', 'b', 'c'), configs, processes=2)
        assert xs == (111, 222, 333)

    def test_split(self):
        configs = ((1, 10, 100),
                   (2, 20, 200),
                   (3, 30, 300))
        a, b = case_runner(foo2, ('a', 'b', 'c'), configs, split=True)
        assert a == (111, 222, 333)
        assert b == (False, True, False)

    def test_single_args(self):
        configs = (1, 2, 3)
        xs = case_runner(foo1, 'a', configs,
                         constants={'b': 10, 'c': 100})
        assert xs == (111, 112, 113)


class TestResConfigsToDS:
    def test_simple(self):
        results = ((1,), (2,), (3,), (4,), (5,))
        configs = (('a', 'a'), ('a', 'b'), ('b', 'c'), ('c', 'c'), ('b', 'a'))
        var_names = ['bananas']
        ds = cases_to_ds(results=results,
                         configs=configs,
                         case_names=['case1', 'case2'],
                         var_names=var_names)
        assert_allclose(ds.bananas.data, [[1, 2, np.nan],
                                          [5, np.nan, 3],
                                          [np.nan, np.nan, 4]])

    def test_single_result_format(self):
        results = [1, 2, 3, 4, 5]
        configs = [('a', 'a'), ('a', 'b'), ('b', 'c'), ('c', 'c'), ('b', 'a')]
        var_names = 'bananas'
        ds = cases_to_ds(results=results,
                         configs=configs,
                         case_names=['case1', 'case2'],
                         var_names=var_names)
        assert_allclose(ds.bananas.data, [[1, 2, np.nan],
                                          [5, np.nan, 3],
                                          [np.nan, np.nan, 4]])

    def test_single_cases_format(self):
        results = [1, 2, 3, 4, 5]
        configs = ['a', 'b', 'c', 'd', 'e']
        var_names = 'bananas'
        ds = cases_to_ds(results=results,
                         configs=configs,
                         case_names='case1',
                         var_names=var_names)
        assert_allclose(ds.bananas.data, [1, 2, 3, 4, 5])

    def test_multires(self):
        var_names = 'lists'
        var_vals = [np.arange(10) + i for i in range(5)]
        var_dims = ['time']
        var_coords = {'time': np.arange(10) / 10}
        case_names = 'letter'
        case_cnfgs = ['a', 'b', 'c', 'd', 'e']

        ds = cases_to_ds(results=var_vals,
                         configs=case_cnfgs,
                         case_names=case_names,
                         var_names=var_names,
                         var_dims=var_dims,
                         var_coords=var_coords)

        assert ds.time.data.dtype == float
