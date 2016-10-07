from collections import OrderedDict
from functools import partial

from pytest import raises, mark
import numpy as np
from numpy.testing import assert_allclose
import xarray as xr

from ..generate import (
    progbar,
    combo_runner,
    combos_to_ds,
    combo_runner_to_ds,
    case_runner,
    cases_to_ds,
    case_runner_to_ds,
    find_missing_cases,
    fill_missing_cases,
)


_PARALLEL_BACKENDS = ['MULTIPROCESSING', 'DISTRIBUTED']


def foo3_scalar(a, b, c):
    assert abs(a) >= 0
    assert abs(a) < 10
    assert abs(b) >= 10
    assert abs(b) < 100
    assert abs(c) >= 100
    assert abs(c) < 1000
    return a + b + c


def foo3_float_bool(a, b, c):
    assert abs(a) >= 0
    assert abs(a) < 10
    assert abs(b) >= 10
    assert abs(b) < 100
    assert abs(c) >= 100
    assert abs(c) < 1000
    return a + b + c, a % 2 == 0


def foo2_array(a, b):
    assert abs(a) >= 0
    assert abs(a) < 10
    assert abs(b) >= 10
    assert abs(b) < 100
    return [b + a + 0.1*i for i in range(10)]


def foo2_array_bool(a, b):
    assert abs(a) >= 0
    assert abs(a) < 10
    assert abs(b) >= 10
    assert abs(b) < 100
    return [b + a + 0.1*i for i in range(10)], a % 2 == 0


def foo2_array_array(a, b):
    assert abs(a) >= 0
    assert abs(a) < 10
    assert abs(b) >= 10
    assert abs(b) < 100
    return ([b + i*a for i in range(5)],
            [b - i*a for i in range(5)])


def foo2_zarray1_zarray2(a, b):
    assert abs(a) >= 0
    assert abs(a) < 10
    assert abs(b) >= 10
    assert abs(b) < 100
    return ([b + a + 0.1j * i for i in range(5)],
            [b + a - 0.1j * i for i in range(5)])


def foo_array_input(a, t):
    return tuple(a * x for x in t)


# --------------------------------------------------------------------------- #
# COMBO_RUNNER tests                                                          #
# --------------------------------------------------------------------------- #

class TestProgbar:
    def test_normal(self):
        for i in progbar(range(10)):
            pass

    def test_overide_ascii(self):
        for i in progbar(range(10), ascii=False):
            pass


class TestComboRunner:
    def test_simple(self):
        combos = [('a', [1, 2]),
                  ('b', [10, 20, 30]),
                  ('c', [100, 200, 300, 400])]
        x = combo_runner(foo3_scalar, combos)
        xn = (np.array([1, 2]).reshape((2, 1, 1)) +
              np.array([10, 20, 30]).reshape((1, 3, 1)) +
              np.array([100, 200, 300, 400]).reshape((1, 1, 4)))
        assert_allclose(x, xn)

    def test_progbars(self):
        combos = [('a', [1, 2]),
                  ('b', [10, 20, 30]),
                  ('c', [100, 200, 300, 400])]
        combo_runner(foo3_scalar, combos, progbars=3)

    def test_dict(self):
        combos = OrderedDict((('a', [1, 2]),
                             ('b', [10, 20, 30]),
                             ('c', [100, 200, 300, 400])))
        x = combo_runner(foo3_scalar, combos)
        xn = (np.array([1, 2]).reshape((2, 1, 1)) +
              np.array([10, 20, 30]).reshape((1, 3, 1)) +
              np.array([100, 200, 300, 400]).reshape((1, 1, 4)))
        assert_allclose(x, xn)

    def test_single_combo(self):
        combos = [('a', [1, 2])]
        x = combo_runner(partial(foo3_scalar, b=20, c=300), combos)
        assert_allclose(x, [321, 322])

    def test_single_combo_single_tuple(self):
        combos = ('a', [1, 2])
        constants = {'b': 20, 'c': 300}
        x = combo_runner(foo3_scalar, combos, constants=constants)
        assert_allclose(x, [321, 322])

    def test_multires(self):
        combos = [('a', [1, 2]),
                  ('b', [10, 20, 30]),
                  ('c', [100, 200, 300, 400])]
        x, y = combo_runner(foo3_float_bool, combos, split=True)
        xn = (np.array([1, 2]).reshape((2, 1, 1)) +
              np.array([10, 20, 30]).reshape((1, 3, 1)) +
              np.array([100, 200, 300, 400]).reshape((1, 1, 4)))
        yn = (np.array([1, 2]).reshape((2, 1, 1)) %
              np.array([2]*24).reshape((2, 3, 4))) == 0
        assert_allclose(x, xn)
        assert_allclose(y, yn)

    @mark.parametrize('parallel_backend', _PARALLEL_BACKENDS)
    def test_parallel_basic(self, parallel_backend):
        combos = (('a', [1, 2]),
                  ('b', [10, 20, 30]),
                  ('c', [100, 200, 300, 400]))
        x = combo_runner(foo3_scalar, combos, processes=2,
                         parallel_backend=parallel_backend)
        xn = (np.array([1, 2]).reshape((2, 1, 1)) +
              np.array([10, 20, 30]).reshape((1, 3, 1)) +
              np.array([100, 200, 300, 400]).reshape((1, 1, 4)))
        assert_allclose(x, xn)

    @mark.parametrize('parallel_backend', _PARALLEL_BACKENDS)
    def test_parallel_multires(self, parallel_backend):
        combos = (('a', [1, 2]),
                  ('b', [10, 20, 30]),
                  ('c', [100, 200, 300, 400]))
        x = combo_runner(foo3_float_bool, combos, processes=2, split=True,
                         parallel_backend=parallel_backend)
        xn = (np.array([1, 2]).reshape((2, 1, 1)) +
              np.array([10, 20, 30]).reshape((1, 3, 1)) +
              np.array([100, 200, 300, 400]).reshape((1, 1, 4)))
        assert_allclose(x[0], xn)
        assert np.all(np.asarray(x[1])[1, ...])

    @mark.parametrize('parallel_backend', _PARALLEL_BACKENDS)
    def test_parallel_dict(self, parallel_backend):
        combos = OrderedDict((('a', [1, 2]),
                             ('b', [10, 20, 30]),
                             ('c', [100, 200, 300, 400])))
        x = [*combo_runner(foo3_scalar, combos, processes=2,
                           parallel_backend=parallel_backend)]
        xn = (np.array([1, 2]).reshape((2, 1, 1)) +
              np.array([10, 20, 30]).reshape((1, 3, 1)) +
              np.array([100, 200, 300, 400]).reshape((1, 1, 4)))
        assert_allclose(x, xn)


class TestCombosToDS:
    def test_simple(self):
        results = [1, 2, 3]
        combos = [('a', [1, 2, 3])]
        var_names = ['sum']
        ds = combos_to_ds(results, combos, var_names)
        assert ds['sum'].data.dtype == int

    def test_add_to_ds(self):
        # TODO -------------------------------------------------------------- #
        pass

    def test_add_to_ds_array(self):
        # TODO -------------------------------------------------------------- #
        pass


class TestComboRunnerToDS:
    def test_basic(self):
        combos = (('a', [1, 2]),
                  ('b', [10, 20, 30]),
                  ('c', [100, 200, 300, 400]))
        ds = combo_runner_to_ds(foo3_scalar, combos, var_names=['bananas'])
        assert ds.sel(a=2, b=30, c=400)['bananas'].data == 432

    def test_multiresult(self):
        combos = (('a', [1, 2]),
                  ('b', [10, 20, 30]),
                  ('c', [100, 200, 300, 400]))
        ds = combo_runner_to_ds(foo3_float_bool, combos,
                                var_names=['bananas', 'cakes'])
        assert ds.bananas.data.dtype == int
        assert ds.cakes.data.dtype == bool
        assert ds.sel(a=2, b=30, c=400)['bananas'].data == 432
        assert ds.sel(a=1, b=10, c=100)['bananas'].data == 111
        assert ds.sel(a=2, b=30, c=400)['cakes'].data
        assert not ds.sel(a=1, b=10, c=100)['cakes'].data

    def test_arrayresult(self):
        combos = (('a', [1, 2]),
                  ('b', [10, 20, 30]))
        ds = combo_runner_to_ds(foo2_array, combos,
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
        ds = combo_runner_to_ds(foo2_array_bool, combos,
                                var_names=['bananas', 'ripe'],
                                var_dims=(['sugar'], []),
                                var_coords={'sugar': [*range(10, 20)]})
        assert ds.ripe.data.dtype == bool
        assert ds.sel(a=2, b=30, sugar=14)['bananas'].data == 32.4
        with raises(KeyError):
            ds['ripe'].sel(sugar=12)

    def test_single_string_var_names_with_no_var_dims(self):
        combos = ('a', [1, 2, 3])
        ds = combo_runner_to_ds(foo3_scalar, combos,
                                constants={'b': 10, 'c': 100},
                                var_names='sum')
        assert_allclose(ds['sum'].data, np.array([111, 112, 113]))

    def test_double_array_return_with_same_dimensions(self):
        combos = (('a', [1, 2]),
                  ('b', [10, 20, 30]))
        ds = combo_runner_to_ds(foo2_array_array, combos,
                                var_names=['apples', 'oranges'],
                                var_dims=(['seeds'],),
                                var_coords={'seeds': [*range(5)]})
        assert ds.oranges.data.dtype == int
        assert_allclose(ds.sel(a=2, b=30).apples.data, [30, 32, 34, 36, 38])

        assert_allclose(ds.sel(a=2, b=30).oranges.data, [30, 28, 26, 24, 22])
        assert 'seeds' in ds.apples.coords
        assert 'seeds' in ds.oranges.coords

    def test_double_array_return_with_no_given_dimensions(self):
        ds = combo_runner_to_ds(foo2_array_array,
                                combos=[('a', [1, 2]),
                                        ('b', [30, 40])],
                                var_names=['array1', 'array2'],
                                var_dims=[['auto']])
        assert (ds['auto'].data.dtype == int or
                ds['auto'].data.dtype == np.int64)
        assert_allclose(ds['auto'].data, [0, 1, 2, 3, 4])

    def test_complex_output(self):
        ds = combo_runner_to_ds(foo2_zarray1_zarray2,
                                combos=[('a', [1, 2]),
                                        ('b', [30, 40])],
                                var_names=['array1', 'array2'],
                                var_dims=[['auto']])
        assert ds['array1'].data.size == 2 * 2 * 5
        assert ds['array2'].data.size == 2 * 2 * 5
        assert ds['array1'].data.dtype == complex
        assert ds['array2'].data.dtype == complex
        assert_allclose(ds['array1'].sel(a=2, b=30).data,
                        32 + np.arange(5) * 0.1j)
        assert_allclose(ds['array2'].sel(a=2, b=30).data,
                        32 - np.arange(5) * 0.1j)

    def test_constants_to_attrs(self):
        ds = combo_runner_to_ds(foo3_scalar,
                                combos=[('a', [1, 2, 3]),
                                        ('c', [100, 200, 300])],
                                constants={'b': 20},
                                var_names='x')
        assert ds.attrs['b'] == 20

    def test_const_array_to_coord(self):
        ds = combo_runner_to_ds(foo_array_input,
                                combos=[('a', [1, 2, 3])],
                                constants={'t': [10, 20, 30]},
                                var_names=['x'],
                                var_dims=[['t']])
        assert 't' in ds.coords
        assert 't' not in ds.attrs


# --------------------------------------------------------------------------- #
# CASE_RUNNER tests                                                           #
# --------------------------------------------------------------------------- #


class TestCaseRunner:
    def test_seq(self):
        cases = ((1, 10, 100),
                 (2, 20, 200),
                 (3, 30, 300))
        xs = case_runner(foo3_scalar, ('a', 'b', 'c'), cases)
        assert xs == (111, 222, 333)

    def test_progbar(self):
        cases = ((1, 10, 100),
                 (2, 20, 200),
                 (3, 30, 300))
        xs = case_runner(foo3_scalar, ('a', 'b', 'c'), cases, progbars=1)
        assert xs == (111, 222, 333)

    def test_constants(self):
        cases = ((1,),
                 (2,),
                 (3,))
        xs = case_runner(foo3_scalar, ('a', 'b', 'c'), cases,
                         constants={'b': 10, 'c': 100})
        assert xs == (111, 112, 113)

    @mark.parametrize("parallel_backend", _PARALLEL_BACKENDS)
    def test_parallel(self, parallel_backend):
        cases = ((1, 10, 100),
                 (2, 20, 200),
                 (3, 30, 300))
        xs = case_runner(foo3_scalar, ('a', 'b', 'c'), cases, processes=2,
                         parallel_backend=parallel_backend)
        assert xs == (111, 222, 333)

    def test_split(self):
        cases = ((1, 10, 100),
                 (2, 20, 200),
                 (3, 30, 300))
        a, b = case_runner(foo3_float_bool, ('a', 'b', 'c'), cases, split=True)
        assert a == (111, 222, 333)
        assert b == (False, True, False)

    def test_single_args(self):
        cases = (1, 2, 3)
        xs = case_runner(foo3_scalar, 'a', cases,
                         constants={'b': 10, 'c': 100})
        assert xs == (111, 112, 113)


class TestCasesToDS:
    def test_simple(self):
        results = ((1,), (2,), (3,), (4,), (5,))
        cases = (('a', 'a'), ('a', 'b'), ('b', 'c'), ('c', 'c'), ('b', 'a'))
        var_names = ['bananas']
        ds = cases_to_ds(results=results,
                         fn_args=['case1', 'case2'],
                         cases=cases,
                         var_names=var_names)
        assert_allclose(ds.bananas.data, [[1, 2, np.nan],
                                          [5, np.nan, 3],
                                          [np.nan, np.nan, 4]])

    def test_single_result_format(self):
        results = [1, 2, 3, 4, 5]
        cases = [('a', 'a'), ('a', 'b'), ('b', 'c'), ('c', 'c'), ('b', 'a')]
        var_names = 'bananas'
        ds = cases_to_ds(results=results,
                         fn_args=['case1', 'case2'],
                         cases=cases,
                         var_names=var_names)
        assert_allclose(ds.bananas.data, [[1, 2, np.nan],
                                          [5, np.nan, 3],
                                          [np.nan, np.nan, 4]])

    def test_single_cases_format(self):
        results = [1, 2, 3, 4, 5]
        cases = ['a', 'b', 'c', 'd', 'e']
        var_names = 'bananas'
        ds = cases_to_ds(results=results,
                         fn_args='case1',
                         cases=cases,
                         var_names=var_names)
        assert_allclose(ds.bananas.data, [1, 2, 3, 4, 5])

    def test_multires(self):
        var_names = 'lists'
        var_vals = [np.arange(10) + i for i in range(5)]
        var_dims = ['time']
        var_coords = {'time': np.arange(10) / 10}
        fn_args = 'letter'
        case_cnfgs = ['a', 'b', 'c', 'd', 'e']
        ds = cases_to_ds(results=var_vals,
                         fn_args=fn_args,
                         cases=case_cnfgs,
                         var_names=var_names,
                         var_dims=var_dims,
                         var_coords=var_coords)
        assert ds.time.data.dtype == float

    def test_add_to_ds(self):
        ds = xr.Dataset(coords={'a': [1, 2],
                                'b': [10, 20]})
        ds['x'] = (('a', 'b'), [[11, 21], [12, 0]])
        assert ds['x'].sel(a=2, b=20).data == 0
        cases_to_ds(results=[[22]],
                    fn_args=['a', 'b'],
                    cases=[[2, 20]],
                    var_names=['x'],
                    add_to_ds=ds,
                    overwrite=True)
        assert ds['x'].data.dtype == int
        assert ds['x'].sel(a=2, b=20).data == 22

    def test_add_to_ds_array(self):
        ds = xr.Dataset(coords={'a': [1, 2],
                                'b': [10, 20],
                                't': [0.1, 0.2, 0.3]})
        ds['x'] = (('a', 'b', 't'), [[[11.1, 11.2, 11.3], [21.1, 21.2, 21.3]],
                                     [[12.1, 12.2, 12.3], [0, 0, 0]]])
        assert_allclose(ds['x'].sel(a=2, b=20).data, [0, 0, 0])
        cases_to_ds(results=[[[22.1, 22.2, 22.3]]],
                    fn_args=['a', 'b'],
                    cases=[[2, 20]],
                    var_names=['x'],
                    var_dims=['t'],
                    add_to_ds=ds, overwrite=True)
        assert_allclose(ds['x'].sel(a=2, b=20).data, [22.1, 22.2, 22.3])

    def test_add_to_ds_no_overwrite(self):
        ds = xr.Dataset(coords={'a': [1, 2],
                                'b': [10, 20]})
        ds['x'] = (('a', 'b'), [[11, 21], [12, 0]])
        assert ds['x'].sel(a=2, b=20).data == 0
        with raises(ValueError):
            cases_to_ds(results=[[22]],
                        fn_args=['a', 'b'],
                        cases=[[2, 20]],
                        var_names=['x'],
                        add_to_ds=ds,
                        overwrite=False)
        ds = xr.Dataset(coords={'a': [1, 2],
                                'b': [10, 20]})
        ds['x'] = (('a', 'b'), [[11, 21], [12, None]])
        cases_to_ds(results=[[22]],
                    fn_args=['a', 'b'],
                    cases=[[2, 20]],
                    var_names=['x'],
                    add_to_ds=ds,
                    overwrite=False)
        assert ds['x'].sel(a=2, b=20).data == 22


class TestCaseRunnerToDS:
    def test_single(self):
        cases = [(1, 20, 300),
                 (3, 20, 100)]
        ds = case_runner_to_ds(foo3_scalar, ['a', 'b', 'c'], cases=cases,
                               var_names='sum')
        assert_allclose(ds['a'].data, [1, 3])
        assert_allclose(ds['b'].data, [20])
        assert_allclose(ds['c'].data, [100, 300])
        assert ds['sum'].loc[{'a': 1, 'b': 20, 'c': 300}].data == 321
        assert ds['sum'].loc[{'a': 3, 'b': 20, 'c': 100}].data == 123
        assert ds['sum'].loc[{'a': 1, 'b': 20, 'c': 100}].isnull()
        assert ds['sum'].loc[{'a': 3, 'b': 20, 'c': 300}].isnull()

    def test_multires(self):
        cases = [(1, 20, 300),
                 (3, 20, 100)]
        ds = case_runner_to_ds(foo3_float_bool,
                               fn_args=['a', 'b', 'c'],
                               cases=cases,
                               var_names=['sum', 'a_even'])
        assert_allclose(ds['a'].data, [1, 3])
        assert_allclose(ds['b'].data, [20])
        assert_allclose(ds['c'].data, [100, 300])
        assert ds['sum'].loc[{'a': 1, 'b': 20, 'c': 300}].data == 321
        assert ds['sum'].loc[{'a': 3, 'b': 20, 'c': 100}].data == 123
        assert ds['sum'].loc[{'a': 1, 'b': 20, 'c': 100}].isnull()
        assert ds['sum'].loc[{'a': 3, 'b': 20, 'c': 300}].isnull()
        assert ds['a_even'].data.dtype == object
        assert bool(ds['a_even'].sel(a=1, b=20, c=300).data) is False
        assert bool(ds['a_even'].sel(a=3, b=20, c=100).data) is False
        assert ds['a_even'].loc[{'a': 1, 'b': 20, 'c': 100}].isnull()
        assert ds['a_even'].loc[{'a': 3, 'b': 20, 'c': 300}].isnull()

    def test_array_return(self):
        ds = case_runner_to_ds(fn=foo2_array, fn_args=['a', 'b'],
                               cases=[(2, 30), (4, 50)],
                               var_names='x',
                               var_dims=['time'],
                               var_coords={'time': np.arange(10) / 10})
        assert ds.x.data.dtype == float
        assert ds.x.sel(a=2, b=50, time=0.7).isnull()
        assert ds.x.sel(a=4, b=50, time=0.3).data == 54.3

    def test_multi_array_return(self):
        ds = case_runner_to_ds(fn=foo2_array_array, fn_args=['a', 'b'],
                               cases=[(2, 30), (4, 50)],
                               var_names=['x', 'y'],
                               var_dims=[['time']],
                               var_coords={'time': ['a', 'b', 'c', 'd', 'e']})
        assert ds['time'].data.dtype != object
        assert_allclose(ds['x'].sel(a=4, b=50).data,
                        [50, 54, 58, 62, 66])
        assert_allclose(ds['y'].sel(a=4, b=50).data,
                        [50, 46, 42, 38, 34])

    def test_align_and_fillna_int(self):
        ds1 = case_runner_to_ds(foo2_array_array, fn_args=['a', 'b'],
                                cases=[(1, 10), (2, 20)],
                                var_names=['x', 'y'],
                                var_dims=[['time']],
                                var_coords={'time':
                                            ['a', 'b', 'c', 'd', 'e']})
        ds2 = case_runner_to_ds(foo2_array_array, fn_args=['a', 'b'],
                                cases=[(2, 10), (1, 20)],
                                var_names=['x', 'y'],
                                var_dims=[['time']],
                                var_coords={'time':
                                            ['a', 'b', 'c', 'd', 'e']})
        assert not np.logical_not(ds1['x'].isnull()).all()
        assert not np.logical_not(ds1['y'].isnull()).all()
        assert not np.logical_not(ds2['x'].isnull()).all()
        assert not np.logical_not(ds2['y'].isnull()).all()
        ds1, ds2 = xr.align(ds1, ds2, join='outer')
        fds = ds1.fillna(ds2)
        assert np.logical_not(fds['x'].isnull()).all()
        assert np.logical_not(fds['y'].isnull()).all()

    def test_align_and_fillna_complex(self):
        ds1 = case_runner_to_ds(foo2_zarray1_zarray2, fn_args=['a', 'b'],
                                cases=[(1j, 10), (2j, 20)],
                                var_names=['x', 'y'],
                                var_dims=[['time']],
                                var_coords={'time':
                                            ['a', 'b', 'c', 'd', 'e']})
        ds2 = case_runner_to_ds(foo2_zarray1_zarray2, fn_args=['a', 'b'],
                                cases=[(2j, 10), (1j, 20)],
                                var_names=['x', 'y'],
                                var_dims=[['time']],
                                var_coords={'time':
                                            ['a', 'b', 'c', 'd', 'e']})
        assert not np.logical_not(np.isnan(ds1['x'].data)).all()
        assert not np.logical_not(np.isnan(ds1['y'].data)).all()
        assert not np.logical_not(np.isnan(ds2['x'].data)).all()
        assert not np.logical_not(np.isnan(ds2['y'].data)).all()
        assert all(t == complex for t in (ds1.x.dtype, ds2.x.dtype,
                                          ds1.y.dtype, ds2.y.dtype))
        assert ds1.y.dtype == complex
        assert ds2.y.dtype == complex
        ds1, ds2 = xr.align(ds1, ds2, join='outer')
        fds = ds1.fillna(ds2)
        assert np.logical_not(np.isnan(fds['x'].data)).all()
        assert np.logical_not(np.isnan(fds['y'].data)).all()


# --------------------------------------------------------------------------- #
# Finding and filling missing data                                            #
# --------------------------------------------------------------------------- #

class TestFindMissingCases:
    def test_simple(self):
        ds = xr.Dataset(coords={'a': [1, 2, 3], 'b': [40, 50]})
        ds['x'] = (('a', 'b'), np.array([[0.1, np.nan],
                                         [np.nan, 0.2],
                                         [np.nan, np.nan]]))
        # Target cases and settings
        t_cases = ((1, 50), (2, 40), (3, 40), (3, 50))
        t_configs = tuple(dict(zip(['a', 'b'], t_case)) for t_case in t_cases)
        # Missing cases and settings
        m_args, m_cases = find_missing_cases(ds)
        m_configs = tuple(dict(zip(m_args, m_case)) for m_case in m_cases)
        # Assert same set of coordinates
        assert all(t_config in m_configs for t_config in t_configs)
        assert all(m_config in t_configs for m_config in m_configs)

    def test_multires(self):
        ds = xr.Dataset(coords={'a': [1, 2, 3], 'b': [40, 50]})
        ds['x'] = (('a', 'b'), np.array([[0.1, np.nan],
                                         [np.nan, 0.2],
                                         [np.nan, np.nan]]))
        ds['y'] = (('a', 'b'), np.array([['a', None],
                                         [None, 'b'],
                                         [None, None]]))
        # Target cases and settings
        t_cases = ((1, 50), (2, 40), (3, 40), (3, 50))
        t_configs = tuple(dict(zip(['a', 'b'], t_case)) for t_case in t_cases)
        # Missing cases and settings
        m_args, m_cases = find_missing_cases(ds)
        m_configs = tuple(dict(zip(m_args, m_case)) for m_case in m_cases)
        # Assert same set of coordinates
        assert set(m_args) == {'a', 'b'}
        assert all(t_config in m_configs for t_config in t_configs)
        assert all(m_config in t_configs for m_config in m_configs)

    def test_var_dims_leave(self):
        ds = xr.Dataset(coords={'a': [1, 2, 3], 'b': [40, 50],
                                't': [0.1, 0.2, 0.3]})
        ds['x'] = (('a', 'b'), np.array([[0.1, np.nan],
                                         [np.nan, 0.2],
                                         [np.nan, np.nan]]))
        ds['y'] = (('a', 'b', 't'), np.array([[[0.2]*3, [np.nan]*3],
                                              [[np.nan]*3, [0.4]*3],
                                              [[np.nan]*3, [np.nan]*3]]))
        # Target cases and settings
        t_cases = ((1, 50), (2, 40), (3, 40), (3, 50))
        t_configs = tuple(dict(zip(['a', 'b'], t_case)) for t_case in t_cases)
        # Missing cases and settings
        m_args, m_cases = find_missing_cases(ds, var_dims='t')
        m_configs = tuple(dict(zip(m_args, m_case)) for m_case in m_cases)
        # Assert same set of coordinates
        assert set(m_args) == {'a', 'b'}
        assert all(t_config in m_configs for t_config in t_configs)
        assert all(m_config in t_configs for m_config in m_configs)


class TestFillMissingCases:
    def test_simple(self):
        ds = xr.Dataset(coords={'a': [1, 2, 3], 'b': [40, 50]})
        ds['x'] = (('a', 'b'), np.array([[641, np.nan],
                                         [np.nan, 652],
                                         [np.nan, np.nan]]))
        fill_missing_cases(ds, fn=foo3_scalar, constants={'c': 600},
                           var_names='x')
        assert_allclose(ds.x.data, [[641, 651], [642, 652], [643, 653]])

    def test_multires(self):
        ds = xr.Dataset(coords={'a': [1, 2, 3], 'b': [40, 50]})
        ds['x'] = (('a', 'b'), np.array([[641, np.nan],
                                         [np.nan, 652],
                                         [np.nan, np.nan]]))
        ds['even'] = (('a', 'b'), np.array([[False, None],
                                            [None, True],
                                            [None, None]]))
        fill_missing_cases(ds, fn=foo3_float_bool, constants={'c': 600},
                           var_names=['x', 'even'])
        assert_allclose(ds.x.data, [[641, 651],
                                    [642, 652],
                                    [643, 653]])
        assert(ds.even.data.tolist() == [[False, False],
                                         [True, True],
                                         [False, False]])

    def test_array_return(self):
        ds = xr.Dataset(coords={'a': [1, 2, 3], 'b': [40, 50],
                                't': [0.0, 0.1, 0.2, 0.3, 0.4]})
        ds['z1'] = (('a', 'b', 't'),
                    np.array([[41 + 0.1j*np.arange(5), [np.nan]*5],
                              [[np.nan]*5, 52 + 0.1j*np.arange(5)],
                              [[np.nan]*5, [np.nan]*5]]))
        ds['z2'] = (('a', 'b', 't'),
                    np.array([[41 - 0.1j*np.arange(5), [np.nan]*5],
                              [[np.nan]*5, 52 - 0.1j*np.arange(5)],
                              [[np.nan]*5, [np.nan]*5]]))
        settings = {'var_names': ['z1', 'z2'], 'var_dims': 't',
                    'var_coords': {'t': [0.0, 0.1, 0.2, 0.3, 0.4]}}
        fill_missing_cases(ds, foo2_zarray1_zarray2, **settings)
        dst = combo_runner_to_ds(foo2_zarray1_zarray2,
                                 combos=[('a', [1, 2, 3]),
                                         ('b', [40, 50])], **settings)
        assert dst.equals(ds)

    def test_float_coords(self):
        ds = xr.Dataset(coords={'a': [1.1, 2.1, 3.1], 'b': [40.01, 50.01]})
        ds['x'] = (('a', 'b'), np.array([[641.11, np.nan],
                                         [np.nan, 652.11],
                                         [np.nan, np.nan]]))
        fill_missing_cases(ds, fn=foo3_scalar, constants={'c': 600},
                           var_names='x')
        assert_allclose(ds.x.data, [[641.11, 651.11],
                                    [642.11, 652.11],
                                    [643.11, 653.11]])
