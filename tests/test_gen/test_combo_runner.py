from collections import OrderedDict
from functools import partial
import pytest
import numpy as np
from numpy.testing import assert_allclose
import xarray as xr

from xyzpy.gen.combo_runner import (
    combo_runner,
    _combos_to_ds,
    combo_runner_to_ds,
)
from . import (
    foo3_scalar,
    foo3_float_bool,
    foo2_array,
    foo2_array_bool,
    foo2_array_array,
    foo2_zarray1_zarray2,
    foo_array_input,
)


# --------------------------------------------------------------------------- #
# COMBO_RUNNER tests                                                          #
# --------------------------------------------------------------------------- #

_test_combos1 = (('a', [1, 2]),
                 ('b', [10, 20, 30]),
                 ('c', [100, 200, 300, 400]))

_test_expect1 = (np.array([1, 2]).reshape((2, 1, 1)) +
                 np.array([10, 20, 30]).reshape((1, 3, 1)) +
                 np.array([100, 200, 300, 400]).reshape((1, 1, 4)))


class TestComboRunner:
    def test_simple(self):
        x = combo_runner(foo3_scalar, _test_combos1)
        assert_allclose(x, _test_expect1)

    def test_progbars(self):
        combo_runner(foo3_scalar, _test_combos1, verbosity=2)

    def test_dict(self):
        combos = OrderedDict(_test_combos1)
        x = combo_runner(foo3_scalar, combos)
        assert_allclose(x, _test_expect1)

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
        combos = _test_combos1
        x, y = combo_runner(foo3_float_bool, combos, split=True)
        xn = _test_expect1
        yn = (np.array([1, 2]).reshape((2, 1, 1)) %
              np.array([2] * 24).reshape((2, 3, 4))) == 0
        assert_allclose(x, xn)
        assert_allclose(y, yn)

    @pytest.mark.parametrize('parallel', [False, True])
    @pytest.mark.parametrize('fn', (foo3_scalar,))
    def test_parallel_basic(self, parallel, fn):
        x = combo_runner(fn, _test_combos1, num_workers=2, parallel=parallel)
        assert_allclose(x, _test_expect1)

    @pytest.mark.parametrize('executor', ['cf-process', 'cf-thread',
                                          'mp-process', 'mp-thread'])
    @pytest.mark.parametrize('fn', (foo3_scalar,))
    def test_executor_basic(self, executor, fn):
        import concurrent.futures as cf
        import multiprocessing as mp
        executor = {
            'cf-process': cf.ProcessPoolExecutor,
            'cf-thread': cf.ThreadPoolExecutor,
            'mp-process': mp.Pool,
            'mp-thread': mp.pool.ThreadPool,
        }[executor](2)
        x = combo_runner(fn, _test_combos1, executor=executor)
        assert_allclose(x, _test_expect1)

    @pytest.mark.parametrize('parallel', [False, True])
    def test_parallel_multires(self, parallel):
        x = combo_runner(foo3_float_bool, _test_combos1, num_workers=2,
                         split=True, parallel=parallel)
        assert_allclose(x[0], _test_expect1)
        assert np.all(np.asarray(x[1])[1, ...])

    @pytest.mark.parametrize('parallel', [False, True])
    def test_parallel_dict(self, parallel):
        combos = OrderedDict(_test_combos1)
        x = [*combo_runner(foo3_scalar, combos, num_workers=2,
                           parallel=parallel, verbosity=2)]
        assert_allclose(x, _test_expect1)


class TestCombosToDS:
    def test_simple(self):
        results = [1, 2, 3]
        combos = [('a', [1, 2, 3])]
        var_names = ['sum']
        ds = _combos_to_ds(results, combos, var_names,
                           var_dims={'sum': ()}, var_coords={})
        assert ds['sum'].data.dtype == int

    def test_add_to_ds(self):
        # TODO -------------------------------------------------------------- #
        pass

    def test_add_to_ds_array(self):
        # TODO -------------------------------------------------------------- #
        pass


class TestComboRunnerToDS:
    def test_basic(self):
        combos = _test_combos1
        ds = combo_runner_to_ds(foo3_scalar, combos, var_names=['bananas'])
        assert ds.sel(a=2, b=30, c=400)['bananas'].data == 432

    def test_multiresult(self):
        ds = combo_runner_to_ds(foo3_float_bool, _test_combos1,
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
                                var_dims={'bananas': ['sugar']},
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
        with pytest.raises((KeyError, ValueError)):
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
                                var_dims={('apples', 'oranges'): ['seeds']},
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
                                var_dims=[['auto'], ['auto']])
        assert (ds['auto'].data.dtype == int or
                ds['auto'].data.dtype == np.int64)
        assert_allclose(ds['auto'].data, [0, 1, 2, 3, 4])

    def test_complex_output(self):
        ds = combo_runner_to_ds(foo2_zarray1_zarray2,
                                combos=[('a', [1, 2]),
                                        ('b', [30, 40])],
                                var_names=['array1', 'array2'],
                                var_dims=[['auto'], ['auto']])
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
        assert 't' in ds.dims
        assert 't' not in ds.attrs

    def test_when_results_are_xobjs(self):

        def fn_ds(a, b):
            ds = xr.Dataset(
                coords={
                    'x': [1, 2],
                    'y': ['foo', 'bar', 'baz'],
                },
                data_vars={
                    'apples': (['x', 'y'], np.tile(10 * a + b, (2, 3))),
                    'lemons': ('y', np.tile(10 * a + b, (3,))),
                }
            )
            return ds

        fds = combo_runner_to_ds(
            fn_ds,
            combos={
                'a': [2, 3, 4, 5],
                'b': [6, 7, 8, 9, 10],
            }, var_names=None)

        assert fds.sel(a=4, b=9, x=2, y='bar')['apples'].values == 49
        assert fds.sel(a=5, b=7, y='bar')['lemons'].values == 57
        assert 'output' not in fds
