import os
import tempfile

import pytest
import numpy as np
import xarray as xr

from xyzpy.manage import load_ds
from xyzpy.gen.farming import Runner, Harvester


# -------------------------------- Fixtures --------------------------------- #

def fn1_x(x):
    return x ** 2


def fn3_fba(a, b, c):
    sm = a + b + c
    ev = ((a + b + c) % 2 == 0)
    ts = a * (b * np.linspace(0, 1.0, 3) + c)
    return sm, int(ev), ts


# ------------------------------ Runner Tests ------------------------------- #

class TestRunner:

    def test_runner_init(self):
        r = Runner(fn3_fba, fn_args=('a', 'b', 'c'),
                   var_names=['sum', 'even', 'array'],
                   var_dims={'array': ['time']},
                   var_coords={'time': np.linspace(0, 1.0, 3)},
                   constants={'c': 100},
                   attrs={'fruit': 'apples'})
        print(r)

    @pytest.mark.parametrize("prop, vals", [('var_names', ['S', 'E', 'A']),
                                            ('fn_args', ['A', 'B', 'C']),
                                            ('var_dims', {'array': ['t']}),
                                            ('var_coords', {'time': [1, 2]}),
                                            ('constants', [('c', 200)]),
                                            ('resources', [('c', 300)])])
    def test_properties(self, prop, vals):
        r = Runner(fn3_fba,
                   fn_args=('a', 'b', 'c'),
                   var_names=['sum', 'even', 'array'],
                   var_dims={'array': ['time']},
                   var_coords={'time': np.linspace(0, 1.0, 3)},
                   constants=[('c', 100)],
                   attrs={'fruit': 'apples'})
        getattr(r, prop)
        setattr(r, prop, vals)
        delattr(r, prop)
        assert getattr(r, prop) is None

    @pytest.mark.parametrize("var_names", ['sq', ('sq',), ['sq']])
    @pytest.mark.parametrize("var_dims", [[], (), {}, {'sq': ()}])
    @pytest.mark.parametrize("combos", [('x', (1, 2, 3)),
                                        (('x', (1, 2, 3)),),
                                        ('x', [1, 2, 3]),
                                        [('x', [1, 2, 3])],
                                        {'x': (1, 2, 3)},
                                        {'x': [1, 2, 3]}])
    def test_runner_combos_parse(self, var_names, var_dims, combos):
        r = Runner(fn1_x, var_names=var_names, var_dims=var_dims)
        r.run_combos(combos)
        expected_ds = xr.Dataset(coords={'x': [1, 2, 3]},
                                 data_vars={'sq': ('x', [1, 4, 9])})
        assert r.last_ds.identical(expected_ds)

    def test_runner_combos(self):
        r = Runner(fn3_fba, var_names=['sum', 'even', 'array'],
                   var_dims={'array': ['time']},
                   var_coords={'time': np.linspace(0, 1.0, 3)},
                   constants={'c': 100},
                   attrs={'fruit': 'apples'})
        r.run_combos((('a', (1, 2)), ('b', (3, 4))))
        expected_ds = xr.Dataset(
            coords={'a': [1, 2], 'b': [3, 4], 'time': np.linspace(0, 1.0, 3)},
            data_vars={'sum': (('a', 'b'), [[104, 105],
                                            [105, 106]]),
                       'even': (('a', 'b'), [[True, False],
                                             [False, True]]),
                       'array': (('a', 'b', 'time'),
                                 [[[100, 101.5, 103], [100, 102, 104]],
                                  [[200, 203, 206], [200, 204, 208]]])},
            attrs={'c': 100, 'fruit': 'apples'})
        assert r.last_ds.identical(expected_ds)

    @pytest.mark.parametrize("cases", [(1, 2, 3),
                                       ([1], [2], [3])])
    def test_runner_cases_parse(self, cases):
        r = Runner(fn1_x, var_names='sq', fn_args='x')
        r.run_cases(cases)
        expected_ds = xr.Dataset(coords={'x': [1, 2, 3]},
                                 data_vars={'sq': ('x', [1, 4, 9])})
        assert r.last_ds.identical(expected_ds)

    def test_runner_cases(self):
        r = Runner(fn3_fba, fn_args=['a', 'b'],
                   var_names=['sum', 'even', 'array'],
                   var_dims={'array': ['time']},
                   var_coords={'time': np.linspace(0, 1.0, 3)},
                   constants={'c': 100},
                   attrs={'fruit': 'apples'})
        r.run_cases([(2, 3), (1, 4)])
        expected_ds = xr.Dataset(
            coords={'a': [1, 2], 'b': [3, 4], 'time': np.linspace(0, 1.0, 3)},
            data_vars={'sum': (('a', 'b'), [[np.nan, 105],
                                            [105, np.nan]]),
                       'even': (('a', 'b'), [[None, False],
                                             [False, None]]),
                       'array': (('a', 'b', 'time'),
                                 [[[np.nan, np.nan, np.nan],
                                   [100, 102, 104]],
                                  [[200, 203, 206],
                                   [np.nan, np.nan, np.nan]]])},
            attrs={'c': 100, 'fruit': 'apples'})
        assert r.last_ds.identical(expected_ds)


# -------------------------- Harvester Tests -------------------------------- #

class TestHarvester:
    def test_harvest_combos_new(self):
        r = Runner(fn3_fba, var_names=['sum', 'even', 'array'],
                   var_dims={'array': ['time']},
                   var_coords={'time': np.linspace(0, 1.0, 3)},
                   constants={'c': 100},
                   attrs={'fruit': 'apples'})
        expected_ds = xr.Dataset(
            coords={'a': [1, 2], 'b': [3, 4], 'time': np.linspace(0, 1.0, 3)},
            data_vars={'sum': (('a', 'b'), [[104, 105],
                                            [105, 106]]),
                       'even': (('a', 'b'), [[True, False],
                                             [False, True]]),
                       'array': (('a', 'b', 'time'),
                                 [[[100, 101.5, 103], [100, 102, 104]],
                                  [[200, 203, 206], [200, 204, 208]]])},
            attrs={'c': 100, 'fruit': 'apples'})
        with tempfile.TemporaryDirectory() as tmpdir:
            fl_pth = os.path.join(tmpdir, 'test.h5')
            h = Harvester(r, fl_pth)
            h.harvest_combos((('a', (1, 2)), ('b', (3, 4))))
            hds = load_ds(fl_pth)
        assert h.last_ds.identical(expected_ds)
        assert h.full_ds.identical(expected_ds)
        assert hds.identical(expected_ds)

    def test_harvest_combos_merge(self):
        r = Runner(fn3_fba, var_names=['sum', 'even', 'array'],
                   var_dims={'array': ['time']},
                   var_coords={'time': np.linspace(0, 1.0, 3)},
                   constants={'c': 100},
                   attrs={'fruit': 'apples'})
        expected_ds = xr.Dataset(
            coords={'a': [1, 2], 'b': [3, 4], 'time': np.linspace(0, 1.0, 3)},
            data_vars={'sum': (('a', 'b'), [[104, 105],
                                            [105, 106]]),
                       'even': (('a', 'b'), [[True, False],
                                             [False, True]]),
                       'array': (('a', 'b', 'time'),
                                 [[[100, 101.5, 103], [100, 102, 104]],
                                  [[200, 203, 206], [200, 204, 208]]])},
            attrs={'c': 100, 'fruit': 'apples'})
        with tempfile.TemporaryDirectory() as tmpdir:
            fl_pth = os.path.join(tmpdir, 'test.h5')
            h = Harvester(r, fl_pth)
            h.harvest_combos((('a', (1,)), ('b', (3, 4))))
            h.harvest_combos((('a', (2,)), ('b', (3, 4))))
            hds = load_ds(fl_pth)
        assert not h.last_ds.identical(expected_ds)
        assert h.full_ds.identical(expected_ds)
        assert hds.identical(expected_ds)

    def test_harvest_combos(self):
        # TODO
        pass

    def test_harvest_cases(self):
        # TODO
        pass
