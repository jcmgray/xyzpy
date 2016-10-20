import pytest
import numpy as np
import xarray as xr

from ...gen.farming import Runner


# Fixtures ------------------------------------------------------------------ #

def fn1_x(x):
    return x ** 2


def fn3_fba(a, b, c):
    sm = a + b + c
    ev = ((a + b + c) % 2 == 0)
    ts = a * (b * np.linspace(0, 1.0, 3) + c)
    return sm, ev, ts


# Tests --------------------------------------------------------------------- #

class TestRunner:
    def test_runner_simple(self):
        r = Runner(fn3_fba, var_names=['sum', 'even', 'array'],
                   var_dims={'array': ['time']},
                   var_coords={'time': np.linspace(0, 1.0, 3)},
                   constants={'c': 100},
                   attrs={'fruit': 'apples'})
        print(r)

    @pytest.mark.parametrize("var_names", ['sq', ('sq',), ['sq']])
    @pytest.mark.parametrize("var_dims", [[], (), {}, {'sq': ()}])
    @pytest.mark.parametrize("combos", [
        ('x', (1, 2, 3)),
        (('x', (1, 2, 3)),),
        ('x', [1, 2, 3]),
        [('x', [1, 2, 3])],
        {'x': (1, 2, 3)},
        {'x': [1, 2, 3]},
    ])
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
        r.run_combos((('a', (1, 2)),
                      ('b', (3, 4))))
        expected_ds = xr.Dataset(
            coords={'a': [1, 2], 'b': [3, 4], 'time': np.linspace(0, 1.0, 3)},
            data_vars={'sum': (('a', 'b'), [[104, 105],
                                            [105, 106]]),
                       'even': (('a', 'b'), [[True, False],
                                             [False, True]]),
                       'array': (('a', 'b', 'time'),
                                 [[[100, 101.5, 103], [100, 102, 104]],
                                  [[200, 203, 206],   [200, 204, 208]]])},
            attrs={'c': 100, 'fruit': 'apples'})
        assert r.last_ds.identical(expected_ds)

    def test_cases_parse(self):
        # TODO
        pass

    def test_runner_cases(self):
        # TODO
        pass


class TestHarvester:
    pass
