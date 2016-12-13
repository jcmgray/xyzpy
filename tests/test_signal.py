import pytest
import numpy as np
from numpy.testing import assert_allclose
import xarray as xr

from xyzpy import Runner
from xyzpy.signal import nan_wrap


# ------------------------------ fixtures ----------------------------------- #

@pytest.fixture
def ds():
    ts = np.linspace(0, 1, 21)
    fs = np.arange(1, 3)

    cs = np.empty((fs.size, ts.size))
    sn = np.empty(ts.size)

    sn = np.sin(ts)
    for i, f in enumerate(fs):
        cs[i, :] = np.cos(f * ts)

    ds = xr.Dataset(coords={'t': ts, 'f': fs},
                    data_vars={'cos': (('f', 't'), cs),
                               'sin': ('t', sn)})

    return ds


@pytest.fixture
def eds():
    ts = np.linspace(0, 1, 100)
    fs = np.arange(1, 3)

    cs = np.empty((fs.size, ts.size))
    sn = np.empty((fs.size, ts.size))

    for i, f in enumerate(fs):
        cs[i, :] = np.cos(f * ts)
        sn[i, :] = np.sin(f * ts)

    ds = xr.Dataset(coords={'t': ts,
                            'f': fs},
                    data_vars={'cos': (('f', 't'), cs),
                               'sin': (('f', 't'), sn)})

    return ds


@pytest.fixture
def nan_ds():
    def make_data(a, b):
        return 10. * a + b, a + 10. * b

    r = Runner(make_data, ('a10sum', 'b10sum'))

    ds1 = r.run_combos((('a', [1, 2, 3]),
                        ('b', [1, 2, 3])))

    ds2 = r.run_combos((('a', [2, 3, 4]),
                        ('b', [1, 2, 3])))

    ds3 = r.run_combos((('a', [4, 5, 6]),
                        ('b', [4, 5, 6])))

    return xr.merge([ds1, ds2, ds3])


# -------------------------------- tests ------------------------------------ #

class TestNanWrap:
    def test_no_nan(self):

        def tfoo(a, b):
            return a + b

        ntfoo = nan_wrap(tfoo)

        a = np.array([1, 2, 3])
        b = np.array([10, 20, 30])
        assert_allclose(ntfoo(a, b), [11, 22, 33])

        a = np.array([np.nan, 2, np.nan])
        b = np.array([10, 20, 30])
        assert_allclose(ntfoo(a, b), [np.nan, 22, np.nan])

        a = np.array([np.nan, np.nan, np.nan])
        b = np.array([10, 20, 30])
        assert_allclose(ntfoo(a, b), [np.nan, np.nan, np.nan])


class TestFornberg:
    def test_order0(self, ds, eds):
        cds = ds.fdiff('t', order=0)

        assert_allclose(cds['cos'].values,
                        eds['cos'].values,
                        atol=1e-6, rtol=1e-6)
        assert_allclose(cds['sin'].values,
                        eds['sin'].sel(f=1).values,
                        atol=1e-6, rtol=1e-6)

    def test_order1(self, ds, eds):
        cds = ds.fdiff('t', order=1)

        for f in range(1, 3):
            assert_allclose(cds['cos'].sel(f=f).values,
                            - f * eds['sin'].sel(f=f).values,
                            atol=1e-4, rtol=1e-3)

    def test_order2(self, ds, eds):
        cds = ds.fdiff('t', order=2)

        for f in range(1, 3):
            assert_allclose(cds['cos'].sel(f=f).values,
                            - f**2 * eds['cos'].sel(f=f).values,
                            atol=1e-3, rtol=1e-2)

    def test_nan(self, nan_ds):
        nan_ds.fdiff('a')
        nan_ds.fdiff('b')
