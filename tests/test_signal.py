import pytest
import numpy as np
from numpy.testing import assert_allclose
import xarray as xr

from xyzpy import Runner
from xyzpy.signal import (
    _broadcast_filtfilt_butter,
    _broadcast_filtfilt_bessel,
)


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
    return xr.Dataset(coords={'t': ts, 'f': fs},
                      data_vars={'cos': (('f', 't'), cs),
                                 'sin': ('t', sn)})


@pytest.fixture
def eds():
    ts = np.linspace(0, 1, 100)
    fs = np.arange(1, 3)
    cs = np.empty((fs.size, ts.size))
    sn = np.empty((fs.size, ts.size))
    for i, f in enumerate(fs):
        cs[i, :] = np.cos(f * ts)
        sn[i, :] = np.sin(f * ts)
    return xr.Dataset(coords={'t': ts,
                              'f': fs},
                      data_vars={'cos': (('f', 't'), cs),
                                 'sin': (('f', 't'), sn)})


@pytest.fixture
def nan_ds():
    def make_data(a, b):
        return 10. * a + b, a + 10. * b
    r = Runner(make_data, ('a10sum', 'b10sum'))
    ds1 = r.run_combos((('a', [1, 2, 3]), ('b', [1, 2, 3])))
    ds2 = r.run_combos((('a', [2, 3, 4]), ('b', [1, 2, 3])))
    ds3 = r.run_combos((('a', [4, 5, 6]), ('b', [4, 5, 6])))
    return xr.merge([ds1, ds2, ds3])


# -------------------------------- tests ------------------------------------ #


class TestFornberg:
    def test_order0(self, ds, eds):
        cds = ds.xyz.diff_fornberg('t', order=0)

        assert_allclose(cds['cos'].values,
                        eds['cos'].values,
                        atol=1e-6, rtol=1e-6)
        assert_allclose(cds['sin'].values,
                        eds['sin'].sel(f=1).values,
                        atol=1e-6, rtol=1e-6)

    def test_order1(self, ds, eds):
        cds = ds.xyz.diff_fornberg('t', order=1)

        for f in range(1, 3):
            assert_allclose(cds['cos'].sel(f=f).values,
                            - f * eds['sin'].sel(f=f).values,
                            atol=1e-4, rtol=1e-3)

    def test_order2(self, ds, eds):
        cds = ds.xyz.diff_fornberg('t', order=2)

        for f in range(1, 3):
            assert_allclose(cds['cos'].sel(f=f).values,
                            - f**2 * eds['cos'].sel(f=f).values,
                            atol=1e-3, rtol=1e-2)

    def test_nan(self, nan_ds):
        nan_ds.xyz.diff_fornberg('a')
        nan_ds.xyz.diff_fornberg('b')


class TestUnevenDiff:
    def test_diff_u(self, eds):
        for f in range(1, 3):
            assert_allclose(eds.sel(f=f).xyz.diff_u('t')['cos'].values,
                            -f * eds.sel(f=f)['sin'].values,
                            atol=1e-5, rtol=1e-3)

    def test_diff_u_err(self, eds):
        eds.xyz.diff_u_err('t')


@pytest.mark.parametrize('fit', ['unispline', 'pchip', 'interp',
                                 'polynomial', 'chebyshev', 'legendre',
                                 'laguerre', 'hermite'])
class TestFit:

    def test_ds_no_upscale(self, ds, fit):
        kws = {'dim': 't', 'ix': None}
        if fit == 'interp':
            nds = ds.xyz.interp(**kws)
        elif fit == 'unispline':
            nds = ds.xyz.unispline(**kws, num_knots=4)
        elif fit == 'pchip':
            nds = ds.xyz.interp_pchip(**kws)
        else:
            nds = ds.xyz.polyfit(**kws, poly=fit)
        assert nds.t.size == ds.t.size

    def test_ds_int_upscale(self, ds, fit):
        kws = {'dim': 't', 'ix': 50}
        if fit == 'interp':
            nds = ds.xyz.interp(**kws, order=3)
        elif fit == 'unispline':
            nds = ds.xyz.unispline(**kws, num_knots=4)
        elif fit == 'pchip':
            nds = ds.xyz.interp_pchip(**kws)
        else:
            nds = ds.xyz.polyfit(**kws, poly=fit)
        assert 't' in nds
        assert nds.t.size == 50

    def test_ds_upscale(self, ds, fit):
        kws = {'dim': 't', 'ix': np.linspace(0.1, 0.3, 13)}
        if fit == 'interp':
            nds = ds.xyz.interp(**kws, order=3)
        elif fit == 'unispline':
            nds = ds.xyz.unispline(**kws, num_knots=4)
        elif fit == 'pchip':
            nds = ds.xyz.interp_pchip(**kws)
        else:
            nds = ds.xyz.polyfit(**kws, poly=fit)
        assert 't' in nds
        assert nds.t.size == 13

    def test_nan_ds_no_upscale(self, nan_ds, fit):
        kws = {'dim': 'a', 'ix': None}
        if fit == 'interp':
            nds = nan_ds.xyz.interp(**kws, order=2)
        elif fit == 'unispline':
            nds = nan_ds.xyz.unispline(**kws, num_knots=4)
        elif fit == 'pchip':
            nds = nan_ds.xyz.interp_pchip(**kws)
        else:
            nds = nan_ds.xyz.polyfit(**kws, poly=fit)
        assert nds.a.size == nan_ds.a.size

    def test_nan_ds_int_upscale(self, nan_ds, fit):
        kws = {'dim': 'a', 'ix': 50}
        if fit == 'interp':
            nds = nan_ds.xyz.interp(**kws, order=2)
        elif fit == 'unispline':
            nds = nan_ds.xyz.unispline(**kws, num_knots=4)
        elif fit == 'pchip':
            nds = nan_ds.xyz.interp_pchip(**kws)
        else:
            nds = nan_ds.xyz.polyfit(**kws, poly=fit)
        assert 'a' in nds
        assert nds.a.size == 50

    def test_nan_ds_upscale(self, nan_ds, fit):
        kws = {'dim': 'a', 'ix': np.linspace(1.5, 2.5, 13)}
        if fit == 'interp':
            nds = nan_ds.xyz.interp(**kws, order=2)
        elif fit == 'unispline':
            nds = nan_ds.xyz.unispline(**kws, num_knots=4)
        elif fit == 'pchip':
            nds = nan_ds.xyz.interp_pchip(**kws)
        else:
            nds = nan_ds.xyz.polyfit(**kws, poly=fit)
        assert 'a' in nds
        assert nds.a.size == 13


@pytest.mark.parametrize("filter_type", ['butter', 'bessel'])
class TestFiltFilt:

    def test_all_finite(self, filter_type):
        fn = {
            'butter': _broadcast_filtfilt_butter,
            'bessel': _broadcast_filtfilt_bessel,
        }[filter_type]
        n = 20
        s = 5 * np.random.rand(n)
        x = np.cumsum(s)
        y = np.cos(x) + 0.2 * np.random.randn(n)
        yf = fn(x, y, 2, 0.3)
        assert np.all(np.isfinite(yf))

    def test_all_nan(self, filter_type):
        fn = {
            'butter': _broadcast_filtfilt_butter,
            'bessel': _broadcast_filtfilt_bessel,
        }[filter_type]
        n = 20
        s = 5 * np.random.rand(n)
        x = np.cumsum(s)
        y = np.tile(np.nan, n)
        yf = fn(x, y, 2, 0.3)
        assert np.all(~np.isfinite(yf))

    def test_some_nan(self, filter_type):
        fn = {
            'butter': _broadcast_filtfilt_butter,
            'bessel': _broadcast_filtfilt_bessel,
        }[filter_type]
        n = 20
        s = 5 * np.random.rand(n)
        x = np.cumsum(s)
        y = np.cos(x) + 0.2 * np.random.randn(n)
        y[[1, 5, 6, 10, 14]] = np.nan
        yf = fn(x, y, 2, 0.3)
        assert np.sum(np.isfinite(yf)) == 15

    def test_ds_version(self, ds, filter_type):
        if filter_type == 'butter':
            ds.xyz.filtfilt_butter(dim='t')
        elif filter_type == 'bessel':
            ds.xyz.filtfilt_bessel(dim='t')

    def test_nan_ds_version(self, nan_ds, filter_type):
        if filter_type == 'butter':
            nan_ds.xyz.filtfilt_butter(dim='a')
            nan_ds.xyz.filtfilt_butter(dim='b')
        elif filter_type == 'bessel':
            nan_ds.xyz.filtfilt_bessel(dim='a')
        nan_ds.xyz.filtfilt_bessel(dim='b')


@pytest.fixture
def ds_idx():
    x0 = np.array([-2, -1, 0, 1, 2])
    y0 = np.array([1, -2, 3])
    xs = np.linspace(-3, 3, 61)

    x0b = x0.reshape(-1, 1, 1)
    y0b = y0.reshape(1, -1, 1)
    xsb = xs.reshape(1, 1, -1)

    z = y0b - (xsb - x0b)**2

    return xr.Dataset(
        coords={
            'x0': x0,
            'y0': y0,
            'x': xs,
        },
        data_vars={
            'z': (['x0', 'y0', 'x'], z)
        }
    )


@pytest.fixture
def eds_idx():
    x0 = np.array([-2, -1, 0, 1, 2])
    y0 = np.array([1, -2, 3])

    x0v, _ = np.meshgrid(x0, y0)
    x0v = x0v.astype(float)
    return xr.Dataset(
        coords={
            'x0': x0,
            'y0': y0,
        },
        data_vars={
            'z': (['x0', 'y0'], x0v.T)
        }
    )


class TestIdxMinMax:
    @pytest.mark.parametrize('type_coord', [
        (int, [1, 2, 3]),
        (float, [1., 2., 3.]),
        (complex, [1j, 2j, 3j]),
        (np.datetime64, [np.datetime64('1991-04-11'),
                         np.datetime64('1991-04-12'),
                         np.datetime64('1991-04-13')]),
        (np.timedelta64, [np.timedelta64(24, 'M'),
                          np.timedelta64(25, 'M'),
                          np.timedelta64(26, 'M')]),
        (object, ['foo', 'bar', 'baz']),
    ])
    @pytest.mark.parametrize('somena', [False, True])
    @pytest.mark.parametrize('allna', [False, True])
    @pytest.mark.parametrize('dask', [False, True])
    def test_all(self, dask, allna, somena, type_coord):
        # Make data, with max at c0, +ve. and -ve.
        c0 = np.array([-1, 0, 1])
        ymax = np.array([-2, 2])
        xs = np.array([-1, 0.5, 0, 1])
        c0v, ymaxv, xsv = np.meshgrid(c0, ymax, xs, indexing='ij')
        z = ymaxv - (xsv - c0v)**2

        ds = xr.Dataset(coords={'c0': c0, 'ymax': ymax, 'x': xs},
                        data_vars={'z': (['c0', 'ymax', 'x'], z)})

        # expected max/min locations
        ix, _ = np.meshgrid(c0, ymax, indexing='ij')
        ieds = xr.Dataset(coords={'c0': c0, 'ymax': ymax},
                          data_vars={'z': (['c0', 'ymax'],
                                           ix.astype(float))})

        # type, coord = type_coord

        if somena:
            ds['z'].loc[{'x': 0.5}] = np.nan
            assert ds['z'].isnull().any()

        if allna:
            ds['z'].loc[{'ymax': -2}] = np.nan
            ieds['z'].loc[{'ymax': -2}] = np.nan

        if dask:
            ds = ds.chunk({'x': 2})

        ids = ds.idxmax(dim='x')
        assert ids.equals(ieds)

        ids = (-ds).idxmin(dim='x')
        assert ids.equals(ieds)
