from pytest import fixture, mark
import numpy as np
import xarray as xr

from ..manage import (
    xrsmoosh,
    xrsave,
    xrload,
)


@fixture
def ds1():
    data1 = np.random.randn(3, 4) + 1.0j * np.random.randn(3, 4)
    data2 = np.asarray([True, False, True, False])
    ds = xr.Dataset()
    ds.coords['b'] = ['l1', 'l2', 'l4']
    ds.coords['a'] = [1, 2, 3, 4]
    ds['x'] = (('b', 'a'), data1)
    ds['isodd'] = ('a', data2)
    return ds


@fixture
def ds2():
    data1 = np.random.randn(2, 3) + 1.0j * np.random.randn(2, 3)
    data2 = np.asarray([True, False, True])
    ds = xr.Dataset()
    ds.coords['b'] = ['l3', 'l5']
    ds.coords['a'] = [3, 4, 5]
    ds['x'] = (('b', 'a'), data1)
    ds['isodd'] = ('a', data2)
    return ds


class TestXRSmoosh:
    def test_simple(self, ds1, ds2):
        # TODO -------------------------------------------------------------- #
        fds = xrsmoosh(ds1, ds2)
        assert fds['x'].dtype == complex
        assert fds['x'].dtype == complex
        assert (fds.loc[{'a': 3, 'b': "l2"}]['x'].data ==
                ds1.loc[{'a': 3, 'b': "l2"}]['x'].data)
        assert (fds.loc[{'a': 5, 'b': "l5"}]['x'].data ==
                ds2.loc[{'a': 5, 'b': "l5"}]['x'].data)
        assert np.isnan(fds.loc[{'a': 2, 'b': "l5"}]['x'].data)
        assert np.isnan(fds.loc[{'a': 5, 'b': "l1"}]['x'].data)


class TestSaveAndLoad:
    def test_simple(self, ds1, tmpdir):
        # TODO -------------------------------------------------------------- #
        pass
        # xrsave(ds1, "test.h5")
        # dsl = xrload("test.h5")
        # assert ds1.equals(dsl)
