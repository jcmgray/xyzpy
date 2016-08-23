from pytest import fixture, raises
import numpy as np
import xarray as xr

from ..manage import (
    aggregate,
    # xrload,
    # xrsave,
)


@fixture
def ds1():
    ds = xr.Dataset()
    ds.coords['b'] = ['l1', 'l2', 'l4']
    ds.coords['a'] = [1, 2, 3, 4]
    ds['x'] = (('b', 'a'),
               np.random.randn(3, 4) + 1.0j * np.random.randn(3, 4))
    ds['isodd'] = ('a', np.asarray([True, False, True, False]))
    return ds


@fixture
def ds2():
    ds = xr.Dataset()
    ds.coords['b'] = ['l3', 'l5']
    ds.coords['a'] = [3, 4, 5]
    ds['x'] = (('b', 'a'),
               np.random.randn(2, 3) + 1.0j * np.random.randn(2, 3))
    ds['isodd'] = ('a', np.asarray([True, False, True]))
    return ds


@fixture
def ds3():
    ds = xr.Dataset()
    ds.coords['b'] = ['l5']
    ds.coords['a'] = [4]
    ds['x'] = (('b', 'a'), [[123. + 456.0j]])
    ds['isodd'] = ('a', np.asarray([True]))
    return ds


class TestAggregate:
    def test_simple(self, ds1, ds2):
        fds = aggregate(ds1, ds2)
        assert fds['x'].dtype == complex
        assert fds['x'].dtype == complex
        assert (fds.loc[{'a': 3, 'b': "l2"}]['x'].data ==
                ds1.loc[{'a': 3, 'b': "l2"}]['x'].data)
        assert (fds.loc[{'a': 5, 'b': "l5"}]['x'].data ==
                ds2.loc[{'a': 5, 'b': "l5"}]['x'].data)
        assert np.isnan(fds.loc[{'a': 2, 'b': "l5"}]['x'].data)
        assert np.isnan(fds.loc[{'a': 5, 'b': "l1"}]['x'].data)

    def test_no_overwrite(self, ds2, ds3):
        with raises(ValueError):
            aggregate(ds2, ds3)

    def test_overwrite(self):
        # TODO ************************************************************** #
        pass

    # TODO: test type maintained when var does not exist in first dataset
    # TODO: test accept_newer


class TestSaveAndLoad:
    def test_simple(self, ds1, tmpdir):
        # TODO ************************************************************** #
        pass
        # xrsave(ds1, "test.h5")
        # dsl = xrload("test.h5")
        # assert ds1.equals(dsl)
