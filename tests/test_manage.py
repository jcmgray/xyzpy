import os
import tempfile

from pytest import fixture, mark
import numpy as np
import xarray as xr

from xyzpy.manage import (
    load_ds,
    save_ds,
)


@fixture
def ds1():
    return xr.Dataset(
        coords={
            'b': ['l1', 'l2', 'l4'],
            'a': [1, 2, 3, 4]},
        data_vars={
            'x': (('b', 'a'),
                  np.random.randn(3, 4) + 1.0j * np.random.randn(3, 4)),
            'isodd': ('a', np.asarray([True, False, True, False]))},
        attrs={'foo': 'bar'})


@fixture
def ds2():
    return xr.Dataset(
        coords={
            'b': ['l3', 'l5'],
            'a': [3, 4, 5]},
        data_vars={
            'x': (('b', 'a'),
                  np.random.randn(2, 3) + 1.0j * np.random.randn(2, 3)),
            'isodd': ('a', np.asarray([True, False, True]))},
        attrs={'bar': 'baz'})


@fixture
def ds3():
    return xr.Dataset(
        coords={
            'b': ['l5'],
            'a': [4]},
        data_vars={
            'x': (('b', 'a'), [[123. + 456.0j]]),
            'isodd': ('a', np.asarray([True]))},
        attrs={'baz': 'qux'})


@fixture
def ds_real():
    return xr.Dataset(
        coords={
            'b': ['l3', 'l5'],
            'a': [3, 4, 5]},
        data_vars={
            'x': (('b', 'a'), np.random.randn(2, 3)),
            'isodd': ('a', np.asarray([True, False, True]))},
        attrs={'qux': 'corge'})


class TestSaveAndLoad:
    @mark.parametrize(("engine_save, engine_load"),
                      [('h5netcdf', 'h5netcdf'),
                       ('h5netcdf', 'netcdf4'),
                       mark.xfail(('netcdf4', 'h5netcdf')),
                       ('netcdf4', 'netcdf4')])
    def test_io_only_real(self, ds_real, engine_save, engine_load):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_ds(ds_real, os.path.join(tmpdir, "test.h5"),
                    engine=engine_save)
            ds2 = load_ds(os.path.join(tmpdir, "test.h5"), engine=engine_load)
            assert ds_real.equals(ds2)

    @mark.parametrize(("engine_save, engine_load"),
                      [('h5netcdf', 'h5netcdf'),
                       mark.xfail(('h5netcdf', 'netcdf4')),
                       mark.xfail(('netcdf4', 'h5netcdf')),
                       mark.xfail(('netcdf4', 'netcdf4'))])
    def test_io_complex_data(self, ds1, engine_save, engine_load):
        with tempfile.TemporaryDirectory() as tmpdir:
            save_ds(ds1, os.path.join(tmpdir, "test.h5"), engine=engine_save)
            ds2 = load_ds(os.path.join(tmpdir, "test.h5"), engine=engine_load)
            assert ds1.identical(ds2)
