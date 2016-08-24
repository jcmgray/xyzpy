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
    return xr.Dataset(
            coords={
                'b': ['l1', 'l2', 'l4'],
                'a': [1, 2, 3, 4]},
            data_vars={
                'x': (('b', 'a'),
                      np.random.randn(3, 4) + 1.0j * np.random.randn(3, 4)),
                'isodd': ('a', np.asarray([True, False, True, False]))})


@fixture
def ds2():
    return xr.Dataset(
        coords={
            'b': ['l3', 'l5'],
            'a': [3, 4, 5]},
        data_vars={
            'x': (('b', 'a'),
                  np.random.randn(2, 3) + 1.0j * np.random.randn(2, 3)),
            'isodd': ('a', np.asarray([True, False, True]))})


@fixture
def ds3():
    return xr.Dataset(
        coords={
            'b': ['l5'],
            'a': [4]},
        data_vars={
            'x': (('b', 'a'), [[123. + 456.0j]]),
            'isodd': ('a', np.asarray([True]))})


class TestNonnullCompatible:
    def test_compatible_no_coo_overlap(self):
        # TODO ************************************************************** #
        pass

    def test_compatible_coo_overlap(self):
        # TODO ************************************************************** #
        pass

    def test_not_compatible(self):
        # TODO ************************************************************** #
        pass

    def test_different_data_vars(self):
        # TODO ************************************************************** #
        pass

    def test_overlapping_but_with_equal_values(self):
        # TODO ************************************************************** #
        pass

    def test_overlapping_but_with_equal_values_float(self):
        # TODO ************************************************************** #
        pass


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

    def test_accept_newer(self):
        # TODO ************************************************************** #
        pass

    def test_dataarrays(self):
        # TODO ************************************************************** #
        pass

    def test_type_propagation_new_variables(self):
        # TODO ************************************************************** #
        pass


class TestSaveAndLoad:
    def test_simple(self, ds1, tmpdir):
        # TODO ************************************************************** #
        pass
        # xrsave(ds1, "test.h5")
        # dsl = xrload("test.h5")
        # assert ds1.equals(dsl)
