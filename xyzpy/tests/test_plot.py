import os
from pytest import fixture, mark
import numpy as np
import xarray as xr
from ..plot.matplotlib_plotter import xmlineplot
from ..plot.plotly_plotter import ilineplot


DISPLAY_PRESENT = 'DISPLAY' in os.environ
no_display_msg = "No display found."
needs_display = mark.skipif(not DISPLAY_PRESENT, reason=no_display_msg)


@fixture
def dataset_3d():
    x = [1, 2, 3, 4, 5, 6, 8]
    z = ['a', 'b', 'c', 'd']
    d = np.random.rand(7, 4)
    ds = xr.Dataset()
    ds["x"] = x
    ds["z"] = z
    ds["y"] = (("x", "z"), d)
    return ds


# --------------------------------------------------------------------------- #
# TESTS                                                                       #
# --------------------------------------------------------------------------- #

class TestLinePlot:
    @needs_display
    def test_works_at_all(self, dataset_3d):
        ds = dataset_3d
        xmlineplot(ds, "y", "x", "z")

    @needs_display
    def test_works_1d(self, dataset_3d):
        ds = dataset_3d
        xmlineplot(ds.loc[{"z": "c"}], "y", "x")


class TestILinePlot:
    def test_works_at_all(self, dataset_3d):
        ds = dataset_3d
        ilineplot(ds, "y", "x", "z", return_fig=True)

    def test_works_1d(self, dataset_3d):
        ds = dataset_3d
        ilineplot(ds.loc[{"z": "c"}], "y", "x", return_fig=True)
