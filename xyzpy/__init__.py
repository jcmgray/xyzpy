"""
XYZPY
=====
"""
# TODO: plotters with interact, automatic ----------------------------------- #
# TODO: harvesting split_data_on, save_data_every --------------------------- #
import functools
import xarray as xr

from .utils import (
    unzip,
    progbar
)
from .gen.combo_runner import (
    combo_runner,
    combo_runner_to_ds
)
from .gen.case_runner import (
    case_runner,
    find_union_coords,
    all_missing_ds,
    case_runner_to_ds,
    find_missing_cases,
    fill_missing_cases
)
from .gen.batch import (
    Crop,
    grow,
)
from .gen.farming import (
    Runner,
    Harvester
)
from .manage import (
    cache_to_disk,
    save_ds,
    load_ds,
    trimna,
    sort_dims,
    check_runs,
    auto_xyz_ds,
    merge_sync_conflict_datasets,
)
from .signal import (
    wfdiff,
    xr_wfdiff,
    xr_sdiff,
    xr_interp_pchip,
    xr_filter_wiener,
    xr_filtfilt_butter,
    xr_filtfilt_bessel,
    xr_unispline,
    xr_polyfit,
)
from .plot.color import (
    convert_colors,
)
# Making static plots with matplotlib
from .plot.plotter_matplotlib import (
    LinePlot,
    lineplot,
    AutoLinePlot,
    auto_lineplot,
    Scatter,
    scatter,
    Histogram,
    histogram,
    HeatMap,
    heatmap,
    visualize_matrix
)
# Making interactive plots with bokeh
from .plot.plotter_bokeh import (
    ilineplot,
    auto_ilineplot
)

__all__ = [
    "Runner",
    "Harvester",
    "combo_runner",
    "combo_runner_to_ds",
    "case_runner",
    "find_union_coords",
    "all_missing_ds",
    "case_runner_to_ds",
    "find_missing_cases",
    "fill_missing_cases",
    "Crop",
    "grow",
    "cache_to_disk",
    "save_ds",
    "load_ds",
    "trimna",
    "sort_dims",
    "check_runs",
    "merge_sync_conflict_datasets",
    "auto_xyz_ds",
    "convert_colors",
    "LinePlot",
    "lineplot",
    "auto_lineplot",
    "AutoLinePlot",
    "Scatter",
    "scatter",
    "Histogram",
    "histogram",
    "HeatMap",
    "heatmap",
    "ilineplot",
    "auto_ilineplot",
    "visualize_matrix",
    "unzip",
    "progbar",
    "wfdiff",
    "xr_wfdiff",
    "xr_sdiff",
    "xr_interp_pchip",
    "xr_filter_wiener",
    "xr_filtfilt_butter",
    "xr_filtfilt_bessel",
    "xr_unispline",
    "xr_polyfit",
]


@xr.register_dataset_accessor('xyz')
class XYZPY(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    @functools.wraps(trimna)
    def trimna(self):
        return trimna(self._obj)

    @functools.wraps(LinePlot)
    def LinePlot(self, *args, **kwargs):
        return LinePlot(self._obj, *args, **kwargs)

    @functools.wraps(lineplot)
    def lineplot(self, *args, **kwargs):
        return lineplot(self._obj, *args, **kwargs)

    @functools.wraps(Scatter)
    def Scatter(self, *args, **kwargs):
        return Scatter(self._obj, *args, **kwargs)

    @functools.wraps(scatter)
    def scatter(self, *args, **kwargs):
        return scatter(self._obj, *args, **kwargs)

    @functools.wraps(Histogram)
    def Histogram(self, *args, **kwargs):
        return Histogram(self._obj, *args, **kwargs)

    @functools.wraps(histogram)
    def histogram(self, *args, **kwargs):
        return histogram(self._obj, *args, **kwargs)

    @functools.wraps(HeatMap)
    def HeatMap(self, *args, **kwargs):
        return HeatMap(self._obj, *args, **kwargs)

    @functools.wraps(heatmap)
    def heatmap(self, *args, **kwargs):
        return heatmap(self._obj, *args, **kwargs)

    @functools.wraps(ilineplot)
    def ilineplot(self, *args, **kwargs):
        return ilineplot(self._obj, *args, **kwargs)

    @functools.wraps(xr_interp_pchip)
    def interp_pchip(self, dim, ix=100):
        return xr_interp_pchip(self._obj, dim=dim, ix=ix)

    @functools.wraps(xr_filter_wiener)
    def filter_wiener(self, dim, mysize=5, noise=1e-2):
        return xr_filter_wiener(self._obj, dim=dim, mysize=mysize, noise=noise)

    @functools.wraps(xr_filtfilt_butter)
    def filtfilt_butter(self, dim, N=2, Wn=0.4):
        return xr_filtfilt_butter(self._obj, dim=dim, N=N, Wn=Wn)

    @functools.wraps(xr_filtfilt_bessel)
    def filtfilt_bessel(self, dim, N=2, Wn=0.4):
        return xr_filtfilt_bessel(self._obj, dim=dim, N=N, Wn=Wn)

    @functools.wraps(xr_unispline)
    def unispline(self, dim, err=None, num_knots=11, ix=None):
        return xr_unispline(self._obj, dim=dim, err=err,
                            num_knots=num_knots, ix=ix)

    @functools.wraps(xr_polyfit)
    def polyfit(self, dim, ix=None, deg=None, poly='chebyshev'):
        return xr_polyfit(self._obj, dim=dim, ix=ix, deg=deg, poly=poly)
