"""
"""
import functools
import xarray as xr

from .utils import (
    unzip,
    progbar,
    Timer,
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
    post_fix,
)
from .signal import (
    xr_diff_fornberg,
    xr_diff_u,
    xr_diff_u_err,
    xr_interp,
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
    AutoScatter,
    auto_scatter,
    Histogram,
    histogram,
    AutoHistogram,
    auto_histogram,
    HeatMap,
    heatmap,
    AutoHeatMap,
    auto_heatmap,
    visualize_matrix
)
# Making interactive plots with bokeh
from .plot.plotter_bokeh import (
    ilineplot,
    auto_ilineplot,
    iscatter,
    auto_iscatter,
    iheatmap,
    auto_iheatmap,
)


# versioneer
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


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
    "AutoScatter",
    "auto_scatter",
    "Histogram",
    "histogram",
    "AutoHistogram",
    "auto_histogram",
    "HeatMap",
    "heatmap",
    "AutoHeatMap",
    "auto_heatmap",
    "ilineplot",
    "auto_ilineplot",
    "iscatter",
    "auto_iscatter",
    "iheatmap",
    "auto_iheatmap",
    "visualize_matrix",
    "unzip",
    "progbar",
    "Timer",
    "xr_diff_fornberg",
    "xr_diff_u",
    "xr_diff_u_err",
    "xr_interp",
    "xr_interp_pchip",
    "xr_filter_wiener",
    "xr_filtfilt_butter",
    "xr_filtfilt_bessel",
    "xr_unispline",
    "xr_polyfit",
]


class XYZPY(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    # ------------------------------- Plotting ------------------------------ #

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

    @functools.wraps(iscatter)
    def iscatter(self, *args, **kwargs):
        return iscatter(self._obj, *args, **kwargs)

    @functools.wraps(iheatmap)
    def iheatmap(self, *args, **kwargs):
        return iheatmap(self._obj, *args, **kwargs)

    # ----------------------------- Processing ------------------------------ #

    @functools.wraps(trimna)
    def trimna(self):
        return trimna(self._obj)

    @functools.wraps(post_fix)
    def post_fix(self, postfix):
        return post_fix(self._obj, postfix)

    @functools.wraps(xr_diff_fornberg)
    def diff_fornberg(self, dim, ix=100, order=1, mode='points', window=5):
        return xr_diff_fornberg(self._obj, dim=dim, ix=ix, order=order,
                                mode=mode, window=window)

    @functools.wraps(xr_diff_u)
    def diff_u(self, dim):
        return xr_diff_u(self._obj, dim=dim)

    @functools.wraps(xr_diff_u_err)
    def diff_u_err(self, dim):
        return xr_diff_u_err(self._obj, dim=dim)

    @functools.wraps(xr_interp)
    def interp(self, dim, ix=100, order=3):
        return xr_interp(self._obj, dim=dim, ix=ix, order=order)

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
    def polyfit(self, dim, ix=None, deg=0.5, poly='chebyshev'):
        return xr_polyfit(self._obj, dim=dim, ix=ix, deg=deg, poly=poly)


xr.register_dataarray_accessor('xyz')(XYZPY)
xr.register_dataset_accessor('xyz')(XYZPY)
