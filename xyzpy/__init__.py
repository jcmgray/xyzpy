"""
"""
import functools
import xarray as xr

from .utils import (
    unzip,
    progbar,
    getsizeof,
    Timer,
    benchmark,
    Benchmarker,
    RunningStatistics,
    RunningCovariance,
    RunningCovarianceMatrix,
    estimate_from_repeats,
)
from .gen.combo_runner import (
    combo_runner,
    combo_runner_to_ds,
    combo_runner_to_df,
)
from .gen.case_runner import (
    case_runner,
    case_runner_to_ds,
    case_runner_to_df,
    find_missing_cases,
)
from .gen.cropping import (
    Crop,
    grow,
    load_crops,
)
from .gen.farming import (
    Runner,
    Harvester,
    label,
    Sampler,
)
from .manage import (
    cache_to_disk,
    save_ds,
    load_ds,
    save_merge_ds,
    save_df,
    load_df,
    trimna,
    sort_dims,
    check_runs,
    auto_xyz_ds,
    merge_sync_conflict_datasets,
    post_fix,
)
from .plot.color import (
    convert_colors,
    cimple,
    cimple_bright,
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
    visualize_matrix,
    visualize_tensor,
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
    "Sampler",
    "label",
    "combo_runner",
    "combo_runner_to_ds",
    "combo_runner_to_df",
    "case_runner",
    "case_runner_to_ds",
    "case_runner_to_df",
    "find_missing_cases",
    "Crop",
    "grow",
    "load_crops",
    "cache_to_disk",
    "save_ds",
    "load_ds",
    "save_merge_ds",
    "save_df",
    "load_df",
    "trimna",
    "sort_dims",
    "check_runs",
    "merge_sync_conflict_datasets",
    "auto_xyz_ds",
    "convert_colors",
    "cimple",
    "cimple_bright",
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
    "visualize_tensor",
    "unzip",
    "progbar",
    "getsizeof",
    "Timer",
    "benchmark",
    "Benchmarker",
    "RunningStatistics",
    "RunningCovariance",
    "RunningCovarianceMatrix",
    "estimate_from_repeats",
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


xr.register_dataarray_accessor('xyz')(XYZPY)
xr.register_dataset_accessor('xyz')(XYZPY)
