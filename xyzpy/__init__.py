""" """

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("xyzpy")
except _PackageNotFoundError:
    try:
        # fallback for source trees where hatch-vcs has generated _version.py.
        from ._version import version as __version__
    except ImportError:
        __version__ = "0.0.0+unknown"


import functools

import xarray as xr

from .gen.case_runner import (
    case_runner,
    case_runner_to_df,
    case_runner_to_ds,
    find_missing_cases,
    is_case_missing,
    parse_into_cases,
)
from .gen.combo_runner import (
    combo_runner,
    combo_runner_to_df,
    combo_runner_to_ds,
)
from .gen.cropping import (
    Crop,
    clean_slurm_outputs,
    grow,
    load_crops,
    manage_slurm_outputs,
)
from .gen.farming import (
    Harvester,
    Runner,
    Sampler,
    cultivate,
    label,
)
from .gen.ray_executor import (
    RayExecutor,
    RayGPUExecutor,
)
from .manage import (
    auto_xyz_ds,
    cache_to_disk,
    check_runs,
    load_df,
    load_ds,
    merge_sync_conflict_datasets,
    post_fix,
    save_df,
    save_ds,
    save_merge_ds,
    sort_dims,
    trimna,
)
from .plot.color import (
    cimluv,
    cimple,
    cimple_bright,
    cmoke,
    convert_colors,
)
from .plot.infiniplot import (
    get_neutral_style,
    infiniplot,
    neutral_style,
)

# Making interactive plots with bokeh
from .plot.plotter_bokeh import (
    auto_iheatmap,
    auto_ilineplot,
    auto_iscatter,
    iheatmap,
    ilineplot,
    iscatter,
)

# Making static plots with matplotlib
from .plot.plotter_matplotlib import (
    AutoHeatMap,
    AutoHistogram,
    AutoLinePlot,
    AutoScatter,
    HeatMap,
    Histogram,
    LinePlot,
    Scatter,
    auto_heatmap,
    auto_histogram,
    auto_lineplot,
    auto_scatter,
    heatmap,
    histogram,
    lineplot,
    scatter,
    visualize_matrix,
    visualize_tensor,
)
from .utils import (
    Benchmarker,
    MemoryMonitor,
    RunningCovariance,
    RunningCovarianceMatrix,
    RunningStatistics,
    Timer,
    benchmark,
    estimate_from_repeats,
    format_number_with_error,
    get_peak_memory_usage,
    getsizeof,
    progbar,
    report_memory,
    report_memory_gpu,
    unzip,
)


def plot(xs, ys=None, **kwargs):
    """Plot y-data against x-data

    If ``ys`` is not given, the function treats ``xs`` as y-data. It uses
    ``range(xs.shape[-1])`` as x-data. The y-data can contain multiple series
    along a z-axis. The x-data can also vary along this axis. The function
    passes all keyword arguments to
    :func:`infiniplot`

    Parameters
    ----------
    xs : array_like
        The x-data or, if ``ys`` is not given, the y-data to plot
    ys : array_like, optional
        The y-data to plot
    kwargs : dict
        Options for :func:`infiniplot`
    """
    if ys is None:
        ys = xs
        xs = range(xr.DataArray(ys).shape[-1])

    ds = auto_xyz_ds(xs, ys)
    if ds.sizes["z"] > 1:
        kwargs.setdefault("color", "z")
    if "x" in ds.data_vars:
        kwargs.setdefault("xlink", "_x")

    return infiniplot(ds, "x", "y", **kwargs)


__all__ = [
    "AutoHeatMap",
    "AutoHistogram",
    "AutoLinePlot",
    "AutoScatter",
    "Benchmarker",
    "Crop",
    "Harvester",
    "HeatMap",
    "Histogram",
    "LinePlot",
    "MemoryMonitor",
    "RayExecutor",
    "RayGPUExecutor",
    "Runner",
    "RunningCovariance",
    "RunningCovarianceMatrix",
    "RunningStatistics",
    "Sampler",
    "Scatter",
    "Timer",
    "auto_heatmap",
    "auto_histogram",
    "auto_iheatmap",
    "auto_ilineplot",
    "auto_iscatter",
    "auto_lineplot",
    "auto_scatter",
    "auto_xyz_ds",
    "benchmark",
    "cache_to_disk",
    "case_runner",
    "case_runner_to_df",
    "case_runner_to_ds",
    "check_runs",
    "cimluv",
    "cimple",
    "cimple_bright",
    "clean_slurm_outputs",
    "cmoke",
    "combo_runner",
    "combo_runner_to_df",
    "combo_runner_to_ds",
    "convert_colors",
    "cultivate",
    "estimate_from_repeats",
    "find_missing_cases",
    "format_number_with_error",
    "get_neutral_style",
    "get_peak_memory_usage",
    "getsizeof",
    "grow",
    "heatmap",
    "histogram",
    "iheatmap",
    "ilineplot",
    "infiniplot",
    "is_case_missing",
    "iscatter",
    "label",
    "lineplot",
    "load_crops",
    "load_df",
    "load_ds",
    "manage_slurm_outputs",
    "merge_sync_conflict_datasets",
    "neutral_style",
    "parse_into_cases",
    "plot",
    "progbar",
    "report_memory",
    "report_memory_gpu",
    "save_df",
    "save_ds",
    "save_merge_ds",
    "scatter",
    "sort_dims",
    "trimna",
    "unzip",
    "visualize_matrix",
    "visualize_tensor",
]


class XYZPY(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    # ------------------------------- Plotting ------------------------------ #

    @functools.wraps(infiniplot)
    def plot(self, *args, **kwargs):
        return infiniplot(self._obj, *args, **kwargs)

    @functools.wraps(infiniplot)
    def infiniplot(self, *args, **kwargs):
        return infiniplot(self._obj, *args, **kwargs)

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


xr.register_dataarray_accessor("xyz")(XYZPY)
xr.register_dataset_accessor("xyz")(XYZPY)
