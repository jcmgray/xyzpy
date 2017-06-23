"""
XYZPY
=====
"""
# XXX: fix h5netcdf attributes recursion error ------------------------------ #
# TODO: plotters with interact, automatic ----------------------------------- #
# TODO: combos add to existing dataset. ------------------------------------- #
# TODO: save to ds every case. For case_runner only? ------------------------ #
# TODO: function for printing ranges of runs done. -------------------------- #
# TODO: logging ------------------------------------------------------------- #
# TODO: pause / finish early interactive commands. -------------------------- #
# TODO: set global progbar options e.g. notebook mode ----------------------- #
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
    combos_sow,
    grow,
    combos_reap,
    combos_reap_to_ds,
    combos_sow_and_reap,
    combos_sow_and_reap_to_ds,
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
)
from .signal import (
    wfdiff,
    xr_wfdiff,
    xr_sdiff,
)
from .plot.color import (
    convert_colors,
)
# Making static plots with matplotlib
from .plot.plotter_matplotlib import (
    LinePlot,
    lineplot,
    Scatter,
    scatter,
    Histogram,
    histogram,
    HeatMap,
    heatmap,
    xyz_lineplot,
    visualize_matrix
)
# Making interactive plots with bokeh
from .plot.plotter_bokeh import (
    ilineplot,
    xyz_ilineplot
)

__all__ = ["Runner",
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
           "combos_sow",
           "grow",
           "combos_reap",
           "combos_reap_to_ds",
           "combos_sow_and_reap",
           "combos_sow_and_reap_to_ds",
           "cache_to_disk",
           "save_ds",
           "load_ds",
           "trimna",
           "sort_dims",
           "check_runs",
           "auto_xyz_ds",
           "convert_colors",
           "LinePlot",
           "lineplot",
           "Scatter",
           "scatter",
           "Histogram",
           "histogram",
           "HeatMap",
           "heatmap",
           "ilineplot",
           "xyz_lineplot",
           "xyz_ilineplot",
           "visualize_matrix",
           "unzip",
           "progbar",
           "wfdiff",
           "xr_wfdiff",
           "xr_sdiff"]


@xr.register_dataset_accessor('xyz')
class XYZPY(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def trimna(self):
        return trimna(self._obj)

    def LinePlot(self, *args, **kwargs):
        return LinePlot(self._obj, *args, **kwargs)

    def lineplot(self, *args, **kwargs):
        return lineplot(self._obj, *args, **kwargs)

    def Scatter(self, *args, **kwargs):
        return Scatter(self._obj, *args, **kwargs)

    def scatter(self, *args, **kwargs):
        return scatter(self._obj, *args, **kwargs)

    def Histogram(self, *args, **kwargs):
        return Histogram(self._obj, *args, **kwargs)

    def histogram(self, *args, **kwargs):
        return histogram(self._obj, *args, **kwargs)

    def HeatMap(self, *args, **kwargs):
        return HeatMap(self._obj, *args, **kwargs)

    def heatmap(self, *args, **kwargs):
        return heatmap(self._obj, *args, **kwargs)

    def ilineplot(self, *args, **kwargs):
        return ilineplot(self._obj, *args, **kwargs)
