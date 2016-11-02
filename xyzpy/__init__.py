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

from .utils import (unzip,
                    progbar)
from .gen.combo_runner import (combo_runner,
                               combo_runner_to_ds)
from .gen.case_runner import (case_runner,
                              find_union_coords,
                              all_missing_ds,
                              case_runner_to_ds,
                              find_missing_cases,
                              fill_missing_cases)
from .gen.farming import (Runner,
                          Harvester)
from .manage import (cache_to_disk,
                     xrsmoosh,
                     xrsave,
                     xrload,
                     auto_xyz_ds)
from .plot.color import (convert_colors,
                         calc_colors)
# Making static plots with matplotlib
from .plot.plotter_matplotlib import (lineplot,
                                      xyz_lineplot,
                                      visualize_matrix)
# Making interactive plots with plotly
from .plot.plotter_plotly import (ishow,
                                  ilineplot,
                                  xyz_ilineplot,
                                  iheatmap,
                                  iscatter,
                                  ihist,
                                  ivisualize_matrix)
# Making interactive plots with bokeh
from .plot.plotter_bokeh import (blineplot,
                                 xyz_blineplot)

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
           "cache_to_disk",
           "xrsave",
           "xrload",
           "auto_xyz_ds",
           "convert_colors",
           "calc_colors",
           "lineplot",
           "xyz_lineplot",
           "visualize_matrix",
           "ishow",
           "ilineplot",
           "xyz_ilineplot",
           "iheatmap",
           "iscatter",
           "ihist",
           "ivisualize_matrix",
           "blineplot",
           "xyz_blineplot",
           "unzip",
           "progbar",
           "xrsmoosh"]
