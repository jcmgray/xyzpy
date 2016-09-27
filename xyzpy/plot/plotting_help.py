"""
Helper functions for preparing data to be plotted.
"""
# TODO: Error bars                                                            #
# TODO: Allow a list of y_coos instead of a z_coo                             #

import itertools
import numpy as np
from .color import calc_colors, convert_colors


def _prepare_data_and_styles(ds, y_coo, x_coo, z_coo, zlabels,
                             colors, colormap, colormap_log,
                             colormap_reverse,
                             engine):
    # Work out whether to iterate over multiple lines
    multi_var = False
    if z_coo is not None:
        z_vals = ds[z_coo].data
    elif not isinstance(y_coo, str):
        multi_var = True
        z_vals = y_coo
    else:
        z_vals = (None,)

    def gen_xy():
        for z in z_vals:
            # Select data for current z coord - flatten for singletons
            if multi_var:
                x = ds[x_coo].data.flatten()
                y = ds[z].data.flatten()
            elif z is not None:
                x = ds.loc[{z_coo: z}][x_coo].data.flatten()
                y = ds.loc[{z_coo: z}][y_coo].data.flatten()
            else:
                x = ds[x_coo].data.flatten()
                y = ds[y_coo].data.flatten()

            # Trim out missing data
            notnull = ~np.isnan(x) & ~np.isnan(y)
            yield x[notnull], y[notnull]

    # Set custom names for each line ("ztick")
    if zlabels is not None:
        zlabels = iter(zlabels)
    elif z_coo is not None or multi_var:
        zlabels = iter(str(z) for z in z_vals)
    else:
        zlabels = itertools.repeat(None)

    # Color lines
    if colors is True:
        cols = iter(calc_colors(ds, z_coo,
                                outformat=engine,
                                colormap=colormap,
                                log_scale=colormap_log,
                                reverse=colormap_reverse))
    elif colors:
        cols = itertools.cycle(convert_colors(colors, outformat=engine))
    else:
        cols = itertools.repeat(None)

    return z_vals, cols, zlabels, gen_xy


def _process_plot_range(xlims, ylims, ds, x_coo, y_coo, padding):
    """Logic for processing limits and padding into plot ranges.
    """
    if xlims is None and padding is None:
        xlims = None
    else:
        if xlims is not None:
            xmin, xmax = xlims
        else:
            xmax, xmin = ds[x_coo].max(), ds[x_coo].min()
            xmin, xmax = float(xmin), float(xmax)
        if padding is not None:
            xrnge = xmax - xmin
            xlims = (xmin - padding * xrnge, xmax + padding * xrnge)

    if ylims is not None or padding is not None:
        if ylims is not None:
            ymin, ymax = ylims
        else:
            if isinstance(y_coo, str):
                ymax, ymin = ds[y_coo].max(), ds[y_coo].min()
            else:
                ymax = max(ds[var].max() for var in y_coo)
                ymin = min(ds[var].min() for var in y_coo)
        ymin, ymax = float(ymin), float(ymax)
        if padding is not None:
            yrnge = ymax - ymin
            ylims = (ymin - padding * yrnge, ymax + padding * yrnge)
    else:
        ylims = None

    return xlims, ylims
