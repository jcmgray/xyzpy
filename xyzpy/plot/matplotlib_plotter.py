"""
Functions for plotting datasets nicely.
"""
# TODO: unify options
# TODO: matplotlib style hlines, vlines
# TODO: logcolor
# TODO: names
# TODO: modularise, with mian fig func, xarray handler, and basic plotter
# TODO: mshow?

from itertools import cycle
from collections import OrderedDict
from itertools import repeat
import numpy as np
from .color import calc_colors


# -------------------------------------------------------------------------- #
# Plots with matplotlib only                                                 #
# -------------------------------------------------------------------------- #

def mpl_markers():
    marker_dict = OrderedDict([
        ("o", "circle"),
        ("x", "x"),
        ("D", "diamond"),
        ("+", "plus"),
        ("s", "square"),
        (".", "point"),
        ("^", "triangle_up"),
        ("3", "tri_left"),
        (">", "triangle_right"),
        ("d", "thin_diamond"),
        ("*", "star"),
        ("v", "triangle_down"),
        ("|", "vline"),
        ("1", "tri_down"),
        ("p", "pentagon"),
        (",", "pixel"),
        ("2", "tri_up"),
        ("<", "triangle_left"),
        ("h", "hexagon1"),
        ("4", "tri_right"),
        (0, "tickleft"),
        (2, "tickup"),
        (3, "tickdown"),
        (4, "caretleft"),
        ("_", "hline"),
        (5, "caretright"),
        ("H", "hexagon2"),
        (1, "tickright"),
        (6, "caretup"),
        ("8", "octagon"),
        (7, "caretdown"),
    ])
    marker_keys = [*marker_dict.keys()]
    return marker_keys


def mplot(x, y_i, fignum=1, logx=False, logy=False,
          xlims=None, ylims=None, markers=True,
          color=False, colormap="viridis", **kwargs):
    """ Function for automatically plotting multiple sets of data
    using matplot lib. """

    # TODO: homogenise options with xmlineplot

    import matplotlib.pyplot as plt
    y_i = np.array(np.squeeze(y_i), ndmin=2)
    if np.size(x) == y_i.shape[0]:
        y_i = np.transpose(y_i)
    n_y = y_i.shape[0]
    fig = plt.figure(fignum, figsize=(8, 6), dpi=100)
    axes = fig.add_axes([0.1, 0.1, 0.85, 0.8])

    if markers:
        mrkrs = cycle(mpl_markers())
    else:
        repeat(None)

    if color:
        from matplotlib import cm
        cmap = getattr(cm, colormap)
        cns = np.linspace(0, 1, n_y)
        cols = [cmap(cn, 1) for cn in cns]
    else:
        cols = repeat(None)

    for y, col, mrkr in zip(y_i, cols, mrkrs):
        axes.plot(x, y, ".-", c=col, marker=mrkr, **kwargs)
    xlims = (np.min(x), np.max(x)) if xlims is None else xlims
    ylims = (np.min(y_i), np.max(y_i)) if ylims is None else ylims
    axes.set_xlim(xlims)
    axes.set_ylim(ylims)
    axes.set_xscale("log" if logx else "linear")
    axes.set_yscale("log" if logy else "linear")
    return fig


# -------------------------------------------------------------------------- #
# Plots with matplotlib and xarray                                           #
# -------------------------------------------------------------------------- #

def xmlineplot(ds, y_coo, x_coo, z_coo=None,
               color=[None],
               colormap="viridis",
               colormap_log=False,
               colormap_reverse=False,
               legend=None,
               legend_loc=0,
               legend_ncol=1,
               markers=None,
               line_styles=None,
               line_widths=None,
               fignum=1,
               font="Arial",
               xlabel=None,
               xlims=None,
               xticks=None,
               logx=False,
               ylabel=None,
               ylims=None,
               yticks=None,
               logy=False,
               zlabel=None,
               padding=0.0,
               vlines=None,
               hlines=None,
               zticks=None,
               title=None,
               fontsize_title=20,
               fontsize_ticks=16,
               fontsize_xlabel=20,
               fontsize_ylabel=20,
               fontsize_zlabel=20,
               fontsize_legend=18,
               add_to_fig=None,
               new_axes_loc=[0.4, 0.6, 0.30, 0.25],
               add_to_axes=None,
               zorder=3,
               ):
    """ Function for automatically plotting multiple sets of data
    using matplotlib and xarray. """

    # TODO: set custom line and marker for single line
    # TODO: fallback fonts
    # TODO: homogenise options with plotly, mplot etc
    # TODO: docs
    # TODO: set colors explicitly.
    # TODO: set canvas size, side by side plots etc.
    # TODO: annotations, arbitrary text

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rc("font", family=font)

    if add_to_fig is not None:
        fig = add_to_fig
        axes = fig.add_axes(new_axes_loc)
        axes.set_title("" if title is None else title, fontsize=fontsize_title)
        axes.tick_params(labelsize=fontsize_ticks)
    elif add_to_axes is not None:
        fig = add_to_axes
        axes = fig.get_axes()[-1]
    else:
        fig = plt.figure(fignum, figsize=(8, 6), dpi=100)
        axes = fig.add_axes([0.15, 0.15, 0.8, 0.75])
        axes.set_title("" if title is None else title, fontsize=fontsize_title)
        axes.tick_params(labelsize=fontsize_ticks)

    markers = (len(ds[y_coo]) <= 50) if markers is None else markers

    if z_coo is not None:

        # Color lines
        if color is True:
            cols = calc_colors(ds, z_coo,
                               plotly=False,
                               colormap=colormap,
                               log_scale=colormap_log,
                               reverse=colormap_reverse)
        else:
            cols = cycle(color)

        # Decide on using markers, and set custom markers and line-styles
        mrkrs = cycle(mpl_markers()) if markers else repeat(None)
        lines = repeat("-") if line_styles is None else cycle(line_styles)

        # Set custom names for each line ("ztick")
        custom_zticks = zticks is not None
        if custom_zticks:
            zticks = iter(zticks)

        if line_widths is not None:
            lws = cycle(line_widths)
        else:
            lws = cycle([1.3])

        # Cycle through lines and plot
        for z, col, mrkr, ln in zip(ds[z_coo].data, cols, mrkrs, lines):
            x = ds.loc[{z_coo: z}][x_coo].data.flatten()
            y = ds.loc[{z_coo: z}][y_coo].data.flatten()
            label = next(zticks) if custom_zticks else str(z)
            axes.plot(x, y, ln, c=col, lw=next(lws), marker=mrkr,
                      label=label, zorder=zorder)
        # Add a legend
        if legend or not (legend is False or len(ds[z_coo]) > 10):
            lgnd = axes.legend(title=(z_coo if zlabel is None else zlabel),
                               loc=legend_loc, fontsize=fontsize_legend,
                               frameon=False, ncol=legend_ncol)
            lgnd.get_title().set_fontsize(fontsize_zlabel)
    else:
        # Plot single line
        x = ds[x_coo].data.flatten()
        y = ds[y_coo].data.flatten()
        # line_styles
        # marker
        axes.plot(x, y, lw=1.3, zorder=3, marker=("." if markers else None))

    # Set axis scale-type and names
    axes.set_xscale("log" if logx else "linear")
    axes.set_yscale("log" if logy else "linear")
    axes.set_xlabel(x_coo if xlabel is None else xlabel,
                    fontsize=fontsize_xlabel)
    axes.set_ylabel(y_coo if ylabel is None else ylabel,
                    fontsize=fontsize_ylabel)

    # Set plot range
    if xlims is None:
        xmax, xmin = ds[x_coo].max(), ds[x_coo].min()
        xrange = xmax - xmin
        xlims = (xmin - padding * xrange, xmax + padding * xrange)
    if ylims is None:
        ymax, ymin = ds[y_coo].max(), ds[y_coo].min()
        yrange = ymax - ymin
        ylims = (ymin - padding * yrange, ymax + padding * yrange)
    axes.set_xlim(xlims)
    axes.set_ylim(ylims)

    # Set custom axis tick marks
    if xticks is not None:
        axes.set_xticks(xticks)
        axes.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    if yticks is not None:
        axes.set_yticks(yticks)
        axes.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

    # Add grid and any custom lines
    axes.grid(True, color="0.666")
    if vlines is not None:
        for x in vlines:
            axes.axvline(x)
    if hlines is not None:
        for y in hlines:
            axes.axhline(y)

    return fig
