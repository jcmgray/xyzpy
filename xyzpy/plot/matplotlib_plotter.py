"""
Functions for plotting datasets nicely.
"""
# TODO: no border on markers.
# TODO: unify options
# TODO: matplotlib style hlines, vlines
# TODO: names
# TODO: modularise, with mian fig func, xarray handler, and basic plotter
# TODO: mshow?
# TODO: custom xtick labels

from itertools import cycle as icycle
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
        mrkrs = icycle(mpl_markers())
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
               add_to_fig=None,
               new_axes_loc=[0.4, 0.6, 0.30, 0.25],
               add_to_axes=None,
               figsize=(8, 6),
               subplot=None,
               fignum=1,
               title=None,
               padding=0.0,

               color=[None],
               colormap="viridis",
               colormap_log=False,
               colormap_reverse=False,

               markers=None,
               line_styles=None,
               line_widths=None,
               zorders=None,

               legend=None,
               legend_loc=0,
               legend_ncol=1,

               xlabel=None,
               xlabel_pad=10,
               xlims=None,
               xticks=None,
               xticklabels_hide=False,
               logx=False,

               ylabel=None,
               ylabel_pad=10,
               ylims=None,
               yticks=None,
               yticklabels_hide=False,
               logy=False,

               zlabel=None,
               zticks=None,

               vlines=None,
               hlines=None,
               gridlines=True,
               font="Arial",
               fontsize_title=20,
               fontsize_ticks=16,
               fontsize_xlabel=20,
               fontsize_ylabel=20,
               fontsize_zlabel=20,
               fontsize_legend=18,
               ):
    """ Function for automatically plotting multiple sets of data
    using matplotlib and xarray. """

    # TODO: fallback fonts
    # TODO: homogenise options with plotly, mplot etc
    # TODO: docs
    # TODO: annotations, arbitrary text

    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rc("font", family=font)

    # Add a new set of axes to an existing plot
    if add_to_fig is not None and subplot is None:
        fig = add_to_fig
        axes = fig.add_axes(new_axes_loc)
    # Add lines to an existing set of axes
    elif add_to_axes is not None:
        fig = add_to_axes
        axes = fig.get_axes()[0]
    elif subplot is not None:
        # Add new axes as subplot to exissting subplot
        if add_to_fig is not None:
            fig = add_to_fig
        #
        else:
            fig = plt.figure(fignum, figsize=figsize, dpi=100)
        axes = fig.add_subplot(subplot)
    else:
        fig = plt.figure(fignum, figsize=figsize, dpi=100)
        axes = fig.add_axes([0.15, 0.15, 0.8, 0.75])
    axes.set_title("" if title is None else title, fontsize=fontsize_title)

    # Color lines
    if color is True:
        cols = calc_colors(ds, z_coo, plotly=False, colormap=colormap,
                           log_scale=colormap_log, reverse=colormap_reverse)
    else:
        cols = icycle(color)

    # Decide on using markers, and set custom markers and line-styles
    markers = (len(ds[y_coo]) <= 50) if markers is None else markers
    mrkrs = icycle(mpl_markers()) if markers else repeat(None)
    lines = repeat("-") if line_styles is None else icycle(line_styles)

    # Set custom names for each line ("ztick")
    if zticks is not None:
        zticks = iter(zticks)
    elif z_coo is not None:
        zticks = iter(str(z) for z in ds[z_coo].data)
    else:
        zticks = iter([None])

    if line_widths is not None:
        lws = icycle(line_widths)
    else:
        lws = icycle([1.3])

    # What order lines appear over one another
    if zorders is not None:
        zorders = icycle(zorders)
    else:
        zorders = icycle([3])

    if z_coo is not None:
        # Cycle through lines and plot
        for z, col, mrkr, ln in zip(ds[z_coo].data, cols, mrkrs, lines):
            # Select data for current z coord - flatten for singlet dimensions
            sds = ds.loc[{z_coo: z}]
            x = sds[x_coo].data.flatten()
            y = sds[y_coo].data.flatten()

            # Trim out missing data
            nans = np.logical_not(np.isnan(x) | np.isnan(y))
            x, y = x[nans], y[nans]

            # add line to axes, with options cycled through
            axes.plot(x, y, ln, c=col, lw=next(lws), marker=mrkr,
                      label=next(zticks), zorder=next(zorders))

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
        axes.plot(x, y, next(lines), lw=next(lws), zorder=next(zorders),
                  c=next(cols), label=next(zticks),
                  marker=("." if markers else None))

    # Set axis scale-type and names
    axes.set_xscale("log" if logx else "linear")
    axes.set_yscale("log" if logy else "linear")
    axes.set_xlabel(x_coo if xlabel is None else xlabel,
                    fontsize=fontsize_xlabel)
    axes.xaxis.labelpad = xlabel_pad
    axes.set_ylabel(y_coo if ylabel is None else ylabel,
                    fontsize=fontsize_ylabel)
    axes.yaxis.labelpad = ylabel_pad

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
    if xticklabels_hide:
        axes.get_xaxis().set_major_formatter(mpl.ticker.NullFormatter())
    if yticklabels_hide:
        axes.get_yaxis().set_major_formatter(mpl.ticker.NullFormatter())
    axes.tick_params(labelsize=fontsize_ticks)

    # Add grid and any custom lines
    if gridlines:
        axes.grid(True, color="0.666")
    if vlines is not None:
        for x in vlines:
            axes.axvline(x)
    if hlines is not None:
        for y in hlines:
            axes.axhline(y)

    return fig
