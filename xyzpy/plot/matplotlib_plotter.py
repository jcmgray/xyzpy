"""
Functions for plotting datasets nicely.
"""
# TODO: unify options with plotly plotters                                    #
# TODO: error bars
# TODO: refactor function names (lineplot, array_lineplot)                    #
# TODO: mshow? Remove any auto, context sensitive (use backend etc.)          #
# TODO: custom xtick labels                                                   #
# TODO: hlines and vlines style                                               #
# TODO: annotations, arbitrary text                                           #
# TODO: docs                                                                  #
# TODO: mpl heatmap                                                           #
# TODO: detect zeros in plotting coordinates and adjust padding auto          #

import itertools
import collections
import numpy as np
from ..manage import auto_xyz_ds
from .color import calc_colors


# -------------------------------------------------------------------------- #
# Plots with matplotlib only                                                 #
# -------------------------------------------------------------------------- #

_MPL_MARKER_DICT = collections.OrderedDict([
    ('o', 'circle'),
    ('x', 'x'),
    ('D', 'diamond'),
    ('+', 'plus'),
    ('s', 'square'),
    ('.', 'point'),
    ('^', 'triangle_up'),
    ('3', 'tri_left'),
    ('>', 'triangle_right'),
    ('d', 'thin_diamond'),
    ('*', 'star'),
    ('v', 'triangle_down'),
    ('|', 'vline'),
    ('1', 'tri_down'),
    ('p', 'pentagon'),
    (',', 'pixel'),
    ('2', 'tri_up'),
    ('<', 'triangle_left'),
    ('h', 'hexagon1'),
    ('4', 'tri_right'),
    (0, 'tickleft'),
    (2, 'tickup'),
    (3, 'tickdown'),
    (4, 'caretleft'),
    ('_', 'hline'),
    (5, 'caretright'),
    ('H', 'hexagon2'),
    (1, 'tickright'),
    (6, 'caretup'),
    ('8', 'octagon'),
    (7, 'caretdown')])
_MPL_MARKERS = [*_MPL_MARKER_DICT.keys()]


# -------------------------------------------------------------------------- #
# Plots with matplotlib and xarray                                           #
# -------------------------------------------------------------------------- #

def lineplot(ds, y_coo, x_coo, z_coo=None,
             # Figure options
             add_to_axes=None,  # add to existing axes
             add_to_fig=None,  # add plot to an exisitng figure
             new_axes_loc=[0.4, 0.6, 0.30, 0.25],  # overlay axes position
             figsize=(8, 6),  # absolute figure size
             subplot=None,  # make plot in subplot
             fignum=1,
             title=None,
             # Line coloring options
             colors=[None],
             colormap="xyz",
             colormap_log=False,
             colormap_reverse=False,
             # Legend options
             legend=None,
             legend_loc=0,
             legend_ncol=1,  # number of columns in the legend
             zlabel=None,  # 'z-title' i.e. legend title
             zticks=None,  # 'z-ticks' i.e. legend labels
             # x-axis options
             xlabel=None,
             xlabel_pad=10,  # distance between label and axes line
             xlims=None,
             xticks=None,
             xticklabels_hide=False,  # hide labels but not actual ticks
             xlog=False,
             # y-axis options
             ylabel=None,
             ylabel_pad=10,  # distance between label and axes line
             ylims=None,
             yticks=None,
             yticklabels_hide=False,  # hide labels but not actual ticks
             ylog=False,
             # Line markers, styles and positions
             markers=None,
             line_styles=None,
             line_widths=None,
             zorders=None,  # draw order
             # Misc options
             padding=0.0,  # plot range padding
             vlines=None,
             hlines=None,
             gridlines=True,
             font=['Source Sans Pro', 'PT Sans', 'Liberation Sans', 'Arial'],
             fontsize_title=20,
             fontsize_ticks=16,
             fontsize_xlabel=20,
             fontsize_ylabel=20,
             fontsize_zlabel=20,
             fontsize_legend=18,
             ):
    """ Take a data set and plot one of its variables as a function of two
    coordinates using matplotlib. """
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
    if colors is True:
        cols = calc_colors(ds, z_coo, plotly=False, colormap=colormap,
                           log_scale=colormap_log, reverse=colormap_reverse)
    elif colors is False:
        cols = itertools.cycle([None])
    else:
        cols = itertools.cycle(colors)

    # Decide on using markers, and set custom markers and line-styles
    markers = (len(ds[y_coo]) <= 50) if markers is None else markers
    mrkrs = (itertools.cycle(_MPL_MARKERS) if markers else
             itertools.repeat(None))
    lines = (itertools.repeat("-") if line_styles is None else
             itertools.cycle(line_styles))

    # Set custom names for each line ("ztick")
    if zticks is not None:
        zticks = iter(zticks)
    elif z_coo is not None:
        zticks = iter(str(z) for z in ds[z_coo].data)
    else:
        zticks = iter([None])

    if line_widths is not None:
        lws = itertools.cycle(line_widths)
    else:
        lws = itertools.cycle([1.3])

    # What order lines appear over one another
    if zorders is not None:
        zorders = itertools.cycle(zorders)
    else:
        zorders = itertools.cycle([3])

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
            axes.plot(x, y, ln,
                      c=col,
                      lw=next(lws),
                      marker=mrkr,
                      markeredgecolor=col,
                      label=next(zticks),
                      zorder=next(zorders))

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
        col = next(cols)
        axes.plot(x, y, next(lines),
                  lw=next(lws),
                  c=col,
                  marker=("." if markers else None),
                  markeredgecolor=col,
                  label=next(zticks),
                  zorder=next(zorders))

    # Set axis scale-type and names
    axes.set_xscale("log" if xlog else "linear")
    axes.set_yscale("log" if ylog else "linear")
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


xmlineplot = lineplot


def xyz_lineplot(x, y_z, **lineplot_opts):
    """ Take some x-coordinates and an array, convert them to a Dataset
    treating as multiple lines, then send to lineplot. """
    ds = auto_xyz_ds(x, y_z)
    # Plot dataset
    return lineplot(ds, 'y', 'x', 'z', **lineplot_opts)
