"""
Functions for plotting datasets nicely.
"""
# TODO: unify options with plotly plotters                                    #
# TODO: error bars                                                            #
# TODO: mshow? Remove any auto, context sensitive (use backend etc.)          #
# TODO: custom xtick labels                                                   #
# TODO: annotations, arbitrary text                                           #
# TODO: docs                                                                  #
# TODO: mpl heatmap                                                           #
# TODO: detect zeros in plotting coordinates and adjust padding auto          #

import itertools
import collections
import numpy as np
from ..manage import auto_xyz_ds
from .color import calc_colors, convert_colors
from .plotting_help import _process_plot_range


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
             add_to_axes=None,        # add to existing axes
             add_to_fig=None,         # add plot to an exisitng figure
             new_axes_loc=[0.4, 0.6, 0.30, 0.25],  # overlay axes position
             figsize=(8, 6),          # absolute figure size
             subplot=None,            # make plot in subplot
             fignum=1,
             title=None,
             # Line coloring options
             colors=None,
             colormap="xyz",
             colormap_log=False,
             colormap_reverse=False,
             # Legend options
             legend=None,
             legend_loc=0,            # legend location
             ztitle=None,             # legend title
             zlabels=None,            # legend labels
             legend_ncol=1,           # number of columns in the legend
             # x-axis options
             xtitle=None,
             xtitle_pad=10,           # distance between label and axes line
             xlims=None,              # plotting range on x axis
             xticks=None,             # where to place x ticks
             xticklabels_hide=False,  # hide labels but not actual ticks
             xlog=False,              # logarithmic x scale
             # y-axis options
             ytitle=None,
             ytitle_pad=10,           # distance between label and axes line
             ylims=None,              # plotting range on y-axis
             yticks=None,             # where to place y ticks
             yticklabels_hide=False,  # hide labels but not actual ticks
             ylog=False,              # logarithmic y scale
             # Shapes
             markers=None,            # use markers for each plotted point
             line_styles=None,        # iterable of line-styles, e.g. '--'
             line_widths=None,        # iterable of line-widths
             zorders=None,            # draw order
             # Misc options
             padding=None,            # plot range padding (as fraction)
             vlines=None,             # iterable of vertical lines to plot
             hlines=None,             # iterable of horizontal lines to plot
             gridlines=True,
             font=['Source Sans Pro', 'PT Sans', 'Liberation Sans', 'Arial'],
             fontsize_title=20,
             fontsize_ticks=16,
             fontsize_xtitle=20,
             fontsize_ytitle=20,
             fontsize_ztitle=20,
             fontsize_legend=18,
             return_fig=False,
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
        # Add new axes as subplot to existing subplot
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

    # Work out whether to iterate over multiple lines
    if z_coo is not None:
        z_vals = ds[z_coo].data
    else:
        z_vals = (None,)

    # Color lines
    if colors is True:
        cols = iter(calc_colors(ds, z_coo, plotly=False,
                                colormap=colormap,
                                log_scale=colormap_log,
                                reverse=colormap_reverse))
    elif colors:
        cols = itertools.cycle(convert_colors(colors, outformat='MATPLOTLIB'))
    else:
        cols = itertools.repeat(None)

    # Set custom names for each line ("ztick")
    if zlabels is not None:
        zlabels = iter(zlabels)
    elif z_coo is not None:
        zlabels = iter(str(z) for z in z_vals)
    else:
        zlabels = itertools.repeat(None)

    # Decide on using markers, and set custom markers and line-styles
    markers = (len(ds[y_coo]) <= 51) if markers is None else markers
    if markers:
        if len(z_vals) > 1:
            mrkrs = itertools.cycle(_MPL_MARKERS)
        else:
            mrkrs = iter('.')
    else:
        mrkrs = itertools.repeat(None)

    lines = (itertools.repeat("-") if line_styles is None else
             itertools.cycle(line_styles))

    # Set custom widths for each line
    if line_widths is not None:
        lws = itertools.cycle(line_widths)
    else:
        lws = itertools.cycle([1.3])

    # What order lines appear over one another
    if zorders is not None:
        zorders = itertools.cycle(zorders)
    else:
        zorders = itertools.cycle([3])

    def gen_xy():
        for z in z_vals:
            # Select data for current z coord - flatten for singletons
            sds = ds.loc[{z_coo: z}] if z is not None else ds
            x = sds[x_coo].data.flatten()
            y = sds[y_coo].data.flatten()
            # Trim out missing data
            notnull = ~np.isnan(x) & ~np.isnan(y)
            yield x[notnull], y[notnull]

    for x, y in gen_xy():
        col = next(cols)

        # add line to axes, with options cycled through
        axes.plot(x, y, next(lines),
                  c=col,
                  lw=next(lws),
                  marker=next(mrkrs),
                  markeredgecolor=col,
                  label=next(zlabels),
                  zorder=next(zorders))

    # Add a legend
    auto_no_legend = (legend is False or len(z_vals) > 10 or len(z_vals) == 1)
    if legend or not auto_no_legend:
        lgnd = axes.legend(title=(z_coo if ztitle is None else ztitle),
                           loc=legend_loc, fontsize=fontsize_legend,
                           frameon=False, ncol=legend_ncol)
        lgnd.get_title().set_fontsize(fontsize_ztitle)

    # Set axis scale-type and names
    axes.set_xscale("log" if xlog else "linear")
    axes.set_yscale("log" if ylog else "linear")
    axes.set_xlabel(x_coo if xtitle is None else xtitle,
                    fontsize=fontsize_xtitle)
    axes.xaxis.labelpad = xtitle_pad
    axes.set_ylabel(y_coo if ytitle is None else ytitle,
                    fontsize=fontsize_ytitle)
    axes.yaxis.labelpad = ytitle_pad

    # Set plot range
    xlims, ylims = _process_plot_range(xlims, ylims, ds, x_coo, y_coo, padding)
    if xlims:
        axes.set_xlim(xlims)
    if ylims:
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
    axes.tick_params(labelsize=fontsize_ticks, direction='out')

    # Add grid and any custom lines
    if gridlines:
        axes.set_axisbelow(True)  # ensures gridlines below everything else
        axes.grid(True, color="0.666")
    if vlines is not None:
        for x in vlines:
            axes.axvline(x, color="0.5", linestyle="dashed")
    if hlines is not None:
        for y in hlines:
            axes.axhline(y, color="0.5", linestyle="dashed")

    if return_fig:
        plt.close(fig)
        return fig


def xyz_lineplot(x, y_z, **lineplot_opts):
    """ Take some x-coordinates and an array, convert them to a Dataset
    treating as multiple lines, then send to lineplot. """
    ds = auto_xyz_ds(x, y_z)
    # Plot dataset
    return lineplot(ds, 'y', 'x', 'z', **lineplot_opts)
