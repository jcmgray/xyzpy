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

from .core import _process_plot_range, _prepare_data_and_styles
from ..manage import auto_xyz_ds


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
             figsize=(8, 6),          # absolute figure size
             axes_loc=None,           # axes location within fig
             add_to_axes=None,        # add to existing axes
             add_to_fig=None,         # add plot to an exisitng figure
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
             legend_bbox=None,        # Where to anchor the legend to
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
             font=('Source Sans Pro', 'PT Sans', 'Liberation Sans', 'Arial'),
             fontsize_title=20,
             fontsize_ticks=16,
             fontsize_xtitle=20,
             fontsize_ytitle=20,
             fontsize_ztitle=20,
             fontsize_zlabels=18,
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
        axes = fig.add_axes((0.4, 0.6, 0.30, 0.25)
                            if axes_loc is None else axes_loc)
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
        axes = fig.add_axes((0.15, 0.15, 0.8, 0.75)
                            if axes_loc is None else axes_loc)
    axes.set_title("" if title is None else title, fontsize=fontsize_title)

    z_vals, cols, zlabels, gen_xy = _prepare_data_and_styles(
        ds=ds, y_coo=y_coo, x_coo=x_coo, z_coo=z_coo, zlabels=zlabels,
        colors=colors, colormap=colormap, colormap_log=colormap_log,
        colormap_reverse=colormap_reverse, engine='MATPLOTLIB')

    # Decide on using markers, and set custom markers
    if markers is None:
        markers = len(ds[x_coo]) <= 51
    if markers:
        if len(z_vals) > 1:
            mrkrs = itertools.cycle(_MPL_MARKERS)
        else:
            mrkrs = iter('.')
    else:
        mrkrs = itertools.repeat(None)

    # Line-styles
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
                           loc=legend_loc,
                           fontsize=fontsize_zlabels,
                           frameon=False,
                           bbox_to_anchor=legend_bbox,
                           ncol=legend_ncol)
        lgnd.get_title().set_fontsize(fontsize_ztitle)

    # Set axis scale-type and names
    axes.set_xscale("log" if xlog else "linear")
    axes.set_yscale("log" if ylog else "linear")
    axes.set_xlabel(x_coo if xtitle is None else xtitle,
                    fontsize=fontsize_xtitle)
    axes.xaxis.labelpad = xtitle_pad
    if ytitle is None and isinstance(y_coo, str):
        ytitle = y_coo
    if ytitle:
        axes.set_ylabel(ytitle, fontsize=fontsize_ytitle)
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


def _choose_squarest_grid(x):
    p = x ** 0.5
    if p.is_integer():
        m = n = int(p)
    else:
        m = int(round(p))
        p = int(p)
        n = p if m * p >= x else p + 1
    return m, n


def visualize_matrix(x, figsize=(6, 6),
                     colormap='Greys',
                     touching=False,
                     return_fig=True):
    import matplotlib.pyplot as plt
    from xyzpy.plot.color import _xyz_colormaps

    fig = plt.figure(figsize=figsize, dpi=100)
    if isinstance(x, np.ndarray):
        x = (x,)

    nx = len(x)
    m, n = _choose_squarest_grid(nx)
    subplots = tuple((m, n, i) for i in range(1, nx+1))

    for img, subplot in zip(x, subplots):

        ax = fig.add_subplot(*subplot)
        ax.imshow(img, cmap=_xyz_colormaps(colormap), interpolation='nearest')
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        # Only show ticks on the left and bottom spines
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1,
                        wspace=-0.001 if touching else 0.05,
                        hspace=-0.001 if touching else 0.05)
    if return_fig:
        plt.close(fig)
        return fig
    else:
        plt.show()
