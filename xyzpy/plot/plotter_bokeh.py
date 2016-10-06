import functools
from ..manage import auto_xyz_ds
from .core import _prepare_data_and_styles, _process_plot_range


@functools.lru_cache(1)
def init_bokeh_nb():
    """Cache this so it doesn't happen over and over again.
    """
    from bokeh.plotting import output_notebook
    output_notebook()


def bshow(figs, nb=True, **kwargs):
    from bokeh.plotting import show
    if nb:
        init_bokeh_nb()
        show(figs)
    else:
        show(figs)


def blineplot(ds, y_coo, x_coo, z_coo=None,
              # Figure options
              figsize=(6, 5),          # absolute figure size
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
              **kwargs):
    """Interactively plot a dataset using bokeh.
    """
    from bokeh.plotting import figure
    from bokeh.models import Span

    # Prepare data and labels etc ------------------------------------------- #
    xlims, ylims = _process_plot_range(xlims, ylims, ds, x_coo, y_coo, padding)

    z_vals, cols, zlabels, gen_xy = _prepare_data_and_styles(
        ds=ds, y_coo=y_coo, x_coo=x_coo, z_coo=z_coo, zlabels=zlabels,
        colors=colors, colormap=colormap, colormap_log=colormap_log,
        colormap_reverse=colormap_reverse, engine='BOKEH')

    # Make figure and custom lines etc -------------------------------------- #
    p = figure(width=int(figsize[0] * 100),
               height=int(figsize[1] * 100),
               x_axis_label=x_coo,
               y_axis_label=y_coo,
               x_axis_type=('log' if xlog else 'linear'),
               y_axis_type=('log' if ylog else 'linear'),
               toolbar_location="above",
               toolbar_sticky=False,
               active_scroll="wheel_zoom",
               x_range=xlims,
               y_range=ylims,
               webgl=False)

    if hlines:
        for hl in hlines:
            p.add_layout(Span(location=hl, dimension='width', level='glyph',
                              line_color=(127, 127, 127), line_dash='dashed',
                              line_width=1))
    if vlines:
        for vl in vlines:
            p.add_layout(Span(location=vl, dimension='height', level='glyph',
                              line_color=(127, 127, 127), line_dash='dashed',
                              line_width=1))

    # Plot lines and markers on figure -------------------------------------- #
    for x, y in gen_xy():
        col = next(cols)
        zlabel = next(zlabels)

        p.line(x, y, legend=zlabel, color=col, line_width=1.3)
        if markers:
            p.circle(x, y, legend=zlabel, color=col)

    if return_fig:
        return p
    bshow(p, **kwargs)


def xyz_blineplot(x, y_z, **blineplot_opts):
    """ Take some x-coordinates and an array, convert them to a Dataset
    treating as multiple lines, then send to ilineplot. """
    ds = auto_xyz_ds(x, y_z)
    return blineplot(ds, 'y', 'x', 'z', **blineplot_opts)
