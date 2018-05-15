"""
"""
# TODO: marker alpha

import functools
import itertools
import numpy as np

from ..manage import auto_xyz_ds
from .core import (
    Plotter,
    AbstractLinePlot,
    AbstractScatter,
    AbstractHeatMap,
    calc_row_col_datasets,
    PLOTTER_DEFAULTS,
    intercept_call_arg,
    prettify,
)


@functools.lru_cache(1)
def _init_bokeh_nb():
    """Cache this so it doesn't happen over and over again.
    """
    from bokeh.plotting import output_notebook
    from bokeh.resources import INLINE
    output_notebook(resources=INLINE)


def bshow(figs, nb=True, interactive=False, **kwargs):
    """
    """
    from bokeh.plotting import show
    if nb:
        _init_bokeh_nb()
        show(figs, notebook_handle=interactive)
    else:
        show(figs)


# --------------------------------------------------------------------------- #
#                     Main lineplot interface for bokeh                       #
# --------------------------------------------------------------------------- #

class PlotterBokeh(Plotter):
    def __init__(self, ds, x, y, z=None, **kwargs):
        """
        """
        # bokeh custom options / defaults
        kwargs['return_fig'] = kwargs.pop('return_fig', False)
        self._interactive = kwargs.pop('interactive', False)

        super().__init__(ds, x, y, z, **kwargs, backend='BOKEH')

    def prepare_axes(self):
        """Make the bokeh plot figure and set options.
        """
        from bokeh.plotting import figure

        if self.add_to_axes is not None:
            self._plot = self.add_to_axes

        else:
            # Currently axes scale type must be set at figure creation?
            self._plot = figure(
                # convert figsize to roughly matplotlib dimensions
                width=int(self.figsize[0] * 80 +
                          (100 if self._use_legend else 0) +
                          (20 if self._ytitle else 0) +
                          (20 if not self.yticklabels_hide else 0)),
                height=int(self.figsize[1] * 80 +
                           (20 if self.title else 0) +
                           (20 if self._xtitle else 0) +
                           (20 if not self.xticklabels_hide else 0)),
                x_axis_type=('log' if self.xlog else 'linear'),
                y_axis_type=('log' if self.ylog else 'linear'),
                y_axis_location=('right' if self.ytitle_right else 'left'),
                title=self.title,
                toolbar_location="above",
                toolbar_sticky=False,
                active_scroll="wheel_zoom",
                logo=None,
            )

    def set_axes_labels(self):
        """Set the labels on the axes.
        """
        if self._xtitle:
            self._plot.xaxis.axis_label = self._xtitle
        if self._ytitle:
            self._plot.yaxis.axis_label = self._ytitle

    def set_axes_range(self):
        """Set the plot ranges of the axes, and the panning limits.
        """
        from bokeh.models import DataRange1d

        self.calc_data_range()

        # plt_x_centre = (self._data_xmax + self._data_xmin) / 2
        # plt_x_range = self._data_xmax - self._data_xmin
        # xbounds = (plt_x_centre - plt_x_range, plt_x_centre + plt_x_range)
        xbounds = None
        self._plot.x_range = (DataRange1d(start=self._xlims[0],
                                          end=self._xlims[1],
                                          bounds=xbounds) if self._xlims else
                              DataRange1d(bounds=xbounds))

        # plt_y_centre = (self._data_ymax + self._data_ymin) / 2
        # plt_y_range = abs(self._data_ymax - self._data_ymin)
        # ybounds = (plt_y_centre - plt_y_range, plt_y_centre + plt_y_range)
        ybounds = None
        self._plot.y_range = (DataRange1d(start=self._ylims[0],
                                          end=self._ylims[1],
                                          bounds=ybounds) if self._ylims else
                              DataRange1d(bounds=ybounds))

    def set_spans(self):
        """Set custom horizontal and verical line spans.
        """
        from bokeh.models import Span

        span_opts = {
            'level': 'glyph',
            'line_dash': 'dashed',
            'line_color': (127, 127, 127),
            'line_width': self.span_width,
        }

        if self.hlines:
            for hl in self.hlines:
                self._plot.add_layout(Span(
                    location=hl, dimension='width', **span_opts))
        if self.vlines:
            for vl in self.vlines:
                self._plot.add_layout(Span(
                    location=vl, dimension='height', **span_opts))

    def set_gridlines(self):
        """Set whether to use gridlines or not.
        """
        if not self.gridlines:
            self._plot.xgrid.visible = False
            self._plot.ygrid.visible = False
        else:
            self._plot.xgrid.grid_line_dash = self.gridline_style
            self._plot.ygrid.grid_line_dash = self.gridline_style

    def set_tick_marks(self):
        """Set custom locations for the tick marks.
        """
        from bokeh.models import FixedTicker

        if self.xticks:
            self._plot.xaxis[0].ticker = FixedTicker(ticks=self.xticks)
        if self.yticks:
            self._plot.yaxis[0].ticker = FixedTicker(ticks=self.yticks)

        if self.xticklabels_hide:
            self._plot.xaxis.major_label_text_font_size = '0pt'
        if self.yticklabels_hide:
            self._plot.yaxis.major_label_text_font_size = '0pt'

    def set_sources_heatmap(self):
        from bokeh.plotting import ColumnDataSource

        # initialize empty source
        if not hasattr(self, '_source'):
            self._source = ColumnDataSource(data=dict())

        # remove mask from data -> not necessary soon? / convert to nan?
        var = np.ma.getdata(self._heatmap_var)

        self._source.add([var], 'image')
        self._source.add([self._data_xmin], 'x')
        self._source.add([self._data_ymin], 'y')
        self._source.add([self._data_xmax - self._data_xmin], 'dw')
        self._source.add([self._data_ymax - self._data_ymin], 'dh')

    def set_sources(self):
        """Set the source dictionaries to be used by the plotter functions.
        This is seperate to allow interactive updates of the data only.
        """
        from bokeh.plotting import ColumnDataSource

        # check if heatmap
        if hasattr(self, '_heatmap_var'):
            return self.set_sources_heatmap()

        # 'copy' the zlabels iterator into src_zlbs
        self._zlbls, src_zlbs = itertools.tee(self._zlbls)

        # Initialise with empty dicts
        if not hasattr(self, "_sources"):
            self._sources = [ColumnDataSource(dict())
                             for _ in range(len(self._z_vals))]

        # range through all data and update the sources
        for i, (zlabel, data) in enumerate(zip(src_zlbs, self._gen_xy())):
            self._sources[i].add(data['x'], 'x')
            self._sources[i].add(data['y'], 'y')
            self._sources[i].add([zlabel] * len(data['x']), 'z_coo')

            # check for color for scatter plot
            if 'c' in data:
                self._sources[i].add(data['c'], 'c')

            # check if should set y_err as well
            if 'ye' in data:
                y_err_p = data['y'] + data['ye']
                y_err_m = data['y'] - data['ye']
                self._sources[i].add(
                    list(zip(data['x'], data['x'])), 'y_err_xs')
                self._sources[i].add(list(zip(y_err_p, y_err_m)), 'y_err_ys')

            # check if should set x_err as well
            if 'xe' in data:
                x_err_p = data['x'] + data['xe']
                x_err_m = data['x'] - data['xe']
                self._sources[i].add(
                    list(zip(data['y'], data['y'])), 'x_err_ys')
                self._sources[i].add(list(zip(x_err_p, x_err_m)), 'x_err_xs')

    def plot_legend(self, legend_items=None):
        """Add a legend to the plot.
        """
        if self._use_legend:
            from bokeh.models import Legend

            loc = {'best': 'top_left'}.get(self.legend_loc, self.legend_loc)
            where = {None: 'right'}.get(self.legend_where, self.legend_where)

            # might be manually specified, e.g. from multiplot
            if legend_items is None:
                legend_items = self._lgnd_items

            lg = Legend(items=legend_items)
            lg.location = loc
            lg.click_policy = 'hide'
            self._plot.add_layout(lg, where)

            # Don't repeatedly redraw legend
            self._use_legend = False

    def set_mappable(self):
        from bokeh.models import LogColorMapper, LinearColorMapper
        import matplotlib as plt

        mappr_fn = (LogColorMapper if self.colormap_log else LinearColorMapper)
        bokehpalette = [plt.colors.rgb2hex(m) for m in self.cmap(range(256))]

        self.mappable = mappr_fn(palette=bokehpalette,
                                 low=self._zmin, high=self._zmax)

    def plot_colorbar(self):
        if self._use_colorbar:

            where = {None: 'right'}.get(self.legend_where, self.legend_where)

            from bokeh.models import ColorBar, LogTicker, BasicTicker
            ticker = LogTicker if self.colormap_log else BasicTicker
            color_bar = ColorBar(color_mapper=self.mappable, location=(0, 0),
                                 ticker=ticker(desired_num_ticks=6),
                                 title=self._ctitle)
            self._plot.add_layout(color_bar, where)

    def set_tools(self):
        """Set which tools appear for the plot.
        """
        from bokeh.models import HoverTool

        self._plot.add_tools(HoverTool(tooltips=[
            ("({}, {})".format(self.x_coo, self.y_coo
                               if isinstance(self.y_coo, str) else None),
             "(@x, @y)"), (self.z_coo, "@z_coo")]))

    def update(self):
        from bokeh.io import push_notebook
        self.set_sources()
        push_notebook()

    def show(self, **kwargs):
        """Show the produced figure.
        """
        if self.return_fig:
            return self._plot
        bshow(self._plot, **kwargs)
        return self

    def prepare_plot(self):
        self.prepare_axes()
        self.set_axes_labels()
        self.set_axes_range()
        self.set_spans()
        self.set_gridlines()
        self.set_tick_marks()
        self.set_sources()


def bokeh_multi_plot(fn):
    """Decorate a plotting function to plot a grid of values.
    """

    @functools.wraps(fn)
    def multi_plotter(ds, *args, row=None, col=None, link=False, **kwargs):

        if (row is None) and (col is None):
            return fn(ds, *args, **kwargs)

        # Set some global parameters
        p = fn(ds, *args, **kwargs, call=False)
        p.prepare_data_multi_grid()

        kwargs['xlims'] = kwargs.get('xlims', [p._data_xmin, p._data_xmax])
        kwargs['ylims'] = kwargs.get('ylims', [p._data_ymin, p._data_ymax])

        kwargs['vmin'] = kwargs.pop('vmin', p.vmin)
        kwargs['vmax'] = kwargs.pop('vmax', p.vmax)

        # split the dataset into its respective rows and columns
        ds_r_c, nrows, ncols = calc_row_col_datasets(ds, row=row, col=col)

        # intercept figsize as meaning *total* size for whole grid
        figsize = kwargs.pop('figsize', None)
        if figsize is None:
            av_n = (ncols + nrows) / 2
            figsize = (2 * (4 / av_n)**0.5, 2 * (4 / av_n)**0.5)
        else:
            figsize = (figsize[0] / ncols, figsize[1] / nrows)
        kwargs['figsize'] = figsize

        # intercept return_fig for the full grid and other options
        return_fig = kwargs.pop('return_fig', False)

        subplots = {}

        # range through rows and do subplots
        for i, ds_r in enumerate(ds_r_c):
            # range through columns
            for j, sub_ds in enumerate(ds_r):
                skws = {'legend': False, 'colorbar': False}

                # if not last row
                if i != nrows - 1:
                    skws['xticklabels_hide'] = True
                    skws['xtitle'] = ''

                # if not first column
                if j != 0:
                    skws['yticklabels_hide'] = True
                    skws['ytitle'] = ''

                # label each column
                if (i == 0) and (col is not None):
                    col_val = prettify(ds[col].values[j])
                    skws['title'] = "{} = {}".format(col, col_val)
                    fx = 'fontsize_xtitle'
                    skws['fontsize_title'] = kwargs.get(
                        fx, PLOTTER_DEFAULTS[fx])

                # label each row
                if (j == ncols - 1) and (row is not None):
                    skws['ytitle_right'] = True
                    row_val = prettify(ds[row].values[i])
                    skws['ytitle'] = "{} = {}".format(row, row_val)

                subplots[i, j] = fn(sub_ds, *args, return_fig=True, call=False,
                                    **{**kwargs, **skws})

        from bokeh.layouts import gridplot

        plts = [[subplots[i, j]() for j in range(ncols)] for i in range(nrows)]

        # link zooming and panning between all plots
        if link:
            x_range, y_range = plts[0][0].x_range, plts[0][0].y_range
            for i in range(nrows):
                for j in range(ncols):
                    plts[i][j].x_range = x_range
                    plts[i][j].y_range = y_range

        # the main grid
        p._plot = gridplot(plts)

        if p._use_legend or p._use_colorbar:
            from bokeh.models import Legend, GlyphRenderer, Range1d, ColorBar
            from bokeh.layouts import row

            # plot dummy using last sub_ds
            skws = {'title': "", 'legend_loc': 'center_left',
                    'legend_where': 'left'}
            lgren = fn(sub_ds, *args, return_fig=True, **{**kwargs, **skws})

            # remove all but legend, colorbar and glyph renderers
            lgren.renderers = [
                r for r in lgren.renderers
                if isinstance(r, (Legend, GlyphRenderer, ColorBar))
            ]
            lgren.toolbar_location = None
            lgren.outline_line_color = None

            # size it - this is pretty hacky at the moment
            lgren.width = 120
            lgren.height = int(80 * figsize[1] * nrows + 100)
            lgren.x_range = Range1d(0, 0)
            lgren.y_range = Range1d(0, 0)

            # append to the right of the gridplot
            p._plot = row([p._plot, lgren])

        if return_fig:
            return p._plot
        bshow(p._plot)

    return multi_plotter


class ILinePlot(PlotterBokeh, AbstractLinePlot):

    def __init__(self, ds, x, y, z=None, y_err=None, x_err=None, **kwargs):
        super().__init__(ds, x, y, z=z, y_err=y_err, x_err=x_err, **kwargs)

    def plot_lines(self):
        """Plot the data and a corresponding legend.
        """
        self._lgnd_items = []

        for src in self._sources:
            col = next(self._cols)
            zlabel = next(self._zlbls)
            legend_pics = []

            if self.lines:
                line = self._plot.line(
                    'x', 'y',
                    source=src,
                    color=col,
                    line_dash=next(self._lines),
                    line_width=next(self._lws) * 1.5,
                )
                legend_pics.append(line)

            if self.markers:
                marker = next(self._mrkrs)
                m = getattr(self._plot, marker)(
                    'x', 'y',
                    source=src,
                    name=zlabel,
                    color=col,
                    fill_alpha=0.5,
                    line_width=0.5,
                    size=self._markersize,
                )
                legend_pics.append(m)

            # Check if errors specified as well
            if self.y_err:
                err = self._plot.multi_line(
                    xs='y_err_xs', ys='y_err_ys', source=src, color=col,
                    line_width=self.errorbar_linewidth)
                legend_pics.append(err)
            if self.x_err:
                err = self._plot.multi_line(
                    xs='x_err_xs', ys='x_err_ys', source=src, color=col,
                    line_width=self.errorbar_linewidth)
                legend_pics.append(err)

            # Add the names and styles of drawn lines for the legend
            self._lgnd_items.append((zlabel, legend_pics))

    def __call__(self):
        self.prepare_data_single()
        # Bokeh preparation
        self.prepare_plot()
        self.plot_lines()
        self.plot_legend()
        self.plot_colorbar()
        self.set_tools()
        return self.show(interactive=self._interactive)


@bokeh_multi_plot
@intercept_call_arg
def ilineplot(ds, x, y, z=None, y_err=None, x_err=None, **kwargs):
    """From  ``ds`` plot lines of ``y`` as a function of ``x``, optionally for
    varying ``z``. Interactive,

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to plot from.
    x : str
        Dimension to plot along the x-axis.
    y : str or tuple[str]
        Variable(s) to plot along the y-axis. If tuple, plot each of the
        variables - instead of ``z``.
    z : str, optional
        Dimension to plot into the page.
    y_err : str, optional
        Variable to plot as y-error.
    x_err : str, optional
        Variable to plot as x-error.
    row : str, optional
        Dimension to vary over as a function of rows.
    col : str, optional
        Dimension to vary over as a function of columns.
    plot_opts
        See ``xyzpy.plot.core.PLOTTER_DEFAULTS``.
    """
    return ILinePlot(ds, x, y, z, y_err=y_err, x_err=x_err, **kwargs)


class AutoILinePlot(ILinePlot):
    """Interactive raw data multi-line plot.
    """

    def __init__(self, x, y_z, **lineplot_opts):
        ds = auto_xyz_ds(x, y_z)
        super().__init__(ds, 'x', 'y', z='z', **lineplot_opts)


def auto_ilineplot(x, y_z, **lineplot_opts):
    """Auto version of :func:`~xyzpy.ilineplot` that accepts array arguments
    by converting them to a ``Dataset`` first.
    """
    return AutoILinePlot(x, y_z, **lineplot_opts)()


# --------------------------------------------------------------------------- #

class IScatter(PlotterBokeh, AbstractScatter):

    def __init__(self, ds, x, y, z=None, **kwargs):
        super().__init__(ds, x, y, z, **kwargs, markers=True)

    def plot_scatter(self):
        self._lgnd_items = []

        for src in self._sources:
            if 'c' in src.column_names:
                col = {'field': 'c', 'transform': self.mappable}
            else:
                col = next(self._cols)
            marker = next(self._mrkrs)
            zlabel = next(self._zlbls)
            legend_pics = []

            m = getattr(self._plot, marker)(
                'x', 'y',
                source=src,
                name=zlabel,
                color=col,
                fill_alpha=0.5,
                line_width=0.5,
                size=self._markersize,
            )
            legend_pics.append(m)

            # Add the names and styles of drawn markers for the legend
            self._lgnd_items.append((zlabel, legend_pics))

    def __call__(self):
        self.prepare_data_single()
        # Bokeh preparation
        self.prepare_plot()
        self.plot_scatter()
        self.plot_legend()
        self.plot_colorbar()
        self.set_tools()
        return self.show(interactive=self._interactive)


@bokeh_multi_plot
@intercept_call_arg
def iscatter(ds, x, y, z=None, y_err=None, x_err=None, **kwargs):
    """From  ``ds`` plot a scatter of ``y`` against ``x``, optionally for
    varying ``z``. Interactive.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to plot from.
    x : str
        Quantity to plot along the x-axis.
    y : str or tuple[str]
        Quantity(s) to plot along the y-axis. If tuple, plot each of the
        variables - instead of ``z``.
    z : str, optional
        Dimension to plot into the page.
    y_err : str, optional
        Variable to plot as y-error.
    x_err : str, optional
        Variable to plot as x-error.
    row : str, optional
        Dimension to vary over as a function of rows.
    col : str, optional
        Dimension to vary over as a function of columns.
    plot_opts
        See ``xyzpy.plot.core.PLOTTER_DEFAULTS``.
    """
    return IScatter(ds, x, y, z, y_err=y_err, x_err=x_err, **kwargs)


class AutoIScatter(IScatter):

    def __init__(self, x, y_z, **iscatter_opts):
        ds = auto_xyz_ds(x, y_z)
        super().__init__(ds, 'x', 'y', z='z', **iscatter_opts)


def auto_iscatter(x, y_z, **iscatter_opts):
    """Auto version of :func:`~xyzpy.iscatter` that accepts array arguments
    by converting them to a ``Dataset`` first.
    """
    return AutoIScatter(x, y_z, **iscatter_opts)


# --------------------------------------------------------------------------- #


_HEATMAP_ALT_DEFAULTS = (
    ('legend', False),
    ('colorbar', True),
    ('colormap', 'inferno'),
    ('gridlines', False),
    ('padding', 0),
    ('figsize', (5, 5)),  # try to be square, maybe use aspect_ratio??
)


class IHeatMap(PlotterBokeh, AbstractHeatMap):

    def __init__(self, ds, x, y, z, **kwargs):
        # set some heatmap specific options
        for k, default in _HEATMAP_ALT_DEFAULTS:
            if k not in kwargs:
                kwargs[k] = default
        super().__init__(ds, x, y, z, **kwargs)

    def plot_heatmap(self):
        self.calc_color_norm()
        self._plot.image(image='image', x='x', y='y', dw='dw', dh='dh',
                         source=self._source, color_mapper=self.mappable)

    def __call__(self):
        # Core preparation
        self.prepare_data_single()
        # matplotlib preparation
        self.prepare_plot()
        self.plot_heatmap()
        self.plot_colorbar()
        self.set_tools()
        return self.show(interactive=self._interactive)


@bokeh_multi_plot
@intercept_call_arg
def iheatmap(ds, x, y, z, **kwargs):
    """From  ``ds`` plot variable ``z`` as a function of ``x`` and ``y`` using
    a 2D heatmap. Interactive,

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to plot from.
    x : str
        Dimension to plot along the x-axis.
    y : str
        Dimension to plot along the y-axis.
    z : str, optional
        Variable to plot as colormap.
    row : str, optional
        Dimension to vary over as a function of rows.
    col : str, optional
        Dimension to vary over as a function of columns.
    plot_opts
        See ``xyzpy.plot.core.PLOTTER_DEFAULTS``.
    """
    return IHeatMap(ds, x, y, z, **kwargs)


class AutoIHeatMap(IHeatMap):

    def __init__(self, x, **iheatmap_opts):
        ds = auto_xyz_ds(x)
        super().__init__(ds, 'x', **iheatmap_opts)


def auto_iheatmap(x, **iheatmap_opts):
    """Auto version of :func:`~xyzpy.iheatmap` that accepts array arguments
    by converting them to a ``Dataset`` first.
    """
    return AutoIHeatMap(x, **iheatmap_opts)()
