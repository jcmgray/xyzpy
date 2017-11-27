"""
"""
# TODO: marker alpha


import functools
import itertools
from ..manage import auto_xyz_ds
from .core import Plotter


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
        super().__init__(ds, x, y, z, **kwargs, backend='BOKEH')
        self._interactive = kwargs.pop('interactive', False)

    def prepare_plot_and_set_axes_scale(self):
        """Make the bokeh plot figure and set options.
        """
        from bokeh.plotting import figure

        if self.add_to_axes is not None:
            self._plot = self.add_to_axes

        else:
            # Currently axes scale type must be set at figure creation?
            self._plot = figure(
                # convert figsize to roughly matplotlib dimensions
                width=int(self.figsize[0] * 80 + 100),
                height=int(self.figsize[1] * 80),
                x_axis_type=('log' if self.xlog else 'linear'),
                y_axis_type=('log' if self.ylog else 'linear'),
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

    def set_sources(self):
        """Set the source dictionaries to be used by the plotter functions.
        This is seperate to allow interactive updates of the data.
        """
        # 'copy' the zlabels iterator into src_zlbs
        self._zlbls, src_zlbs = itertools.tee(self._zlbls)

        # Initialise with empty dicts
        if not hasattr(self, "_sources"):
            from bokeh.plotting import ColumnDataSource
            self._sources = [ColumnDataSource(dict())
                             for _ in range(len(self._z_vals))]

        # range through all data and update the sources
        for i, (zlabel, data) in enumerate(zip(src_zlbs, self._gen_xy())):
            self._sources[i].add(data['x'], 'x')
            self._sources[i].add(data['y'], 'y')
            self._sources[i].add([zlabel] * len(data['x']), 'z_coo')

            # check if should set y_err as well
            if self.y_err:
                y_err_p = data['y'] + data['ye']
                y_err_m = data['y'] - data['ye']
                self._sources[i].add(
                    list(zip(data['x'], data['x'])), 'y_err_xs')
                self._sources[i].add(list(zip(y_err_p, y_err_m)), 'y_err_ys')

            # check if should set x_err as well
            if self.x_err:
                x_err_p = data['x'] + data['xe']
                x_err_m = data['x'] - data['xe']
                self._sources[i].add(
                    list(zip(data['y'], data['y'])), 'x_err_ys')
                self._sources[i].add(list(zip(x_err_p, x_err_m)), 'x_err_xs')

    def plot_lines(self):
        """Plot the data and a corresponding legend.
        """
        if self._use_legend:
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
            if self._use_legend:
                self._lgnd_items.append((zlabel, legend_pics))

    def plot_scatter(self):
        if self._use_legend:
            self._lgnd_items = []

        for src in self._sources:
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
            if self._use_legend:
                self._lgnd_items.append((zlabel, legend_pics))

    def plot_legend(self):
        """Add a legend to the plot.
        """
        if self._use_legend:
            from bokeh.models import Legend
            lg = Legend(items=self._lgnd_items)
            lg.location = 'top_right'
            lg.click_policy = 'hide'
            self._plot.add_layout(lg, 'right')

            # Don't repeatedly redraw legend
            self._use_legend = False

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


def ilineplot(ds, x, y, z=None, *, interactive=False,
              return_fig=False, **kwargs):
    """Interactive ``xarray``-object multi-line plot using ``bokeh``.
    """
    p = PlotterBokeh(ds, x, y, z, return_fig=return_fig, **kwargs)
    # Core preparation
    p.prepare_axes_labels()
    p.prepare_z_vals()
    p.prepare_z_labels()
    p.calc_use_legend_or_colorbar()
    p.prepare_xy_vals_lineplot()
    p.prepare_line_colors()
    p.prepare_markers()
    p.prepare_line_styles()
    p.prepare_zorders()
    p.calc_plot_range()
    # Bokeh preparation
    p.prepare_plot_and_set_axes_scale()
    p.set_axes_labels()
    p.set_axes_range()
    p.set_spans()
    p.set_gridlines()
    p.set_tick_marks()
    p.set_sources()
    p.plot_lines()
    p.plot_legend()
    p.set_tools()
    return p.show(interactive=interactive)


def iscatter(ds, x, y, z=None, *, interactive=False,
             return_fig=False, **kwargs):
    """Interactive ``xarray``-object multi-scatter plot using ``bokeh``.
    """
    p = PlotterBokeh(ds, x, y, z, markers=True, lines=False,
                     return_fig=return_fig, **kwargs)
    # Core preparation
    p.prepare_axes_labels()
    p.prepare_z_vals()
    p.prepare_z_labels()
    p.calc_use_legend_or_colorbar()
    p.prepare_xy_vals_lineplot()
    p.prepare_line_colors()
    p.prepare_markers()
    p.prepare_line_styles()
    p.prepare_zorders()
    p.calc_plot_range()
    # Bokeh preparation
    p.prepare_plot_and_set_axes_scale()
    p.set_axes_labels()
    p.set_axes_range()
    p.set_spans()
    p.set_gridlines()
    p.set_tick_marks()
    p.set_sources()
    p.plot_scatter()
    p.plot_legend()
    p.set_tools()
    return p.show(interactive=interactive)


# --------------------------------------------------------------------------- #
#                    Miscellenous bokeh plotting functions                    #
# --------------------------------------------------------------------------- #

def auto_ilineplot(x, y_z, **ilineplot_opts):
    """ Take some x-coordinates and an array, convert them to a Dataset
    treating as multiple lines, then send to ilineplot. """
    ds = auto_xyz_ds(x, y_z)
    return ilineplot(ds, 'x', 'y', 'z', **ilineplot_opts)


def auto_iscatter(x, y_z, **ilineplot_opts):
    """ Take some x-coordinates and an array, convert them to a Dataset
    treating as multiple lines, then send to ilineplot. """
    ds = auto_xyz_ds(x, y_z)
    return iscatter(ds, 'x', 'y', 'z', **ilineplot_opts)
