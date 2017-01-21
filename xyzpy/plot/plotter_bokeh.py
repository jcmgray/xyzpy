import functools
from ..manage import auto_xyz_ds
from .core import LinePlotter


@functools.lru_cache(1)
def _init_bokeh_nb():
    """Cache this so it doesn't happen over and over again.
    """
    from bokeh.plotting import output_notebook
    from bokeh.resources import INLINE
    output_notebook(resources=INLINE)


def bshow(figs, nb=True, **kwargs):
    from bokeh.plotting import show
    if nb:
        _init_bokeh_nb()
        show(figs)
    else:
        show(figs)


# --------------------------------------------------------------------------- #
#                     Main lineplot interface for bokeh                       #
# --------------------------------------------------------------------------- #

class LinePlotterBokeh(LinePlotter):
    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__(*args, engine='BOKEH', **kwargs)
        self.prepare_plot_and_set_axes_scale()
        self.set_axes_labels()
        self.set_axes_range()
        self.set_spans()
        self.set_gridlines()
        self.set_tick_marks()
        self.plot_lines()
        self.plot_legend()
        self.set_tools()

    def prepare_plot_and_set_axes_scale(self):
        """Make the bokeh plot figure and set options.
        """
        from bokeh.plotting import figure

        # Currently axes scale type must be set at figure creation?
        self._plot = figure(width=int(self.figsize[0] * 80 + 100),
                            height=int(self.figsize[1] * 80),
                            x_axis_type=('log' if self.xlog else 'linear'),
                            y_axis_type=('log' if self.ylog else 'linear'),
                            title=self.title,
                            toolbar_location="above",
                            toolbar_sticky=False,
                            active_scroll="wheel_zoom",
                            logo=None,
                            webgl=False)

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

        plt_x_centre = (self._data_xmax + self._data_xmin) / 2
        plt_x_range = self._data_xmax - self._data_xmin
        xbounds = (plt_x_centre - plt_x_range, plt_x_centre + plt_x_range)
        self._plot.x_range = (DataRange1d(start=self._xlims[0],
                                          end=self._xlims[1],
                                          bounds=xbounds) if self._xlims else
                              DataRange1d(bounds=xbounds))

        plt_y_centre = (self._data_ymax + self._data_ymin) / 2
        plt_y_range = abs(self._data_ymax - self._data_ymin)
        ybounds = (plt_y_centre - plt_y_range, plt_y_centre + plt_y_range)
        self._plot.y_range = (DataRange1d(start=self._ylims[0],
                                          end=self._ylims[1],
                                          bounds=ybounds) if self._ylims else
                              DataRange1d(bounds=ybounds))

    def set_spans(self):
        """Set custom horizontal and verical line spans.
        """
        from bokeh.models import Span

        if self.hlines:
            for hl in self.hlines:
                self._plot.add_layout(Span(location=hl, dimension='width',
                                           level='glyph', line_dash='dashed',
                                           line_color=(127, 127, 127),
                                           line_width=1))
        if self.vlines:
            for vl in self.vlines:
                self._plot.add_layout(Span(location=vl, dimension='height',
                                           level='glyph', line_dash='dashed',
                                           line_color=(127, 127, 127),
                                           line_width=1))

    def set_gridlines(self):
        """Set whether to use gridlines or not.
        """
        if not self.gridlines:
            self._plot.xgrid.visible = False
            self._plot.ygrid.visible = False
        else:
            self._plot.xgrid.grid_line_dash = [2, 3]
            self._plot.ygrid.grid_line_dash = [2, 3]

    def set_tick_marks(self):
        """Set custom locations for the tick marks.
        """
        from bokeh.models import FixedTicker

        if self.xticks:
            self._plot.xaxis[0].ticker = FixedTicker(ticks=self.xticks)
        if self.yticks:
            self._plot.yaxis[0].ticker = FixedTicker(ticks=self.yticks)

    def plot_lines(self):
        """Plot the data and a corresponding legend.
        """
        from bokeh.plotting import ColumnDataSource

        if self._lgnd:
            self._lgnd_items = []

        for data in self._gen_xy():
            col = next(self._cols)
            zlabel = next(self._zlbls)
            source = ColumnDataSource(data={'x': data[0], 'y': data[1],
                                            'z_coo': [zlabel] * len(data[0])})

            line = self._plot.line('x', 'y', source=source,
                                   line_width=next(self._lws) * 1.5,
                                   color=col)
            legend_pics = [line]

            if self.markers:
                marker = next(self._mrkrs)
                m = getattr(self._plot, marker)('x', 'y', source=source,
                                                name=zlabel, color=col)
                legend_pics.append(m)

            if len(data) > 2:
                y_err_p = data[1] + data[2]
                y_err_m = data[1] - data[2]
                err = self._plot.multi_line(list(zip(data[0], data[0])),
                                            list(zip(y_err_p, y_err_m)),
                                            color=col)
                legend_pics.append(err)

            if self._lgnd:
                self._lgnd_items.append((zlabel, legend_pics))

    def plot_legend(self):
        """Add a legend to the plot.
        """
        if self._lgnd:
            from bokeh.models import Legend
            self._plot.add_layout(
                Legend(items=self._lgnd_items, location=(0, 0)), 'right')
            # Don't repeatedly redraw legend
            self._lgnd = False

    def set_tools(self):
        """Set which tools appear for the plot.
        """
        from bokeh.models import HoverTool

        self._plot.add_tools(HoverTool(tooltips=[
            ("({}, {})".format(self.x_coo, self.y_coo
                               if isinstance(self.y_coo, str) else None),
             "($x, $y)"), (self.z_coo, "@z_coo")]))

    def show(self):
        """Show the produced figure.
        """
        if self.return_fig:
            return self._plot
        bshow(self._plot)


def ilineplot(ds, y_coo, x_coo, z_coo=None, return_fig=False, **kwargs):
    """
    """
    p = LinePlotterBokeh(ds, y_coo, x_coo, z_coo,
                         return_fig=return_fig, **kwargs)
    return p.show()


# --------------------------------------------------------------------------- #
#                    Miscellenous bokeh plotting functions                    #
# --------------------------------------------------------------------------- #

def xyz_ilineplot(x, y_z, **ilineplot_opts):
    """ Take some x-coordinates and an array, convert them to a Dataset
    treating as multiple lines, then send to ilineplot. """
    ds = auto_xyz_ds(x, y_z)
    return ilineplot(ds, 'y', 'x', 'z', **ilineplot_opts)
