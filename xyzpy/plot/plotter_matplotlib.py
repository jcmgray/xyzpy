"""
Functions for plotting datasets nicely.
"""
# TODO: custom xtick labels                                                   #
# TODO: annotations, arbitrary text                                           #
# TODO: docs                                                                  #

import functools
import numpy as np
from ..manage import auto_xyz_ds
from .core import (
    Plotter,
    AbstractLinePlot,
    AbstractScatter,
    AbstractHistogram,
    AbstractHeatMap,
    PLOTTER_DEFAULTS,
    calc_row_col_datasets,
    intercept_call_arg,
    prettify,
)
from .color import xyz_colormaps


# ----------------- Main lineplot interface for matplotlib ------------------ #

class PlotterMatplotlib(Plotter):
    """
    """

    def __init__(self, ds, x, y, z=None, y_err=None, x_err=None, **kwargs):
        super().__init__(ds, x, y, z=z, y_err=y_err, x_err=x_err,
                         **kwargs, backend='MATPLOTLIB')

    def prepare_axes(self):
        """
        """
        import matplotlib as mpl
        if self.math_serif:
            mpl.rcParams['mathtext.fontset'] = 'cm'
            mpl.rcParams['mathtext.rm'] = 'serif'
        mpl.rcParams['font.family'] = self.font
        import matplotlib.pyplot as plt

        if self.axes_rloc is not None:
            if self.axes_loc is not None:
                raise ValueError("Cannot specify absolute and relative "
                                 "location of axes at the same time.")
            if self.add_to_fig is None:
                raise ValueError("Can only specify relative axes position "
                                 "when adding to a figure, i.e. when "
                                 "add_to_fig != None")

        if self.axes_rloc is not None:
            self._axes_loc = self._cax_rel2abs_rect(
                self.axes_rloc, self.add_to_fig.get_axes()[-1])
        else:
            self._axes_loc = self.axes_loc

        # Add a new set of axes to an existing plot
        if self.add_to_fig is not None and self.subplot is None:
            self._fig = self.add_to_fig
            self._axes = self._fig.add_axes((0.4, 0.6, 0.30, 0.25)
                                            if self._axes_loc is None else
                                            self._axes_loc)

        # Add lines to an existing set of axes
        elif self.add_to_axes is not None:
            self._fig = self.add_to_axes
            self._axes = self._fig.get_axes()[-1]

        # Add lines to existing axes but only sharing the x-axis
        elif self.add_to_xaxes is not None:
            self._fig = self.add_to_xaxes
            self._axes = self._fig.get_axes()[-1].twinx()

        elif self.subplot is not None:
            # Add new axes as subplot to existing subplot
            if self.add_to_fig is not None:
                self._fig = self.add_to_fig

            # New figure but add as subplot
            else:
                self._fig = plt.figure(self.fignum, figsize=self.figsize,
                                       dpi=100)
            self._axes = self._fig.add_subplot(self.subplot)

        # Make new figure and axes
        else:
            self._fig = plt.figure(self.fignum, figsize=self.figsize, dpi=100)
            self._axes = self._fig.add_axes((0.15, 0.15, 0.8, 0.75)
                                            if self._axes_loc is None else
                                            self._axes_loc)
        self._axes.set_title("" if self.title is None else self.title,
                             fontsize=self.fontsize_title)

    def set_axes_labels(self):
        if self._xtitle:
            self._axes.set_xlabel(self._xtitle, fontsize=self.fontsize_xtitle)
            self._axes.xaxis.labelpad = self.xtitle_pad
        if self._ytitle:
            self._axes.set_ylabel(self._ytitle, fontsize=self.fontsize_ytitle)
            self._axes.yaxis.labelpad = self.ytitle_pad
            if self.ytitle_right:
                self._axes.yaxis.set_label_position("right")

    def set_axes_scale(self):
        """
        """
        self._axes.set_xscale("log" if self.xlog else "linear")
        self._axes.set_yscale("log" if self.ylog else "linear")

    def set_axes_range(self):
        """
        """
        if self._xlims:
            self._axes.set_xlim(self._xlims)
        if self._ylims:
            self._axes.set_ylim(self._ylims)

    def set_spans(self):
        """
        """
        if self.vlines is not None:
            for x in self.vlines:
                self._axes.axvline(x, lw=self.span_width,
                                   color=self.span_color,
                                   linestyle=self.span_style)
        if self.hlines is not None:
            for y in self.hlines:
                self._axes.axhline(y, lw=self.span_width,
                                   color=self.span_color,
                                   linestyle=self.span_style)

    def set_gridlines(self):
        """
        """
        for axis in ('top', 'bottom', 'left', 'right'):
            self._axes.spines[axis].set_linewidth(1.0)

        if self.gridlines:
            # matplotlib has coarser gridine style than bokeh
            self._gridline_style = [x / 2 for x in self.gridline_style]
            self._axes.set_axisbelow(True)  # ensures gridlines below all
            self._axes.grid(True, color="0.9", which='major',
                            linestyle=(0, self._gridline_style))
            self._axes.grid(True, color="0.95", which='minor',
                            linestyle=(0, self._gridline_style))

    def set_tick_marks(self):
        """
        """
        import matplotlib as mpl

        if self.xticks is not None:
            self._axes.set_xticks(self.xticks, minor=False)
            self._axes.get_xaxis().set_major_formatter(
                mpl.ticker.ScalarFormatter())
        if self.yticks is not None:
            self._axes.set_yticks(self.yticks, minor=False)
            self._axes.get_yaxis().set_major_formatter(
                mpl.ticker.ScalarFormatter())

        if self.xtick_labels is not None:
            self._axes.set_xticklabels(self.xtick_labels)

        if self.xticklabels_hide:
            (self._axes.get_xaxis()
             .set_major_formatter(mpl.ticker.NullFormatter()))
        if self.yticklabels_hide:
            (self._axes.get_yaxis()
             .set_major_formatter(mpl.ticker.NullFormatter()))

        self._axes.tick_params(labelsize=self.fontsize_ticks, direction='out',
                               bottom='bottom' in self.ticks_where,
                               top='top' in self.ticks_where,
                               left='left' in self.ticks_where,
                               right='right' in self.ticks_where)

        if self.yticklabels_right or (self.yticklabels_right is None and
                                      self.ytitle_right is True):
            self._axes.yaxis.tick_right()

    def _cax_rel2abs_rect(self, rel_rect, cax=None):
        """Turn a relative axes specification into a absolute one.
        """
        if cax is None:
            cax = self._axes
        bbox = cax.get_position()
        l, b, w, h = bbox.x0, bbox.y0, bbox.width, bbox.height
        cl = l + w * rel_rect[0]
        cb = b + h * rel_rect[1]
        try:
            cw = w * rel_rect[2]
            ch = h * rel_rect[3]
        except IndexError:
            return cl, cb
        return cl, cb, cw, ch

    def plot_legend(self, grid=False, labels_handles=None):
        """Add a legend
        """
        if self._use_legend:

            if labels_handles:
                labels, handles = zip(*labels_handles.items())
            else:
                handles, labels = self._legend_handles, self._legend_labels

            if self.legend_reverse:
                handles, labels = handles[::-1], labels[::-1]

            # Limit minimum size of markers that appear in legend
            should_auto_scale_legend_markers = (
                (self.legend_marker_scale is None) and  # not already set
                hasattr(self, '_markersize') and  # is a valid parameter
                self._markersize < 3  # and is small
            )
            if should_auto_scale_legend_markers:
                self.legend_marker_scale = 3 / self._markersize

            opts = {
                'title': (self.z_coo if self.ztitle is None else self.ztitle),
                'loc': self.legend_loc,
                'fontsize': self.fontsize_zlabels,
                'frameon': self.legend_frame,
                'numpoints': 1,
                'scatterpoints': 1,
                'handlelength': self.legend_handlelength,
                'markerscale': self.legend_marker_scale,
                'labelspacing': self.legend_label_spacing,
                'columnspacing': self.legend_column_spacing,
                'bbox_to_anchor': self.legend_bbox,
                'ncol': self.legend_ncol
            }

            if grid:
                bb = opts['bbox_to_anchor']
                if bb is None:
                    opts['bbox_to_anchor'] = (1, 0.5, 0, 0)
                    opts['loc'] = 'center left'
                else:
                    loc = opts['loc']
                    # will get warning for 'best'
                    opts['loc'] = 'center' if loc in ('best', 0) else loc
                lgnd = self._fig.legend(handles, labels, **opts)
            else:
                lgnd = self._axes.legend(handles, labels, **opts)

            lgnd.get_title().set_fontsize(self.fontsize_ztitle)

            if self.legend_marker_alpha is not None:
                for l in lgnd.legendHandles:
                    l.set_alpha(1.0)

    def set_mappable(self):
        """Mappale object for colorbars.
        """
        from matplotlib.cm import ScalarMappable
        self.mappable = ScalarMappable(cmap=self.cmap, norm=self._color_norm)
        self.mappable.set_array([])

    def plot_colorbar(self, grid=False):
        """Add a colorbar to the data.
        """

        if self._use_colorbar:
            # Whether the colorbar should clip at either end
            extendmin = (self.vmin is not None) and (self.vmin > self._zmin)
            extendmax = (self.vmax is not None) and (self.vmax < self._zmax)
            extend = ('both' if extendmin and extendmax else
                      'min' if extendmin else
                      'max' if extendmax else
                      'neither')

            opts = {'extend': extend, 'ticks': self.zticks}

            if self.colorbar_relative_position:
                opts['cax'] = self._fig.add_axes(
                    self._cax_rel2abs_rect(self.colorbar_relative_position))

            if grid:
                opts['ax'] = self._fig.axes
                opts['anchor'] = (0.5, 0.5)

            self._cbar = self._fig.colorbar(
                self.mappable, **opts, **self.colorbar_opts)

            self._cbar.ax.tick_params(labelsize=self.fontsize_zlabels)

            self._cbar.ax.set_title(
                self._ctitle, fontsize=self.fontsize_ztitle,
                color=self.colorbar_color if self.colorbar_color else None)

            if self.colorbar_color:
                self._cbar.ax.yaxis.set_tick_params(
                    color=self.colorbar_color, labelcolor=self.colorbar_color)
                self._cbar.outline.set_edgecolor(self.colorbar_color)

    def set_panel_label(self):
        if self.panel_label is not None:
            self._axes.text(*self.panel_label_loc, self.panel_label,
                            transform=self._axes.transAxes,
                            fontsize=self.fontsize_panel_label,
                            color=self.panel_label_color,
                            ha='left', va='top')

    def show(self):
        import matplotlib.pyplot as plt
        if self.return_fig:
            plt.close(self._fig)
            return self._fig

    def prepare_plot(self):
        """Do all the things that every plot has.
        """
        self.prepare_axes()
        self.set_axes_labels()
        self.set_axes_scale()
        self.set_axes_range()
        self.set_spans()
        self.set_gridlines()
        self.set_tick_marks()

# --------------------------------------------------------------------------- #


def mpl_multi_plot(fn):
    """Decorate a plotting function to plot a grid of values.
    """

    @functools.wraps(fn)
    def multi_plotter(ds, *args, row=None, col=None, hspace=None, wspace=None,
                      tight_layout=True, **kwargs):

        if (row is None) and (col is None):
            return fn(ds, *args, **kwargs)

        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        # Set some global parameters
        p = fn(ds, *args, **kwargs, call=False)
        p.prepare_data_multi_grid()

        kwargs['vmin'] = kwargs.pop('vmin', p.vmin)
        kwargs['vmax'] = kwargs.pop('vmax', p.vmax)

        # split the dataset into its respective rows and columns
        ds_r_c, nrows, ncols = calc_row_col_datasets(ds, row=row, col=col)

        figsize = kwargs.pop('figsize', (3 * ncols, 3 * nrows))
        return_fig = kwargs.pop('return_fig', PLOTTER_DEFAULTS['return_fig'])

        # generate a figure for all the plots to use
        p._fig = plt.figure(figsize=figsize, dpi=100,
                            constrained_layout=tight_layout)
        p._fig.set_constrained_layout_pads(hspace=hspace, wspace=wspace)
        # and a gridspec to position them
        gs = GridSpec(nrows=nrows, ncols=ncols, figure=p._fig,
                      hspace=hspace, wspace=wspace)

        # want to collect all entries for legend
        labels_handles = {}

        # range through rows and do subplots
        for i, ds_r in enumerate(ds_r_c):
            skws = {'legend': False, 'colorbar': False}

            # if not last row
            if i != nrows - 1:
                skws['xticklabels_hide'] = True
                skws['xtitle'] = ''

            # range through columns
            for j, sub_ds in enumerate(ds_r):

                if hspace == 0 and wspace == 0:
                    ticks_where = []
                    if j == 0:
                        ticks_where.append('left')
                    if i == 0:
                        ticks_where.append('top')
                    if j == ncols - 1:
                        ticks_where.append('right')
                    if i == nrows - 1:
                        ticks_where.append('bottom')
                    skws['ticks_where'] = ticks_where

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
                    # XXX: if number of cols==1 this hide yaxis - want both
                    row_val = prettify(ds[row].values[i])
                    skws['ytitle_right'] = True
                    skws['ytitle'] = "{} = {}".format(row, row_val)

                sP = fn(sub_ds, *args, add_to_fig=p._fig, call='both',
                        subplot=gs[i, j], **{**kwargs, **skws})

                try:
                    labels_handles.update(dict(zip(sP._legend_labels,
                                                   sP._legend_handles)))
                except AttributeError:
                    pass

        # make sure all have the same plot ranges
        xmins, xmaxs = zip(*(gax.get_xlim() for gax in p._fig.axes))
        ymins, ymaxs = zip(*(gax.get_ylim() for gax in p._fig.axes))
        xmin, xmax = min(xmins), max(xmaxs)
        ymin, ymax = min(ymins), max(ymaxs)
        for gax in p._fig.axes:
            gax.set_xlim(xmin, xmax)
            gax.set_ylim(ymin, ymax)

        # add global legend or colorbar
        p.plot_legend(grid=True, labels_handles=labels_handles)
        p.plot_colorbar(grid=True)

        if return_fig:
            plt.close(p._fig)
            return p._fig

    return multi_plotter


# --------------------------------------------------------------------------- #

class LinePlot(PlotterMatplotlib, AbstractLinePlot):
    """
    """

    def __init__(self, ds, x, y, z=None, *, y_err=None, x_err=None, **kwargs):
        super().__init__(ds, x, y, z=z, y_err=y_err, x_err=x_err, **kwargs)

    def plot_lines(self):
        """
        """
        for data in self._gen_xy():
            col = next(self._cols)

            line_opts = {
                'c': col,
                'lw': next(self._lws),
                'marker': next(self._mrkrs),
                'markersize': self._markersize,
                'markeredgecolor': col[:3] + (self.marker_alpha * col[3],),
                'markerfacecolor': col[:3] + (self.marker_alpha * col[3] / 2,),
                'label': next(self._zlbls),
                'zorder': next(self._zordrs),
                'linestyle': next(self._lines),
                'rasterized': self.rasterize,
            }

            if ('ye' in data) or ('xe' in data):
                self._axes.errorbar(data['x'], data['y'],
                                    yerr=data.get('ye', None),
                                    xerr=data.get('xe', None),
                                    ecolor=col,
                                    capsize=self.errorbar_capsize,
                                    capthick=self.errorbar_capthick,
                                    elinewidth=self.errorbar_linewidth,
                                    **line_opts)
            else:
                # add line to axes, with options cycled through
                self._axes.plot(data['x'], data['y'], **line_opts)

            self._legend_handles, self._legend_labels = \
                self._axes.get_legend_handles_labels()

    def __call__(self):
        self.prepare_data_single()
        # matplotlib preparation
        self.prepare_plot()
        self.plot_lines()
        self.plot_legend()
        self.plot_colorbar()
        self.set_panel_label()
        return self.show()


@mpl_multi_plot
@intercept_call_arg
def lineplot(ds, x, y, z=None, y_err=None, x_err=None, **plot_opts):
    """From  ``ds`` plot lines of ``y`` as a function of ``x``, optionally for
    varying ``z``.

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
    return LinePlot(ds, x, y, z, y_err=y_err, x_err=x_err, **plot_opts)


class AutoLinePlot(LinePlot):
    def __init__(self, x, y_z, **lineplot_opts):
        ds = auto_xyz_ds(x, y_z)
        super().__init__(ds, 'x', 'y', z='z', **lineplot_opts)


def auto_lineplot(x, y_z, **lineplot_opts):
    """Auto version of :func:`~xyzpy.lineplot` that accepts array arguments
    by converting them to a ``Dataset`` first.
    """
    return AutoLinePlot(x, y_z, **lineplot_opts)()


# --------------------------------------------------------------------------- #

_SCATTER_ALT_DEFAULTS = (
    ('legend_handlelength', 0),
)


class Scatter(PlotterMatplotlib, AbstractScatter):

    def __init__(self, ds, x, y, z=None, **kwargs):
        # set some scatter specific options
        for k, default in _SCATTER_ALT_DEFAULTS:
            if k not in kwargs:
                kwargs[k] = default
        super().__init__(ds, x, y, z, **kwargs)

    def plot_scatter(self):
        """
        """
        self._legend_handles = []
        self._legend_labels = []

        for data in self._gen_xy():
            if 'c' in data:
                col = data['c']
            else:
                col = next(self._cols)

            scatter_opts = {
                'c': col,
                'marker': next(self._mrkrs),
                's': self._markersize,
                'alpha': self.marker_alpha,
                'label': next(self._zlbls),
                'zorder': next(self._zordrs),
                'rasterized': self.rasterize,
            }

            if 'c' in data:
                scatter_opts['cmap'] = self.cmap

            self._legend_handles.append(
                self._axes.scatter(data['x'], data['y'], **scatter_opts))
            self._legend_labels.append(
                scatter_opts['label'])

    def __call__(self):
        self.prepare_data_single()
        # matplotlib preparation
        self.prepare_plot()
        self.plot_scatter()
        self.plot_legend()
        self.plot_colorbar()
        self.set_panel_label()
        return self.show()


@mpl_multi_plot
@intercept_call_arg
def scatter(ds, x, y, z=None, y_err=None, x_err=None, **plot_opts):
    """From  ``ds`` plot a scatter of ``y`` against ``x``, optionally for
    varying ``z``.

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
    return Scatter(ds, x, y, z, y_err=y_err, x_err=x_err, **plot_opts)


class AutoScatter(Scatter):

    def __init__(self, x, y_z, **scatter_opts):
        ds = auto_xyz_ds(x, y_z)
        super().__init__(ds, 'x', 'y', z='z', **scatter_opts)


def auto_scatter(x, y_z, **scatter_opts):
    """Auto version of :func:`~xyzpy.scatter` that accepts array arguments
    by converting them to a ``Dataset`` first.
    """
    return AutoScatter(x, y_z, **scatter_opts)


# --------------------------------------------------------------------------- #

_HISTOGRAM_SPECIFIC_OPTIONS = {
    'stacked': False,
}

_HISTOGRAM_ALT_DEFAULTS = {
    'xtitle': 'x',
    'ytitle': 'f(x)',
}


class Histogram(PlotterMatplotlib, AbstractHistogram):

    def __init__(self, ds, x, z=None, **kwargs):

        # Set the alternative defaults
        for opt, default in _HISTOGRAM_ALT_DEFAULTS.items():
            if opt not in kwargs:
                kwargs[opt] = default

        # Set histogram specfic options
        for opt, default in _HISTOGRAM_SPECIFIC_OPTIONS.items():
            setattr(self, opt, kwargs.pop(opt, default))

        super().__init__(ds, x, None, z=z, **kwargs)

    def plot_histogram(self):
        from matplotlib.patches import Rectangle, Polygon

        def gen_ind_plots():
            for data in self._gen_xy():
                col = next(self._cols)

                edgecolor = col[:3] + (self.marker_alpha * col[3],)
                facecolor = col[:3] + (self.marker_alpha * col[3] / 4,)
                linewidth = next(self._lws)
                zorder = next(self._zordrs)
                label = next(self._zlbls)

                handle = Rectangle((0, 0), 1, 1, color=facecolor, ec=edgecolor)

                yield (data['x'], edgecolor, facecolor, linewidth, zorder,
                       label, handle)

        xs, ecs, fcs, lws, zds, lbs, hnds = zip(*gen_ind_plots())

        histogram_opts = {
            'label': lbs,
            'bins': self.bins,
            'density': True,
            'histtype': 'stepfilled',
            'fill': True,
            'stacked': self.stacked,
            'rasterized': self.rasterize,
        }

        _, _, patches = self._axes.hist(xs, **histogram_opts)

        # Need to set varying colors, linewidths etc seperately
        for patch, ec, fc, lw, zd in zip(patches, ecs, fcs, lws, zds):

            # patch is not iterable if only one set of data created
            if isinstance(patch, Polygon):
                patch = (patch,)

            for sub_patch in patch:
                sub_patch.set_edgecolor(ec)
                sub_patch.set_facecolor(fc)
                sub_patch.set_linewidth(lw)
                sub_patch.set_zorder(zd)

        # store handles for legend
        self._legend_handles, self._legend_labels = hnds, lbs

    def __call__(self):
        # Core preparation
        self.prepare_data_single()
        # matplotlib preparation
        self.prepare_plot()
        self.plot_histogram()
        self.plot_legend()
        self.plot_colorbar()
        self.set_panel_label()
        return self.show()


@mpl_multi_plot
@intercept_call_arg
def histogram(ds, x, z=None, **plot_opts):
    """Dataset histogram.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to plot.
    x : str, sequence of str
        The variable(s) to plot the probability density of. If sequence, plot a
        histogram of each instead of using a ``z`` coordinate.
    z : str, optional
        If given, range over this coordinate a plot a histogram for each.
    row : str, optional
        Dimension to vary over as a function of rows.
    col : str, optional
        Dimension to vary over as a function of columns.
    plot_opts
        See ``xyzpy.plot.core.PLOTTER_DEFAULTS``.
    """
    return Histogram(ds, x, z=z, **plot_opts)


class AutoHistogram(Histogram):

    def __init__(self, x, **histogram_opts):
        ds = auto_xyz_ds(x)
        super().__init__(ds, 'x', **histogram_opts)


def auto_histogram(x, **histogram_opts):
    """Auto version of :func:`~xyzpy.histogram` that accepts array arguments
    by converting them to a ``Dataset`` first.
    """
    return AutoHistogram(x, **histogram_opts)()


# --------------------------------------------------------------------------- #

_HEATMAP_ALT_DEFAULTS = (
    ('legend', False),
    ('colorbar', True),
    ('colormap', 'inferno'),
    ('method', 'pcolormesh'),
    ('gridlines', False),
    ('rasterize', True),
)


class HeatMap(PlotterMatplotlib, AbstractHeatMap):

    def __init__(self, ds, x, y, z, **kwargs):
        # set some heatmap specific options
        for k, default in _HEATMAP_ALT_DEFAULTS:
            if k not in kwargs:
                kwargs[k] = default
        super().__init__(ds, x, y, z, **kwargs)

    def plot_heatmap(self):
        """Plot the data as a heatmap.
        """
        self.calc_color_norm()
        self._heatmap = getattr(self._axes, self.method)(
            self._heatmap_x,
            self._heatmap_y,
            self._heatmap_var,
            norm=self._color_norm,
            cmap=xyz_colormaps(self.colormap),
            rasterized=self.rasterize)

    def __call__(self):
        # Core preparation
        self.prepare_data_single()
        # matplotlib preparation
        self.prepare_plot()
        self.plot_heatmap()
        self.plot_colorbar()
        self.set_panel_label()
        return self.show()


@mpl_multi_plot
@intercept_call_arg
def heatmap(ds, x, y, z, **kwargs):
    """From  ``ds`` plot variable ``z`` as a function of ``x`` and ``y`` using
    a 2D heatmap.

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
    return HeatMap(ds, x, y, z, **kwargs)


class AutoHeatMap(HeatMap):

    def __init__(self, x, **heatmap_opts):
        ds = auto_xyz_ds(x)
        super().__init__(ds, 'x', **heatmap_opts)


def auto_heatmap(x, **heatmap_opts):
    """Auto version of :func:`~xyzpy.heatmap` that accepts array arguments
    by converting them to a ``Dataset`` first.
    """
    return AutoHeatMap(x, **heatmap_opts)()


# --------------- Miscellenous matplotlib plotting functions ---------------- #

def choose_squarest_grid(x):
    p = x ** 0.5
    if p.is_integer():
        m = n = int(p)
    else:
        m = int(round(p))
        p = int(p)
        n = p if m * p >= x else p + 1
    return m, n


def visualize_matrix(x, figsize=(4, 4),
                     colormap='Greys',
                     touching=False,
                     zlims=(None, None),
                     gridsize=None,
                     tri=None,
                     return_fig=True):
    """Plot the elements of one or more matrices.

    Parameters
    ----------
    x : array or iterable of arrays
        2d-matrix or matrices to plot.
    figsize : tuple
        Total size of plot.
    colormap : str
        Colormap to use to weight elements.
    touching : bool
        If plotting more than one matrix, whether the edges should touch.
    zlims:
        Scaling parameters for the element colorings, (i.e. if these are
        set then the weightings are not normalized).
    return_fig : bool
        Whether to return the figure created or just show it.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize, dpi=100)
    if isinstance(x, np.ndarray):
        x = (x,)

    nx = len(x)
    if gridsize:
        m, n = gridsize
    else:
        m, n = choose_squarest_grid(nx)
    subplots = tuple((m, n, i) for i in range(1, nx + 1))

    for img, subplot in zip(x, subplots):

        if tri is not None:
            if tri not in {'upper', 'lower'}:
                raise ValueError("'tri' should be one of {'upper', 'lower}.")

            ma_fn = np.tril if tri == 'upper' else np.triu
            img = np.ma.array(img, mask=ma_fn(np.ones_like(img), k=-1))

        ax = fig.add_subplot(*subplot)
        ax.imshow(img, cmap=xyz_colormaps(colormap), interpolation='nearest',
                  vmin=zlims[0], vmax=zlims[1])
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
