"""
Functions for plotting datasets nicely.
"""
# TODO: custom xtick labels                                                   #
# TODO: annotations, arbitrary text                                           #
# TODO: docs                                                                  #

import functools
import itertools
import collections

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
from .color import xyz_colormaps, cimple


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
        mpl.rcParams['font.weight'] = self.font_weight
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
                             fontsize=self.fontsize_title, pad=self.title_pad)

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
                hasattr(self, '_marker_size') and  # is a valid parameter
                self._marker_size < 3  # and is small
            )
            if should_auto_scale_legend_markers:
                self.legend_marker_scale = 3 / self._marker_size

            opts = {
                'title': (self.z_coo if self.ztitle is None else self.ztitle),
                'loc': self.legend_loc,
                'fontsize': self.fontsize_zlabels,
                'frameon': self.legend_frame,
                'framealpha': self.legend_frame_alpha,
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
                for legendline in lgnd.legendHandles:
                    legendline.set_alpha(1.0)

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
                      tight_layout=True, coltitle=None, rowtitle=None,
                      **kwargs):

        if (row is None) and (col is None):
            return fn(ds, *args, **kwargs)

        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        # Set some global parameters
        p = fn(ds, *args, **kwargs, call=False)
        p.prepare_data_multi_grid()

        kwargs['vmin'] = kwargs.pop('vmin', p.vmin)
        kwargs['vmax'] = kwargs.pop('vmax', p.vmax)

        coltitle = col if coltitle is None else coltitle
        rowtitle = row if rowtitle is None else rowtitle

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
                    skws['title'] = "{} = {}".format(coltitle, col_val)
                    fx = 'fontsize_xtitle'
                    skws['fontsize_title'] = kwargs.get(
                        fx, PLOTTER_DEFAULTS[fx])

                # label each row
                if (j == ncols - 1) and (row is not None):
                    # XXX: if number of cols==1 this hide yaxis - want both
                    row_val = prettify(ds[row].values[i])
                    skws['ytitle_right'] = True
                    skws['ytitle'] = "{} = {}".format(rowtitle, row_val)

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
                'markersize': self._marker_size,
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
                col = [next(self._cols)]

            scatter_opts = {
                'c': col,
                'marker': next(self._mrkrs),
                's': self._marker_size,
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
    return AutoScatter(x, y_z, **scatter_opts)()


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

        # add extra coords since they *bound* the quads placed -> want ticks
        #     at center of quads
        X = self._heatmap_x
        av_x_bin = np.mean(np.abs(X[:-1] - X[1:]))
        X = np.append(X - av_x_bin / 2, X[-1] + av_x_bin / 2)

        Y = self._heatmap_y
        av_Y_bin = np.mean(np.abs(Y[:-1] - Y[1:]))
        Y = np.append(Y - av_Y_bin / 2, Y[-1] + av_Y_bin / 2)

        self._heatmap = getattr(self._axes, self.method)(
            X, Y, self._heatmap_var,
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
        super().__init__(ds, 'y', 'z', 'x', **heatmap_opts)


def auto_heatmap(x, **heatmap_opts):
    """Auto version of :func:`~xyzpy.heatmap` that accepts array arguments
    by converting them to a ``Dataset`` first.
    """
    return AutoHeatMap(x, **heatmap_opts)()


# --------------- Miscellenous matplotlib plotting functions ---------------- #

def setup_fig_ax(
    nrows=1,
    ncols=1,
    facecolor=None,
    rasterize=False,
    rasterize_dpi=300,
    figsize=(5, 5),
    ax=None,
    **kwargs,
):
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)

        fig.patch.set_alpha(0.0)
    else:
        fig = None

    if not isinstance(ax, np.ndarray):
        # squeezed to single axis
        ax.set_aspect('equal')
        ax.axis('off')

    if facecolor is not None:
        fig.patch.set_facecolor(facecolor)

    if rasterize:
        ax.set_rasterization_zorder(0)
        if fig is not None:
            fig.set_dpi(rasterize_dpi)

    return fig, ax


def show_and_close(fn):

    @functools.wraps(fn)
    def wrapped(*args, show_and_close=True, **kwargs):
        import warnings
        import matplotlib.pyplot as plt

        # remove annoying regular warning about all-nan slices
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action='ignore',
                message='All-NaN slice',
            )
            fig, ax = fn(*args, **kwargs)

        if fig is not None:
            if show_and_close:
                plt.show()
                plt.close(fig)

        return fig, ax

    return wrapped


def choose_squarest_grid(x):
    p = x ** 0.5
    if p.is_integer():
        m = n = int(p)
    else:
        m = int(round(p))
        p = int(p)
        n = p if m * p >= x else p + 1
    return m, n


def _compute_hue(z):
    # the constant rotation is to have blue and orange as + and -
    # negation puts +i as green -i as pink
    return ((-np.angle(z) + 9 * np.pi / 8) / (2 * np.pi)) % 1


def to_colors(
    zs,
    magscale='linear',
    max_mag=None,
    alpha_map=True,
    alpha_pow=1/2,
):
    import matplotlib as mpl
    arraymag = np.abs(zs)

    if magscale == 'linear':
        mapped_mag = arraymag
    elif magscale == 'log':
        # XXX: need some kind of cutoff?
        raise NotImplementedError("log scale not implemented.")
    else:
        # more robust way to 'flatten' the perceived magnitude
        mapped_mag = arraymag**magscale

    if max_mag is None:
        max_mag = np.max(mapped_mag)

    hue = _compute_hue(zs)
    sat = mapped_mag / max_mag
    val = np.tile(1.0, hue.shape)
    zs = mpl.colors.hsv_to_rgb(np.stack((hue, sat, val), axis=-1))

    if alpha_map:
        # append alpha channel
        zalpha = (mapped_mag / max_mag)**alpha_pow
        zs = np.concatenate([zs, np.expand_dims(zalpha, -1)], axis=-1)

    return zs, mapped_mag, max_mag


def add_visualize_legend(
    ax,
    complexobj,
    max_mag,
    max_projections,
    legend_loc='auto',
    legend_size=0.15,
    legend_bounds=None,
    legend_resolution=3,
):
    import matplotlib as mpl

    # choose where to put the legend
    if legend_bounds is None:
        if legend_loc == 'auto':
            if (max_projections <= 2):
                # move compass and legends beyond the plot rectangle, which
                # will be filled when there are only 2 plot dimensions
                legend_loc = (1 - 0.03, 0.0 - legend_size + 0.03)
            else:
                # occupy space within the rectangle
                legend_loc = (1.0 - legend_size, 0.0)
        legend_bounds = (*legend_loc, legend_size, legend_size)

    lax = ax.inset_axes(legend_bounds)
    lax.axis('off')
    lax.set_aspect('equal')

    num = legend_resolution * 2 + 1
    if complexobj:
        re = np.linspace(-1, 1, num=num).reshape(1, -1)
        im = 1j * np.linspace(1, -1, num=num).reshape(-1, 1)
        z = re + im
    else:
        re = np.linspace(-1, 1, num=num)
        # repeat a few times to make the bar thick enough
        z = np.tile(re, (max(1, legend_resolution // 3), 1))

    zmag = np.abs(z)
    mask = zmag > 1.0
    z[mask] = zmag[mask] = 0.0

    # compute the color for each point
    hue = _compute_hue(z)
    sat = zmag
    val = np.tile(1.0, hue.shape)

    # convert to rgb
    bars = mpl.colors.hsv_to_rgb(np.stack((hue, sat, val), axis=-1))
    # add alpha channel
    bars = np.concatenate([bars, np.tile(1.0, hue.shape + (1,))], -1)
    # add make area outside disk transparent
    bars[..., 3][mask] = 0.0

    # plot the actual colorbar legend
    lax.imshow(bars)

    # add axis orientation labels
    lax.text(1.02, 0.5, '$+1$', ha='left', va='center',
             transform=lax.transAxes, color=(.5, .5, .5), size=6)
    lax.text(-0.02, 0.5, '$-1$', ha='right', va='center',
             transform=lax.transAxes, color=(.5, .5, .5), size=6)
    if complexobj:
        lax.text(0.5, -0.02, '$-j$', ha='center', va='top',
                 transform=lax.transAxes, color=(.5, .5, .5), size=6)
        lax.text(0.5, 1.02, '$+j$', ha='center', va='bottom',
                 transform=lax.transAxes, color=(.5, .5, .5), size=6)

    # show the overall scale
    if complexobj:
        overall_scale_opts = {'x': .85, 'y': .15, 'ha': 'left'}
    else:
        overall_scale_opts = {'x': .5, 'y': -0.15, 'ha': 'center'}

    lax.text(
        s=f'$\\times {float(max_mag):.3}$', va='top',
        **overall_scale_opts, transform=lax.transAxes,
        color=(.5, .5, .5), size=6
    )

def make_ax_square_after_plotted(ax):
    xmin, xmax = ax.get_xlim()
    xrange = abs(xmax - xmin)
    ymin, ymax = ax.get_ylim()
    yrange = abs(ymax - ymin)

    # pad either x or y to make square
    if xrange > yrange:
        ypad = (xrange - yrange) / 2
        if ymin > ymax:
            # fipped y axis
            ax.set_ylim(ymin + ypad, ymax - ypad)
        else:
            ax.set_ylim(ymin - ypad, ymax + ypad)

    elif yrange > xrange:
        xpad = (yrange - xrange) / 2
        if xmin > xmax:
            # flipped x axis
            ax.set_xlim(xmin + xpad, xmax - xpad)
        else:
            ax.set_xlim(xmin - xpad, xmax + xpad)


def handle_sequence_of_arrays(f):
    """Simple wrapper to handle sequence of arrays as input to e.g.
    ``visualize_tensor``.
    """

    @functools.wraps(f)
    def wrapped(array, *args, show_and_close=True, **kwargs):
        import matplotlib.pyplot as plt

        if (
            isinstance(array, (tuple, list)) and
            all(hasattr(x, 'shape') for x in array)
        ):
            # assume sequence of tensors to plot
            figsize = kwargs.get('figsize', (5, 5))
            rasterize_dpi = kwargs.get('rasterize_dpi', 300)
            nplot = len(array)
            fig, axs = plt.subplots(
                1, nplot, figsize=figsize, squeeze=False, sharey=True,
            )
            fig.set_dpi(rasterize_dpi)
            fig.patch.set_alpha(0.0)

            # plot all tensors with same magnitude scale
            kwargs.setdefault("max_mag", max(np.max(np.abs(x)) for x in array))

            # only show legend on last plot
            legend = kwargs.pop('legend', False)

            for i in range(nplot):
                f(
                    array[i], *args,
                    ax=axs[0, i],
                    # only show legend on last plot
                    legend=(legend and i == nplot - 1),
                    show_and_close=False,
                    **kwargs
                )
                make_ax_square_after_plotted(axs[0, i])

            if show_and_close:
                plt.show()
                plt.close(fig)
            return fig, axs

        else:
            # treat as single tensor
            return f(array, *args, show_and_close=show_and_close, **kwargs)

    return wrapped


@handle_sequence_of_arrays
@show_and_close
def visualize_matrix(
    array,
    max_mag=None,
    magscale='linear',
    alpha_map=True,
    alpha_pow=1/2,
    legend=False,
    legend_loc='auto',
    legend_size=0.15,
    legend_bounds=None,
    legend_resolution=3,
    facecolor=None,
    rasterize=4096,
    rasterize_dpi=300,
    figsize=(5, 5),
    ax=None,
):
    # can only plot numpy
    array = np.asarray(array)
    if array.ndim == 1:
        # draw vectors as diagonals
        array = np.diag(array)

    if isinstance(rasterize, (float, int)):
        # only turn on above a certain size
        rasterize = array.size > rasterize

    fig, ax = setup_fig_ax(
        facecolor=facecolor,
        rasterize=rasterize,
        rasterize_dpi=rasterize_dpi,
        figsize=figsize,
        ax=ax,
    )

    zs, _, max_mag = to_colors(
        array,
        magscale=magscale,
        max_mag=max_mag,
        alpha_map=alpha_map,
        alpha_pow=alpha_pow,
    )
    ax.imshow(zs, interpolation='nearest', zorder=-1)

    if legend:
        add_visualize_legend(
            ax=ax,
            complexobj=np.iscomplexobj(array),
            max_mag=max_mag,
            max_projections=2,
            legend_loc=legend_loc,
            legend_size=legend_size,
            legend_bounds=legend_bounds,
            legend_resolution=legend_resolution,
        )

    return fig, ax


@handle_sequence_of_arrays
@show_and_close
def visualize_tensor(
    array,
    max_projections=None,
    angles=None,
    scales=None,
    projection_overlap_spacing=1.05,
    skew_factor=0.05,
    spacing_factor=1.0,
    magscale='linear',
    size_map=True,
    size_pow=1/2,
    size_scale=1.0,
    alpha_map=True,
    alpha_pow=1/2,
    alpha=0.8,
    marker='o',
    linewidths=0,
    show_lattice=True,
    lattice_opts=None,
    compass=False,
    compass_loc='auto',
    compass_size=0.1,
    compass_bounds=None,
    compass_labels=None,
    compass_opts=None,
    max_mag=None,
    legend=False,
    legend_loc='auto',
    legend_size=0.15,
    legend_bounds=None,
    legend_resolution=3,
    interleave_projections=False,
    reverse_projections=False,
    facecolor=None,
    rasterize=4096,
    rasterize_dpi=300,
    figsize=(5, 5),
    ax=None,
):
    """Visualize all entries of a tensor, with indices mapped into the plane
    and values mapped into a color wheel.

    Parameters
    ----------
    array : numpy.ndarray
        The tensor to visualize.
    skew_factor : float, optional
        When there are more than two dimensions, a factor to scale the
        rotations by to avoid overlapping data points.
    size_map : bool, optional
        Whether to map the tensor value magnitudes to marker size.
    size_scale : float, optional
        An overall factor to scale the marker size by.
    alpha_map : bool, optional
        Whether to map the tensor value magnitudes to marker alpha.
    alpha_pow : float, optional
        The power to raise the magnitude to when mapping to alpha.
    alpha : float, optional
        The overall alpha to use for all markers if ``not alpha_map``.
    show_lattice : bool, optional
        Show a small grey dot for every 'lattice' point regardless of value.
    lattice_opts : dict, optional
        Options to pass to ``maplotlib.Axis.scatter`` for the lattice points.
    linewidths : float, optional
        The linewidth to use for the markers.
    marker : str, optional
        The marker to use for the markers.
    figsize : tuple, optional
        The size of the figure to create, if ``ax`` is not provided.
    ax : matplotlib.Axis, optional
        The axis to draw to. If not provided, a new figure will be created.

    Returns
    -------
    fig : matplotlib.Figure
        The figure containing the plot, or ``None`` if ``ax`` was provided.
    ax : matplotlib.Axis
        The axis containing the plot.
    """
    import matplotlib as mpl

    # can only plot numpy
    array = np.asarray(array)

    if isinstance(rasterize, (float, int)):
        # only turn on above a certain size
        rasterize = array.size > rasterize

    fig, ax = setup_fig_ax(
        facecolor=facecolor,
        rasterize=rasterize,
        rasterize_dpi=rasterize_dpi,
        figsize=figsize,
        ax=ax,
    )

    auto_angles = angles is None
    if scales == "equal":
        scales = [1] * array.ndim
    auto_scales = scales is None

    if max_projections is None:
        max_projections = array.ndim

    # map each dimension to an angle
    if not auto_angles:
        angles = np.array(angles)
    else:
        # if max_projections == array.ndim, then each dimension has its own
        # angle, if max_projections < array.dim, then we will
        # reuse the same angles, initially round robin distributed
        angles = np.tile(
            np.linspace(0.0, np.pi, max_projections, endpoint=False),
            array.ndim // max_projections + 1
        )[:array.ndim]

        if not interleave_projections:
            # 'fill up' one angle before moving on, rather than round-robin,
            # doing this matches the behavior of fusing adjacent dimensions
            angles = np.sort(angles)

        def angle_modulate(x):
            return x * (x - np.pi / 2) * (x - x[-1])

        # modulate the angles slightly to avoid overlapping data points
        angles += angle_modulate(angles) * skew_factor

    if auto_scales:
        scales = np.empty(angles.shape)
    else:
        scales = np.array(scales)

    # the logic here is, when grouping dimensions into the same angles we
    # need to offset each overlapping dimension by increasing amount
    first_size = {}
    grouped_size = {}
    group_counter = {}
    group_rank = {}
    fastest_varying = []

    iphis = list(enumerate(angles))
    if not reverse_projections:
        iphis.reverse()

    for i, phi in iphis:
        if phi not in first_size:
            # first time we have encountered an axis at this angle
            fastest_varying.append((i, array.shape[i]))
            first_size[phi] = array.shape[i]
            grouped_size[phi] = array.shape[i]
            group_counter[phi] = 1
        else:
            # already an axis at this angle, space this one larger
            grouped_size[phi] *= array.shape[i]
            group_counter[phi] += 1

        # what rank among axes at this angle is i?
        group_rank[i] = group_counter[phi]

        if auto_scales:
            scales[i] = (
                grouped_size[phi] // array.shape[i]
                # put extra space between distinct dimensions
                * projection_overlap_spacing**group_counter[phi]
                # account for spacing out of first dimensions
                / max(1, (first_size[phi] - 1))**spacing_factor
            )

    eff_width = max(grouped_size.values())
    eff_ndim = max_projections

    # define the core mappings of coordinate to 2D plane

    def xcomponent(i, coo):
        return scales[i] * np.sin(angles[i]) * coo

    def ycomponent(i, coo):
        return scales[i] * -np.cos(angles[i]) * coo

    # compute projection into 2D coordinates for every index location
    coos = np.indices(array.shape)
    x = np.zeros(array.shape)
    y = np.zeros(array.shape)
    for i, coo in enumerate(coos):
        x += xcomponent(i, coo)
        y += ycomponent(i, coo)

    # compute colors
    zs, mapped_mag, max_mag = to_colors(
        array.flat,
        magscale=magscale,
        max_mag=max_mag,
        alpha_map=alpha_map,
        alpha_pow=alpha_pow,
    )

    # compute a sensible base size based on expected density of points
    base_size = size_scale * 3000 / (eff_width * eff_ndim**1.2)
    if size_map:
        s = base_size * (mapped_mag / max_mag)**size_pow
    else:
        s = base_size

    if show_lattice:
        # put a small grey line on every edge
        ls = []
        for i, isize in fastest_varying:
            other_shape = array.shape[:i] + array.shape[i + 1:]
            other_coos = np.indices(other_shape)
            coo_start = np.insert(other_coos, i, 0, 0)
            coo_stop = np.insert(other_coos, i, isize - 1, 0)

            xi = np.zeros(coo_start.shape)
            yi = np.zeros(coo_start.shape)

            for i, coo in enumerate(coo_start):
                xi += xcomponent(i, coo)
                yi += ycomponent(i, coo)

            xf = np.zeros(coo_start.shape)
            yf = np.zeros(coo_start.shape)

            for i, coo in enumerate(coo_stop):
                xf += xcomponent(i, coo)
                yf += ycomponent(i, coo)

            li = np.stack((xi.flat, yi.flat), 1)
            lf = np.stack((xf.flat, yf.flat), 1)

            ls.append(np.stack((li, lf), 1))

        segments = np.concatenate(ls)

        lattice_opts = {} if lattice_opts is None else dict(lattice_opts)
        lattice_opts.setdefault('color', (.6, .6, .6))
        lattice_opts.setdefault(
            'alpha',
            0.01 + 2**(-(array.size**0.2 + eff_ndim**0.8))
        )
        lattice_opts.setdefault('linewidth', 1)
        lattice_opts.setdefault('zorder', -2)
        lines = mpl.collections.LineCollection(segments, **lattice_opts)
        ax.add_collection(lines)

    # plot the actual points
    ax.scatter(
        # mapped variables
        # (reverse the data so that the correct points are shown on top)
        x=x.flat[::-1],
        y=y.flat[::-1],
        c=zs[::-1],
        s=s[::-1] if size_map else s,
        # constants
        alpha=None if alpha_map else alpha,  # folded into color if alpha_map
        linewidths=linewidths,
        marker=marker,
        zorder=-1,
        clip_on=False,
    )

    if compass:
        # choose where to put the compass
        if compass_bounds is None:
            if compass_loc == 'auto':
                if (max_projections <= 2):
                    # move compass and legends beyond the plot rectangle, which
                    # will be filled when there are only 2 plot dimensions
                    compass_loc = (-0.05, 1.0 - compass_size + 0.05)
                else:
                    # occupy space within the rectangle
                    compass_loc = (0.0, 1 - compass_size)
            compass_bounds = (*compass_loc, compass_size, compass_size)

        cax = ax.inset_axes(compass_bounds)
        cax.axis('off')
        cax.set_aspect('equal')

        compass_opts = {} if compass_opts is None else dict(compass_opts)
        compass_opts.setdefault('color', (0.5, 0.5, 0.5))
        compass_opts.setdefault('width', 0.002)
        compass_opts.setdefault('length_includes_head', True)

        if compass_labels is None:
            compass_labels = range(len(angles))
        elif compass_labels is False:
            compass_labels = [''] * len(angles)

        for i, phi in enumerate(angles):
            dx = np.sin(phi) * group_rank[i]
            dy = -np.cos(phi) * group_rank[i]
            cax.arrow(0, 0, dx, dy, **compass_opts)
            cax.text(
                dx, dy, f" {compass_labels[i]}",
                ha='left', va='top',
                color=compass_opts['color'],
                size=6,
                rotation=180 * phi / np.pi - 90,
                rotation_mode='anchor',
            )

    if legend:
        add_visualize_legend(
            ax=ax,
            complexobj=np.iscomplexobj(array),
            max_mag=max_mag,
            max_projections=max_projections,
            legend_loc=legend_loc,
            legend_size=legend_size,
            legend_bounds=legend_bounds,
            legend_resolution=legend_resolution,
        )

    return fig, ax


@functools.lru_cache(16)
def get_neutral_style(draw_color=(.5, .5, .5)):
    return {
        'axes.edgecolor': draw_color,
        'axes.facecolor': (0, 0, 0, 0),
        'axes.grid': True,
        'axes.labelcolor': draw_color,
        'axes.spines.right': False,
        'axes.spines.top': False,
        'figure.facecolor': (0, 0, 0, 0),
        'grid.alpha': 0.1,
        'grid.color': draw_color,
        'legend.frameon': False,
        'text.color': draw_color,
        'xtick.color': draw_color,
        'xtick.minor.visible': True,
        'ytick.color': draw_color,
        'ytick.minor.visible': True,
    }


def use_neutral_style(fn):
    import matplotlib as mpl

    @functools.wraps(fn)
    def new_fn(
        *args,
        use_neutral_style=True,
        draw_color=(.5, .5, .5),
        **kwargs
    ):
        if not use_neutral_style:
            return fn(*args, **kwargs)

        style = get_neutral_style(draw_color=draw_color)

        with mpl.rc_context(style):
            return fn(*args, **kwargs)

    return new_fn


# colorblind palettes by Bang Wong (https://www.nature.com/articles/nmeth.1618)

_COLORS_DEFAULT = (
    '#56B4E9',  # light blue
    '#E69F00',  # orange
    '#009E73',  # green
    '#D55E00',  # red
    '#F0E442',  # yellow
    '#CC79A7',  # purple
    '#0072B2',  # dark blue
)
_COLORS_SORTED = (
    '#0072B2',  # dark blue
    '#56B4E9',  # light blue
    '#009E73',  # green
    '#F0E442',  # yellow
    '#E69F00',  # orange
    '#D55E00',  # red
    '#CC79A7',  # purple
)


def mod_sat(c, mod):
    """Modify the luminosity of rgb color ``c``.
    """
    from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

    h, s, v = rgb_to_hsv(c[:3])
    return (*hsv_to_rgb((h, mod * s, v)), 1.0)


def auto_colors(N):
    import math
    from matplotlib.colors import LinearSegmentedColormap

    if N < len(_COLORS_DEFAULT):
        return _COLORS_DEFAULT[:N]

    cmap = LinearSegmentedColormap.from_list('wong', _COLORS_SORTED)

    xs = list(map(cmap, np.linspace(0, 1.0, N)))

    # modulate color saturation with sine to generate local distinguishability
    # ... but only turn on gradually for increasing number of nodes
    sat_mod_period = min(4, N / 7)
    sat_mod_factor = max(0.0, 2 / 3 * math.tanh((N - 7) / 4))

    return [
        mod_sat(
            c, 1 - sat_mod_factor * math.sin(math.pi * i / sat_mod_period)**2
        )
        for i, c in enumerate(xs)
    ]


def color_to_colormap(c, vdiff=0.5, sdiff=0.25):
    import matplotlib as mpl
    rgb = mpl.colors.to_rgb(c)
    h, s, v = mpl.colors.rgb_to_hsv(rgb)

    vhi = min(1.0, v + vdiff / 2)
    vlo = max(0.0, vhi - vdiff)
    vhi = vlo + vdiff

    shi = min(1.0, s + sdiff / 2)
    slo = max(0.0, shi - sdiff)
    shi = slo + sdiff

    hsv_i = (h, max(slo, 0.0), min(vhi, 1.0))
    hsv_f = (h, min(shi, 1.0), max(vlo, 0.0))

    c1 = mpl.colors.hsv_to_rgb(hsv_i)
    c2 = mpl.colors.hsv_to_rgb(hsv_f)
    cdict = {
        'red': [(0.0, c1[0], c1[0]), (1.0, c2[0], c2[0])],
        'green': [(0.0, c1[1], c1[1]), (1.0, c2[1], c2[1])],
        'blue': [(0.0, c1[2], c1[2]), (1.0, c2[2], c2[2])],
    }
    return mpl.colors.LinearSegmentedColormap('', cdict)


def get_default_cmap(i, vdiff=0.5, sdiff=0.25):
    return color_to_colormap(_COLORS_DEFAULT[i], vdiff=vdiff, sdiff=sdiff)


def to_colormap(c, **autohue_opts):
    import numbers
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    if isinstance(c, mpl.colors.Colormap):
        return c

    if isinstance(c, numbers.Number):
        return cimple(c, **autohue_opts)

    try:
        return plt.get_cmap(c)
    except ValueError:
        return color_to_colormap(c)


def _make_bold(s):
    return r'$\bf{' + s.replace('_', r'\_') + r'}$'


_LINESTYLES_DEFAULT = (
    'solid',
    (0.0, (3, 1)),
    (0.5, (1, 1)),
    (1.0, (3, 1, 1, 1)),
    (1.5, (3, 1, 3, 1, 1, 1)),
    (2.0, (3, 1, 1, 1, 1, 1)),
)


_MARKERS_DEFAULT = (
    'o',
    'X',
    'v',
    's',
    'P',
    'D',
    '^',
    'h',
    '*',
    'p',
    '<',
    'd',
    '8',
    '>',
    'H',
)


def init_mapped_dim(
    sizes,
    domains,
    values,
    labels,
    mapped,
    base_style,
    ds,
    name,
    dim,
    order=None,
    dim_label=None,
    custom_values=None,
    default_values=None,
):
    if isinstance(dim, (tuple, list)) and all(x in ds.dims for x in dim):
        # create a new nested effective dimension
        new_dim = ", ".join(dim)
        ds = ds.stack({new_dim: dim})
        dim = new_dim

    elif (dim is not None) and (dim not in ds.dims):
        # attribute is just manually specified, not mapped to dimension
        base_style[name] = dim
        sizes[name] = 1
        return ds, None

    if (dim is not None) and (order is not None):
        # select and order along dimension
        ds = ds.sel({dim: list(order)})

    if dim is not None:
        ds = ds.dropna(dim, how='all')

        domains[name] = ds[dim].values
        sizes[name] = len(domains[name])
        labels[dim] = _make_bold(dim) if dim_label is None else dim_label
        mapped.add(dim)

        if custom_values is None:
            if default_values is not None:
                if callable(default_values):
                    # allow default values to depend on number of values
                    default_values = default_values(sizes[name])

                values[name] = tuple(
                    x for x, _ in zip(default_values, range(sizes[name]))
                )
        else:
            values[name] = custom_values

    else:
        sizes[name] = 1

    return ds, dim


@show_and_close
@use_neutral_style
def infiniplot(
    ds,
    x,
    y=None,
    *,
    bins=None,
    bins_density=True,
    aggregate=None,
    aggregate_method='median',
    aggregate_err_range=0.5,
    err=None,
    err_style=None,
    err_kws=None,
    xlink=None,
    color=None,
    colors=None,
    color_order=None,
    color_label=None,
    colormap_start=0.0,
    colormap_stop=1.0,
    hue=None,
    hues=None,
    hue_order=None,
    hue_label=None,
    palette=None,
    autohue_start=0.6,
    autohue_sweep=-1.0,
    autohue_opts=None,
    marker=None,
    markers=None,
    marker_order=None,
    marker_label=None,
    markersize=None,
    markersizes=None,
    markersize_order=None,
    markersize_label=None,
    markeredgecolor='white',
    markeredgecolor_order=None,
    markeredgecolor_label=None,
    markeredgecolors=None,
    linewidth=None,
    linewidths=None,
    linewidth_order=None,
    linewidth_label=None,
    linestyle=None,
    linestyles=None,
    linestyle_order=None,
    linestyle_label=None,
    text=None,
    text_formatter=str,
    text_opts=None,
    col=None,
    col_order=None,
    col_label=None,
    row=None,
    row_order=None,
    row_label=None,
    alpha=1.0,
    err_band_alpha=0.1,
    err_bar_capsize=1,
    xlabel=None,
    ylabel=None,
    xlim=None,
    ylim=None,
    xscale=None,
    yscale=None,
    xbase=10,
    ybase=10,
    vspans=(),
    hspans=(),
    span_color=(0.5, 0.5, 0.5),
    span_alpha=0.5,
    span_linewidth=1,
    span_linestyle=':',
    grid=True,
    grid_which='major',
    grid_alpha=0.1,
    legend=True,
    legend_ncol=None,
    legend_merge=False,
    legend_reverse=False,
    legend_entries=None,
    legend_opts=None,
    title=None,
    ax=None,
    axs=None,
    figsize=None,
    height=3,
    width=None,
    hspace=0.12,
    wspace=0.12,
    sharex=True,
    sharey=True,
    **kwargs,
):
    """
    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to plot.
    x : str
        Name of x-axis dimension.
    y : str
        Name of y-axis dimension.
    aggregate : str or tuple[str], optional
        Name of dimension(s) to aggregate before plotting.
    aggregate_method : str, optional
        Aggregation method used to show main line.
    aggregate_err_range : float, optional
        Inter-quantile range to use to show aggregation bands.
    """
    import matplotlib as mpl
    from matplotlib import pyplot as plt
    from matplotlib.ticker import (
        AutoMinorLocator,
        LogLocator,
        NullFormatter,
        ScalarFormatter,
        StrMethodFormatter,
    )
    from matplotlib.lines import Line2D

    autohue_opts = {} if autohue_opts is None else autohue_opts
    autohue_opts.setdefault("val1", 1.0)
    autohue_opts.setdefault("sat1", 0.3)
    autohue_opts.setdefault("val2", 0.6)

    if text is not None:
        text_opts = {} if text_opts is None else text_opts
        text_opts.setdefault('size', 6)
        text_opts.setdefault('horizontalalignment', 'left')
        text_opts.setdefault('verticalalignment', 'bottom')
        text_opts.setdefault('clip_on', True)

    if err_kws is None:
        err_kws = {}

    # if only one is specified allow it to be either
    if (hue is not None) and (color is None):
        color, color_order, colors = hue, hue_order, hues
        hue = hue_order = hues = None

    # default style options
    base_style = {
        'alpha': alpha,
        'markersize': 6,
        'color': '#0ca0eb',
        'marker': '.',
        'markeredgecolor': 'white',
    }
    # the size of each mapped dimension
    sizes = {}
    # the domain (i.e. input) of each mapped dimension
    domains = {}
    # the range (i.e. output) of each mappend dimension
    values = {}
    # how to label each mapped dimension
    labels = {
        x: x if xlabel is None else xlabel,
        y: y if ylabel is None else ylabel,
    }

    # work out all the dim mapping information

    def default_colormaps(N):
        if N <= 0:
            hs = _COLORS_DEFAULT[:N]
        else:
            hs = np.linspace(
                autohue_start, autohue_start + autohue_sweep, N, endpoint=False
            )
        return [to_colormap(h, **autohue_opts) for h in hs]

    # drop irrelevant variables and dimensions
    ds = ds.drop_vars([k for k in ds if k not in (x, y)])
    possible_dims = set()
    if x in ds.data_vars:
        possible_dims.update(ds[x].dims)
    if y in ds.data_vars:
        possible_dims.update(ds[y].dims)
    ds = ds.drop_dims([k for k in ds.dims if k not in possible_dims])

    mapped = set()

    ds, hue = init_mapped_dim(
        sizes, domains, values, labels, mapped, base_style, ds,
        "hue", hue, hue_order, hue_label,
        custom_values=(
            [to_colormap(h, **autohue_opts) for h in hues]
            if hues is not None else None
        ),
        default_values=default_colormaps,
    )
    ds, color = init_mapped_dim(
        sizes, domains, values, labels, mapped, base_style, ds,
        "color", color, color_order, color_label,
        custom_values=colors,
        default_values=lambda N: np.linspace(colormap_start, colormap_stop, N)
    )

    if (hue is not None) and (color is not None):
        # need special label
        labels[f"{hue}, {color}"] = f"{labels[hue]}, {labels[color]}"

    if (hue is None) and (color is not None):
        # set a global colormap or sequence
        if colors is None:
            cmap_or_colors = (
                to_colormap(palette) if palette is not None else
                auto_colors(sizes["color"])
            )
        else:
            cmap_or_colors = values["color"]

    ds, marker = init_mapped_dim(
        sizes, domains, values, labels, mapped, base_style, ds,
        "marker", marker, marker_order, marker_label,
        custom_values=markers,
        default_values=itertools.cycle(_MARKERS_DEFAULT)
    )
    ds, markersize = init_mapped_dim(
        sizes, domains, values, labels, mapped, base_style, ds,
        "markersize", markersize, markersize_order, markersize_label,
        custom_values=markersizes,
        default_values=lambda N: np.linspace(3.0, 9.0, N)
    )
    ds, markeredgecolor = init_mapped_dim(
        sizes, domains, values, labels, mapped, base_style, ds,
        "markeredgecolor", markeredgecolor,
        markeredgecolor_order, markeredgecolor_label,
        custom_values=markeredgecolors,
        default_values=lambda N: auto_colors(N)
    )
    ds, linestyle = init_mapped_dim(
        sizes, domains, values, labels, mapped, base_style, ds,
        "linestyle", linestyle, linestyle_order, linestyle_label,
        custom_values=linestyles,
        default_values=itertools.cycle(_LINESTYLES_DEFAULT)
    )
    ds, linewidth = init_mapped_dim(
        sizes, domains, values, labels, mapped, base_style, ds,
        "linewidth", linewidth, linewidth_order, linewidth_label,
        custom_values=linewidths,
        default_values=lambda N: np.linspace(1.0, 3.0, N)
    )
    ds, col = init_mapped_dim(
        sizes, domains, values, labels, mapped, base_style, ds,
        "col", col, col_order, col_label,
    )
    ds, row = init_mapped_dim(
        sizes, domains, values, labels, mapped, base_style, ds,
        "row", row, row_order, row_label,
    )

    # compute which dimensions are not target or mapped dimensions
    unmapped = sorted(set(ds.dims) - mapped - {x, y, xlink})

    is_histogram = y is None
    if is_histogram:
        # assume we want a histogram: create y as probability density / counts
        import xarray as xr

        # bin over all unmapped dimensions
        ds = ds.stack({'__hist_dim__': unmapped})

        # work out the bin coordinates
        if bins is None or isinstance(bins, int):
            if bins is None:
                nbins = min(max(3, int(ds['__hist_dim__'].size ** 0.5)), 50)
            else:
                nbins = bins
            xmin, xmax = ds[x].min(), ds[x].max()
            bins = np.linspace(xmin, xmax, nbins + 1)
        elif not isinstance(bins, np.ndarray):
            bins = np.asarray(bins)

        bin_coords = (bins[1:] + bins[:-1]) / 2

        if bins_density:
            y = f'prob({x})'
        else:
            y = f'count({x})'

        if ylabel is None:
            labels[y] = y
        else:
            labels[y] = ylabel

        ds = (
            xr.apply_ufunc(
                lambda x: np.histogram(x, bins=bins, density=bins_density)[0],
                ds[x],
                input_core_dims=[['__hist_dim__']],
                output_core_dims=[[x]],
                vectorize=True,
            )
            .to_dataset(name=y)
            .assign_coords({x: bin_coords})
        )
        kwargs.setdefault("drawstyle", "steps-mid")

    # get the target data array and possibly aggregate some dimensions
    if aggregate:
        if aggregate is True:
            # select all unmapped dimensions
            aggregate = unmapped

        # compute data ranges to maybe show spread bars or bands
        if aggregate_err_range == "std":

            da_std_mean = ds[y].mean(aggregate)
            da_std = ds[y].std(aggregate)

            da_ql = da_std_mean - da_std
            da_qu = da_std_mean + da_std

        elif aggregate_err_range == "stderr":

            da_stderr_mean = ds[y].mean(aggregate)
            da_stderr_cnt = ds[y].notnull().sum(aggregate)
            da_stderr = ds[y].std(aggregate) / np.sqrt(da_stderr_cnt)

            da_ql = da_stderr_mean - da_stderr
            da_qu = da_stderr_mean + da_stderr

        else:
            aggregate_err_range = min(max(0.0, aggregate_err_range), 1.0)
            ql = 0.5 - aggregate_err_range / 2.0
            qu = 0.5 + aggregate_err_range / 2.0
            da_ql = ds[y].quantile(ql, aggregate)
            da_qu = ds[y].quantile(qu, aggregate)

        # default to showing spread as bands
        if err is None:
            err = True
        if err_style is None:
            err_style = "band"

        # main data for central line
        ds = getattr(ds, aggregate_method)(aggregate)

    # default to bars if err not taken from aggregating
    if err_style is None:
        err_style = 'bars'

    # all the coordinates we will iterate over
    remaining_dims = []
    remaining_sizes = []
    for dim, sz in ds.sizes.items():
        if dim not in (x, xlink):
            remaining_dims.append(dim)
            remaining_sizes.append(sz)
    ranges = list(map(range, remaining_sizes))

    # maybe create the figure and axes
    if ax is not None:
        if axs is not None:
            raise ValueError("cannot specify both `ax` and `axs`")
        axs = np.array([[ax]])

    if axs is None:
        if figsize is None:
            if width is None:
                width = height
            if height is None:
                height = width
            figsize = (width * sizes["col"], height * sizes["row"])

        fig, axs = plt.subplots(
            sizes["row"], sizes["col"],
            sharex=sharex, sharey=sharey,
            squeeze=False,
            gridspec_kw={'hspace': hspace, 'wspace': wspace},
            figsize=figsize,
        )
        fig.patch.set_alpha(0.0)
    else:
        fig = None

    if (fig is not None) and (title is not None):
        fig.suptitle(title)

    # iterate over and plot all data
    handles = {}
    split_handles = collections.defaultdict(
        lambda: collections.defaultdict(dict)
    )

    x_is_constant = (x not in ds.data_vars)
    if x_is_constant:
        # is a constant coordinate
        xdata = ds[x].values

    for iloc in itertools.product(*ranges):
        # current coordinates
        loc = dict(zip(remaining_dims, iloc))

        # get the right set of axes to plot on
        if row is not None:
            i_ax = loc[row]
        else:
            i_ax = 0
        if col is not None:
            j_ax = loc[col]
        else:
            j_ax = 0
        ax = axs[i_ax, j_ax]

        # map coordinate into relevant styles and keep track of each uniquely
        sub_key = {}
        specific_style = {}

        # need to handle hue and color separately
        if color is not None:
            if hue is not None:
                ihue = loc[hue]
                hue_in = domains["hue"][ihue]
                sub_key[hue] = hue_in
                cmap_or_colors = values["hue"][ihue]

            icolor = loc[color]
            color_in = domains["color"][icolor]
            if not callable(cmap_or_colors):
                color_out = cmap_or_colors[icolor]
            else:
                color_out = cmap_or_colors(values["color"][icolor])

            sub_key[color] = color_in
            specific_style["color"] = color_out
            if hue is None:
                legend_dim = color
                legend_in = color_in
            else:
                legend_dim = ", ".join((hue, color))
                legend_in = ", ".join(map(str, (hue_in, color_in)))

            split_handles[legend_dim][legend_in]["color"] = color_out
        else:
            legend_dim = None

        for prop, dim in [
            ("marker", marker),
            ("markersize", markersize),
            ("markeredgecolor", markeredgecolor),
            ("linewidth", linewidth),
            ("linestyle", linestyle),
        ]:
            if dim is not None:
                idx = loc[dim]
                prop_in = domains[prop][idx]
                prop_out = values[prop][idx]
                sub_key[dim] = prop_in
                specific_style[prop] = prop_out

                if dim in (color, hue):
                    split_handles[legend_dim][legend_in][prop] = prop_out
                else:
                    split_handles[dim][prop_in][prop] = prop_out

        # get the masked x and y data
        ds_loc = ds.isel(loc)
        mdata = ds_loc[y].notnull().values

        if not x_is_constant:
            # x also varying
            xdata = ds_loc[x].values
            # both x and y must be non-null
            mdata &= ds_loc[x].notnull().values

        xmdata = xdata[mdata]
        if not np.any(xmdata):
            # don't plot all null lines
            continue
        ymdata = ds_loc[y].values[mdata]

        if (err is not None):

            if (err is True) and (aggregate is not None):
                da_ql_loc = da_ql.isel(loc)
                da_qu_loc = da_qu.isel(loc)
                y1 = da_ql_loc.values[mdata]
                y2 = da_qu_loc.values[mdata]
                yneg = ymdata - y1
                ypos = y2 - ymdata
            else:
                yerr_mdata = ds_loc[err].values[mdata]
                yneg = - yerr_mdata
                ypos = + yerr_mdata
                y1 = ymdata + yneg
                y2 = ymdata + ypos

            if err_style == 'bars':
                ax.errorbar(
                    x=xmdata, y=ymdata, yerr=[abs(yneg), abs(ypos)],
                    fmt='none',
                    capsize=err_bar_capsize,
                    **{**base_style, **specific_style, **err_kws},
                )
            elif err_style == 'band':
                ax.fill_between(
                    x=xmdata, y1=y1, y2=y2,
                    color=specific_style.get("color", base_style["color"]),
                    alpha=err_band_alpha,
                    **err_kws,
                )

        if is_histogram:
            ax.fill_between(
                x=xmdata, y1=ymdata, y2=0,
                step={
                    None: None,
                    'default': None,
                    'steps': 'pre',
                    'steps-pre': 'pre',
                    'steps-mid': 'mid',
                    'steps-post': 'post',
                }[kwargs.get("drawstyle", None)],
                color=mpl.colors.to_rgb(
                    specific_style.get("color", base_style["color"])
                ),
                alpha=err_band_alpha,
            )

        plot_opts = {**base_style, **specific_style}

        # do the plotting!
        handle, = ax.plot(
            xmdata, ymdata,
            label=", ".join(map(str, sub_key.values())),
            **plot_opts, **kwargs,
        )

        # add a text label next to each point
        if text is not None:
            smdata = ds_loc[text].values[mdata]
            for txx, txy, txs in zip(xmdata, ymdata, smdata):

                specific_text_opts = {}
                if 'color' not in text_opts:
                    # default to line color
                    specific_text_opts['color'] = plot_opts['color']

                ax.text(
                    txx, txy, text_formatter(txs),
                    **text_opts, **specific_text_opts,
                )

        # only want one legend entry per unique style
        key = frozenset(sub_key.items())
        handles.setdefault(key, handle)

    # perform axes level formatting

    for (i, j), ax in np.ndenumerate(axs):

        if fig is not None:
            # only change this stuff if we created the figure
            title = []
            if col is not None:
                title.append(f"{labels[col]}={domains['col'][j]}")
            if row is not None:
                title.append(f"{labels[row]}={domains['row'][i]}")
            if title:
                title = ", ".join(title)
                ax.text(0.5, 1.0, title, transform=ax.transAxes,
                        horizontalalignment='center', verticalalignment='bottom')

            # only label outermost plot axes
            if i + 1 == sizes["row"]:
                ax.set_xlabel(labels[x])
            if j == 0:
                ax.set_ylabel(labels[y])

            # set some nice defaults
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)

            if grid:
                ax.grid(True, which=grid_which, alpha=grid_alpha)
                ax.set_axisbelow(True)

            if xlim is not None:
                ax.set_xlim(xlim)
            if ylim is not None:
                ax.set_ylim(ylim)

            if xscale is not None:
                ax.set_xscale(xscale)
            if yscale is not None:
                ax.set_yscale(yscale)

            for scale, base, axis in [
                (xscale, xbase, ax.xaxis),
                (yscale, ybase, ax.yaxis),
            ]:
                if scale == 'log':
                    axis.set_major_locator(LogLocator(base=base, numticks=6))
                    if base != 10:
                        if isinstance(base, int):
                            axis.set_major_formatter(StrMethodFormatter("{x:.0f}"))
                        else:
                            axis.set_major_formatter(ScalarFormatter())
                    if base < 3:
                        subs = [1.5]
                    else:
                        subs = np.arange(2, base)
                    axis.set_minor_locator(LogLocator(base=base, subs=subs))
                    axis.set_minor_formatter(NullFormatter())
                elif scale == 'symlog':
                    # TODO: choose some nice defaults
                    pass
                else:
                    axis.set_minor_locator(AutoMinorLocator(5))

        for hline in hspans:
            ax.axhline(hline, color=span_color, alpha=span_alpha,
                       linestyle=span_linestyle, linewidth=span_linewidth)
        for vline in vspans:
            ax.axvline(vline, color=span_color, alpha=span_alpha,
                       linestyle=span_linestyle, linewidth=span_linewidth)

    # create a legend
    if legend:
        try:
            # try to extend current legend with more entries
            legend_handles = axs[0, -1].get_legend().get_lines()
            legend_handles.append(
                Line2D([0], [0], markersize=0, linewidth=0, label='')
            )
        except AttributeError:
            legend_handles = []

        legend_opts = {} if legend_opts is None else legend_opts

        if handles and legend_merge:
            # show every unique style combination as single legend try

            if legend_entries:
                # only keep manually specified legend entries
                remove = set()
                for k in handles:
                    for dim, val in k:
                        if dim in legend_entries:
                            if val not in legend_entries[dim]:
                                remove.add(k)
                for k in remove:
                    del handles[k]

            sorters = []
            legend_title = []
            for dim, dim_order in [
                (hue, hue_order),
                (color, color_order),
                (marker, marker_order),
                (markersize, markersize_order),
                (markeredgecolor, markeredgecolor_order),
                (linewidth, linewidth_order),
                (linestyle, linestyle_order),
            ]:
                if dim is not None and labels[dim] not in legend_title:
                    # check if not in legend_title, as multiple attributes can
                    # be mapped to the same dimension
                    legend_title.append(labels[dim])

                if dim is not None and dim_order is not None:
                    sorters.append((dim, dim_order.index))
                else:
                    sorters.append((dim, lambda x: x))

            def legend_sort(key_handle):
                loc = dict(key_handle[0])
                return tuple(
                    sorter(loc.get(dim, None)) for dim, sorter in sorters
                )

            legend_handles.extend(
                v for _, v in
                sorted(
                    handles.items(), key=legend_sort, reverse=legend_reverse
                )
            )

            if legend_ncol is None:
                if sizes["color"] == 1 or len(handles) <= 10:
                    legend_ncol = 1
                else:
                    legend_ncol = sizes["hue"]

            legend_opts.setdefault('title', ', '.join(legend_title))
            legend_opts.setdefault('ncol', legend_ncol)

        elif split_handles:
            # separate legend for each style

            if legend_entries:
                # only keep manually specified legend entries
                for k, vals in legend_entries.items():
                    split_handles[k] = {
                        key: val for key, val in split_handles[k].items()
                        if key in vals
                    }

            base_style["color"] = (0.5, 0.5, 0.5)
            base_style["marker"] = ''
            base_style["linestyle"] = ''

            ncol = len(split_handles)
            nrow = max(map(len, split_handles.values()))

            for legend_dim, inputs in split_handles.items():
                legend_handles.append(
                    Line2D(
                        [0], [0],
                        markersize=0,
                        linewidth=0,
                        label=labels[legend_dim]
                    )
                )
                for key, style in sorted(
                    inputs.items(),
                    key=lambda x: x[0],
                    # key=lambda x: 1,
                    reverse=legend_reverse,
                ):

                    if any("marker" in prop for prop in style):
                        style.setdefault('marker', 'o')
                    if any("line" in prop for prop in style):
                        style.setdefault('linestyle', '-')
                    if 'color' in style:
                        style.setdefault('marker', '.')
                        style.setdefault('linestyle', '-')

                    legend_handles.append(
                        Line2D([0], [0], **{**base_style, **style}, label=str(key))
                    )

                if legend_ncol is None:
                    npad = nrow - len(inputs)
                else:
                    npad = 1
                for _ in range(npad):
                    legend_handles.append(
                        Line2D([0], [0], markersize=0, linewidth=0, label='')
                    )

            if legend_ncol is None:
                legend_opts.setdefault('ncol', ncol)

        else:
            legend_handles = None

        if legend_handles is not None:
            lax = axs[0, -1]
            legend_opts.setdefault('loc', 'upper left')
            legend_opts.setdefault('bbox_to_anchor', (1.0, 1.0))
            legend_opts.setdefault('columnspacing', 1.0)
            legend_opts.setdefault('edgecolor', 'none')
            legend_opts.setdefault('framealpha', 0.0)
            lax.legend(handles=legend_handles, **legend_opts)

    return fig, axs
