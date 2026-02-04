"""
Functions for plotting datasets nicely.
"""
# TODO: custom xtick labels                                                   #
# TODO: annotations, arbitrary text                                           #
# TODO: docs                                                                  #

import functools

import numpy as np

from ..manage import auto_xyz_ds
from ..utils import autocorrect_kwargs
from .color import xyz_colormaps
from .core import (
    PLOTTER_DEFAULTS,
    AbstractHeatMap,
    AbstractHistogram,
    AbstractLinePlot,
    AbstractScatter,
    Plotter,
    calc_row_col_datasets,
    intercept_call_arg,
    prettify,
)

# ----------------- Main lineplot interface for matplotlib ------------------ #


class PlotterMatplotlib(Plotter):
    """ """

    def __init__(self, ds, x, y, z=None, y_err=None, x_err=None, **kwargs):
        super().__init__(
            ds,
            x,
            y,
            z=z,
            y_err=y_err,
            x_err=x_err,
            **kwargs,
            backend="MATPLOTLIB",
        )

    def prepare_axes(self):
        """ """
        import matplotlib as mpl

        if self.math_serif:
            mpl.rcParams["mathtext.fontset"] = "cm"
            mpl.rcParams["mathtext.rm"] = "serif"
        mpl.rcParams["font.family"] = self.font
        mpl.rcParams["font.weight"] = self.font_weight
        import matplotlib.pyplot as plt

        if self.axes_rloc is not None:
            if self.axes_loc is not None:
                raise ValueError(
                    "Cannot specify absolute and relative "
                    "location of axes at the same time."
                )
            if self.add_to_fig is None:
                raise ValueError(
                    "Can only specify relative axes position "
                    "when adding to a figure, i.e. when "
                    "add_to_fig != None"
                )

        if self.axes_rloc is not None:
            self._axes_loc = self._cax_rel2abs_rect(
                self.axes_rloc, self.add_to_fig.get_axes()[-1]
            )
        else:
            self._axes_loc = self.axes_loc

        # Add a new set of axes to an existing plot
        if self.add_to_fig is not None and self.subplot is None:
            self._fig = self.add_to_fig
            self._axes = self._fig.add_axes(
                (0.4, 0.6, 0.30, 0.25)
                if self._axes_loc is None
                else self._axes_loc
            )

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
                self._fig = plt.figure(
                    self.fignum, figsize=self.figsize, dpi=100
                )
            self._axes = self._fig.add_subplot(self.subplot)

        # Make new figure and axes
        else:
            self._fig = plt.figure(self.fignum, figsize=self.figsize, dpi=100)
            self._axes = self._fig.add_axes(
                (0.15, 0.15, 0.8, 0.75)
                if self._axes_loc is None
                else self._axes_loc
            )
        self._axes.set_title(
            "" if self.title is None else self.title,
            fontsize=self.fontsize_title,
            pad=self.title_pad,
        )

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
        """ """
        self._axes.set_xscale("log" if self.xlog else "linear")
        self._axes.set_yscale("log" if self.ylog else "linear")

    def set_axes_range(self):
        """ """
        if self._xlims:
            self._axes.set_xlim(self._xlims)
        if self._ylims:
            self._axes.set_ylim(self._ylims)

    def set_spans(self):
        """ """
        if self.vlines is not None:
            for x in self.vlines:
                self._axes.axvline(
                    x,
                    lw=self.span_width,
                    color=self.span_color,
                    linestyle=self.span_style,
                )
        if self.hlines is not None:
            for y in self.hlines:
                self._axes.axhline(
                    y,
                    lw=self.span_width,
                    color=self.span_color,
                    linestyle=self.span_style,
                )

    def set_gridlines(self):
        """ """
        for axis in ("top", "bottom", "left", "right"):
            self._axes.spines[axis].set_linewidth(1.0)

        if self.gridlines:
            # matplotlib has coarser gridine style than bokeh
            self._gridline_style = [x / 2 for x in self.gridline_style]
            self._axes.set_axisbelow(True)  # ensures gridlines below all
            self._axes.grid(
                True,
                color="0.9",
                which="major",
                linestyle=(0, self._gridline_style),
            )
            self._axes.grid(
                True,
                color="0.95",
                which="minor",
                linestyle=(0, self._gridline_style),
            )

    def set_tick_marks(self):
        """ """
        import matplotlib as mpl

        if self.xticks is not None:
            self._axes.set_xticks(self.xticks, minor=False)
            self._axes.get_xaxis().set_major_formatter(
                mpl.ticker.ScalarFormatter()
            )
        if self.yticks is not None:
            self._axes.set_yticks(self.yticks, minor=False)
            self._axes.get_yaxis().set_major_formatter(
                mpl.ticker.ScalarFormatter()
            )

        if self.xtick_labels is not None:
            self._axes.set_xticklabels(self.xtick_labels)

        if self.xticklabels_hide:
            (
                self._axes.get_xaxis().set_major_formatter(
                    mpl.ticker.NullFormatter()
                )
            )
        if self.yticklabels_hide:
            (
                self._axes.get_yaxis().set_major_formatter(
                    mpl.ticker.NullFormatter()
                )
            )

        self._axes.tick_params(
            labelsize=self.fontsize_ticks,
            direction="out",
            bottom="bottom" in self.ticks_where,
            top="top" in self.ticks_where,
            left="left" in self.ticks_where,
            right="right" in self.ticks_where,
        )

        if self.yticklabels_right or (
            self.yticklabels_right is None and self.ytitle_right is True
        ):
            self._axes.yaxis.tick_right()

    def _cax_rel2abs_rect(self, rel_rect, cax=None):
        """Turn a relative axes specification into a absolute one."""
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
        """Add a legend"""
        if self._use_legend:
            if labels_handles:
                labels, handles = zip(*labels_handles.items())
            else:
                handles, labels = self._legend_handles, self._legend_labels

            if self.legend_reverse:
                handles, labels = handles[::-1], labels[::-1]

            # Limit minimum size of markers that appear in legend
            should_auto_scale_legend_markers = (
                (self.legend_marker_scale is None)  # not already set
                and hasattr(self, "_marker_size")  # is a valid parameter
                and self._marker_size < 3  # and is small
            )
            if should_auto_scale_legend_markers:
                self.legend_marker_scale = 3 / self._marker_size

            opts = {
                "title": (self.z_coo if self.ztitle is None else self.ztitle),
                "loc": self.legend_loc,
                "fontsize": self.fontsize_zlabels,
                "frameon": self.legend_frame,
                "framealpha": self.legend_frame_alpha,
                "numpoints": 1,
                "scatterpoints": 1,
                "handlelength": self.legend_handlelength,
                "markerscale": self.legend_marker_scale,
                "labelspacing": self.legend_label_spacing,
                "columnspacing": self.legend_column_spacing,
                "bbox_to_anchor": self.legend_bbox,
                "ncol": self.legend_ncol,
            }

            if grid:
                bb = opts["bbox_to_anchor"]
                if bb is None:
                    opts["bbox_to_anchor"] = (1, 0.5, 0, 0)
                    opts["loc"] = "center left"
                else:
                    loc = opts["loc"]
                    # will get warning for 'best'
                    opts["loc"] = "center" if loc in ("best", 0) else loc
                lgnd = self._fig.legend(handles, labels, **opts)
            else:
                lgnd = self._axes.legend(handles, labels, **opts)

            lgnd.get_title().set_fontsize(self.fontsize_ztitle)

            if self.legend_marker_alpha is not None:
                for legendline in lgnd.legendHandles:
                    legendline.set_alpha(1.0)

    def set_mappable(self):
        """Mappale object for colorbars."""
        from matplotlib.cm import ScalarMappable

        self.mappable = ScalarMappable(cmap=self.cmap, norm=self._color_norm)
        self.mappable.set_array([])

    def plot_colorbar(self, grid=False):
        """Add a colorbar to the data."""

        if self._use_colorbar:
            # Whether the colorbar should clip at either end
            extendmin = (self.vmin is not None) and (self.vmin > self._zmin)
            extendmax = (self.vmax is not None) and (self.vmax < self._zmax)
            extend = (
                "both"
                if extendmin and extendmax
                else "min"
                if extendmin
                else "max"
                if extendmax
                else "neither"
            )

            opts = {"extend": extend, "ticks": self.zticks}

            if self.colorbar_relative_position:
                opts["cax"] = self._fig.add_axes(
                    self._cax_rel2abs_rect(self.colorbar_relative_position)
                )

            if grid:
                opts["ax"] = self._fig.axes
                opts["anchor"] = (0.5, 0.5)

            self._cbar = self._fig.colorbar(
                self.mappable, **opts, **self.colorbar_opts
            )

            self._cbar.ax.tick_params(labelsize=self.fontsize_zlabels)

            self._cbar.ax.set_title(
                self._ctitle,
                fontsize=self.fontsize_ztitle,
                color=self.colorbar_color if self.colorbar_color else None,
            )

            if self.colorbar_color:
                self._cbar.ax.yaxis.set_tick_params(
                    color=self.colorbar_color, labelcolor=self.colorbar_color
                )
                self._cbar.outline.set_edgecolor(self.colorbar_color)

    def set_panel_label(self):
        if self.panel_label is not None:
            self._axes.text(
                *self.panel_label_loc,
                self.panel_label,
                transform=self._axes.transAxes,
                fontsize=self.fontsize_panel_label,
                color=self.panel_label_color,
                ha="left",
                va="top",
            )

    def show(self):
        import matplotlib.pyplot as plt

        if self.return_fig:
            plt.close(self._fig)
            return self._fig

    def prepare_plot(self):
        """Do all the things that every plot has."""
        self.prepare_axes()
        self.set_axes_labels()
        self.set_axes_scale()
        self.set_axes_range()
        self.set_spans()
        self.set_gridlines()
        self.set_tick_marks()


# --------------------------------------------------------------------------- #


def mpl_multi_plot(fn):
    """Decorate a plotting function to plot a grid of values."""

    @functools.wraps(fn)
    def multi_plotter(
        ds,
        *args,
        row=None,
        col=None,
        hspace=None,
        wspace=None,
        tight_layout=True,
        coltitle=None,
        rowtitle=None,
        **kwargs,
    ):
        if (row is None) and (col is None):
            return fn(ds, *args, **kwargs)

        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        # Set some global parameters
        p = fn(ds, *args, **kwargs, call=False)
        p.prepare_data_multi_grid()

        kwargs["vmin"] = kwargs.pop("vmin", p.vmin)
        kwargs["vmax"] = kwargs.pop("vmax", p.vmax)

        coltitle = col if coltitle is None else coltitle
        rowtitle = row if rowtitle is None else rowtitle

        # split the dataset into its respective rows and columns
        ds_r_c, nrows, ncols = calc_row_col_datasets(ds, row=row, col=col)

        figsize = kwargs.pop("figsize", (3 * ncols, 3 * nrows))
        return_fig = kwargs.pop("return_fig", PLOTTER_DEFAULTS["return_fig"])

        # generate a figure for all the plots to use
        p._fig = plt.figure(
            figsize=figsize, dpi=100, constrained_layout=tight_layout
        )
        p._fig.set_constrained_layout_pads(hspace=hspace, wspace=wspace)
        # and a gridspec to position them
        gs = GridSpec(
            nrows=nrows,
            ncols=ncols,
            figure=p._fig,
            hspace=hspace,
            wspace=wspace,
        )

        # want to collect all entries for legend
        labels_handles = {}

        # range through rows and do subplots
        for i, ds_r in enumerate(ds_r_c):
            skws = {"legend": False, "colorbar": False}

            # if not last row
            if i != nrows - 1:
                skws["xticklabels_hide"] = True
                skws["xtitle"] = ""

            # range through columns
            for j, sub_ds in enumerate(ds_r):
                if hspace == 0 and wspace == 0:
                    ticks_where = []
                    if j == 0:
                        ticks_where.append("left")
                    if i == 0:
                        ticks_where.append("top")
                    if j == ncols - 1:
                        ticks_where.append("right")
                    if i == nrows - 1:
                        ticks_where.append("bottom")
                    skws["ticks_where"] = ticks_where

                # if not first column
                if j != 0:
                    skws["yticklabels_hide"] = True
                    skws["ytitle"] = ""

                # label each column
                if (i == 0) and (col is not None):
                    col_val = prettify(ds[col].values[j])
                    skws["title"] = "{} = {}".format(coltitle, col_val)
                    fx = "fontsize_xtitle"
                    skws["fontsize_title"] = kwargs.get(
                        fx, PLOTTER_DEFAULTS[fx]
                    )

                # label each row
                if (j == ncols - 1) and (row is not None):
                    # XXX: if number of cols==1 this hide yaxis - want both
                    row_val = prettify(ds[row].values[i])
                    skws["ytitle_right"] = True
                    skws["ytitle"] = "{} = {}".format(rowtitle, row_val)

                sP = fn(
                    sub_ds,
                    *args,
                    add_to_fig=p._fig,
                    call="both",
                    subplot=gs[i, j],
                    **{**kwargs, **skws},
                )

                try:
                    labels_handles.update(
                        dict(zip(sP._legend_labels, sP._legend_handles))
                    )
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
    """ """

    def __init__(self, ds, x, y, z=None, *, y_err=None, x_err=None, **kwargs):
        super().__init__(ds, x, y, z=z, y_err=y_err, x_err=x_err, **kwargs)

    def plot_lines(self):
        """ """
        for data in self._gen_xy():
            col = next(self._cols)

            line_opts = {
                "c": col,
                "lw": next(self._lws),
                "marker": next(self._mrkrs),
                "markersize": self._marker_size,
                "markeredgecolor": col[:3] + (self.marker_alpha * col[3],),
                "markerfacecolor": col[:3] + (self.marker_alpha * col[3] / 2,),
                "label": next(self._zlbls),
                "zorder": next(self._zordrs),
                "linestyle": next(self._lines),
                "rasterized": self.rasterize,
            }

            if ("ye" in data) or ("xe" in data):
                self._axes.errorbar(
                    data["x"],
                    data["y"],
                    yerr=data.get("ye", None),
                    xerr=data.get("xe", None),
                    ecolor=col,
                    capsize=self.errorbar_capsize,
                    capthick=self.errorbar_capthick,
                    elinewidth=self.errorbar_linewidth,
                    **line_opts,
                )
            else:
                # add line to axes, with options cycled through
                self._axes.plot(data["x"], data["y"], **line_opts)

            (
                self._legend_handles,
                self._legend_labels,
            ) = self._axes.get_legend_handles_labels()

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
        super().__init__(ds, "x", "y", z="z", **lineplot_opts)


def auto_lineplot(x, y_z, **lineplot_opts):
    """Auto version of :func:`~xyzpy.lineplot` that accepts array arguments
    by converting them to a ``Dataset`` first.
    """
    return AutoLinePlot(x, y_z, **lineplot_opts)()


# --------------------------------------------------------------------------- #

_SCATTER_ALT_DEFAULTS = (("legend_handlelength", 0),)


class Scatter(PlotterMatplotlib, AbstractScatter):
    def __init__(self, ds, x, y, z=None, **kwargs):
        # set some scatter specific options
        for k, default in _SCATTER_ALT_DEFAULTS:
            if k not in kwargs:
                kwargs[k] = default
        super().__init__(ds, x, y, z, **kwargs)

    def plot_scatter(self):
        """ """
        self._legend_handles = []
        self._legend_labels = []

        for data in self._gen_xy():
            if "c" in data:
                col = data["c"]
            else:
                col = [next(self._cols)]

            scatter_opts = {
                "c": col,
                "marker": next(self._mrkrs),
                "s": self._marker_size,
                "alpha": self.marker_alpha,
                "label": next(self._zlbls),
                "zorder": next(self._zordrs),
                "rasterized": self.rasterize,
            }

            if "c" in data:
                scatter_opts["cmap"] = self.cmap

            self._legend_handles.append(
                self._axes.scatter(data["x"], data["y"], **scatter_opts)
            )
            self._legend_labels.append(scatter_opts["label"])

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
        super().__init__(ds, "x", "y", z="z", **scatter_opts)


def auto_scatter(x, y_z, **scatter_opts):
    """Auto version of :func:`~xyzpy.scatter` that accepts array arguments
    by converting them to a ``Dataset`` first.
    """
    return AutoScatter(x, y_z, **scatter_opts)()


# --------------------------------------------------------------------------- #

_HISTOGRAM_SPECIFIC_OPTIONS = {
    "stacked": False,
}

_HISTOGRAM_ALT_DEFAULTS = {
    "xtitle": "x",
    "ytitle": "f(x)",
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
        from matplotlib.patches import Polygon, Rectangle

        def gen_ind_plots():
            for data in self._gen_xy():
                col = next(self._cols)

                edgecolor = col[:3] + (self.marker_alpha * col[3],)
                facecolor = col[:3] + (self.marker_alpha * col[3] / 4,)
                linewidth = next(self._lws)
                zorder = next(self._zordrs)
                label = next(self._zlbls)

                handle = Rectangle((0, 0), 1, 1, color=facecolor, ec=edgecolor)

                yield (
                    data["x"],
                    edgecolor,
                    facecolor,
                    linewidth,
                    zorder,
                    label,
                    handle,
                )

        xs, ecs, fcs, lws, zds, lbs, hnds = zip(*gen_ind_plots())

        histogram_opts = {
            "label": lbs,
            "bins": self.bins,
            "density": True,
            "histtype": "stepfilled",
            "fill": True,
            "stacked": self.stacked,
            "rasterized": self.rasterize,
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
        super().__init__(ds, "x", **histogram_opts)


def auto_histogram(x, **histogram_opts):
    """Auto version of :func:`~xyzpy.histogram` that accepts array arguments
    by converting them to a ``Dataset`` first.
    """
    return AutoHistogram(x, **histogram_opts)()


# --------------------------------------------------------------------------- #

_HEATMAP_ALT_DEFAULTS = (
    ("legend", False),
    ("colorbar", True),
    ("colormap", "inferno"),
    ("method", "pcolormesh"),
    ("gridlines", False),
    ("rasterize", True),
)


class HeatMap(PlotterMatplotlib, AbstractHeatMap):
    def __init__(self, ds, x, y, z, **kwargs):
        # set some heatmap specific options
        for k, default in _HEATMAP_ALT_DEFAULTS:
            if k not in kwargs:
                kwargs[k] = default
        super().__init__(ds, x, y, z, **kwargs)

    def plot_heatmap(self):
        """Plot the data as a heatmap."""
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
            X,
            Y,
            self._heatmap_var,
            norm=self._color_norm,
            cmap=xyz_colormaps(self.colormap),
            rasterized=self.rasterize,
        )

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
        super().__init__(ds, "y", "z", "x", **heatmap_opts)


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
        ax.set_aspect("equal")
        ax.axis("off")

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
                action="ignore",
                message="All-NaN slice",
            )
            fig, ax = fn(*args, **kwargs)

        if fig is not None:
            if show_and_close:
                plt.show()
                plt.close(fig)

        return fig, ax

    return wrapped


def choose_squarest_grid(x):
    p = x**0.5
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
    magscale="linear",
    max_mag=None,
    alpha_map=True,
    alpha_pow=1 / 2,
):
    """Convert complex array to RGBA colors.

    Parameters
    ----------
    zs : array_like
        Complex array to convert to colors.
    magscale : {"linear", "log", float}, optional
        Scale to apply to the magnitude before mapping to saturation.
        If "linear", no scaling is applied. If "log", logarithmic scaling
        is applied. If float, the magnitude is raised to this power.
        Default is "linear".
    max_mag : float, optional
        Maximum magnitude to map to full saturation. If None, the maximum
        magnitude in `zs` is used.
    alpha_map : bool, optional
        Whether to append an alpha channel based on magnitude. Default is True.
    alpha_pow : float, optional
        Power to raise the magnitude to when computing the alpha channel.
        Default is 1/2.

    Returns
    -------
    colors : ndarray
        Array of shape (..., 3) or (..., 4) of RGB(A) colors.
    mapped_mag : ndarray
        The mapped magnitudes used for saturation and alpha.
    max_mag : float
        The maximum magnitude used for saturation mapping.
    """
    import matplotlib as mpl

    arraymag = np.abs(zs)

    if magscale == "linear":
        mapped_mag = arraymag
    elif magscale == "log":
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
        zalpha = (mapped_mag / max_mag) ** alpha_pow
        zs = np.concatenate([zs, np.expand_dims(zalpha, -1)], axis=-1)

    return zs, mapped_mag, max_mag


def add_visualize_legend(
    ax,
    complexobj,
    max_mag,
    legend_inside=False,
    auto_pad=0.03,
    legend_loc="auto",
    legend_size=0.15,
    legend_bounds=None,
    legend_resolution=3,
):
    import matplotlib as mpl

    # choose where to put the legend
    if legend_bounds is None:
        if legend_loc in ("auto", "bottom right"):
            if not legend_inside:
                # move compass and legends beyond the plot rectangle, which
                # will be filled when there are only 2 plot dimensions
                legend_loc = (1 - auto_pad, 0.0 - legend_size + auto_pad)
            else:
                # occupy space within the rectangle
                legend_loc = (1.0 - legend_size, 0.0)

        elif legend_loc == "top right":
            if not legend_inside:
                legend_loc = (1 - auto_pad, 1 - legend_size - auto_pad)
            else:
                legend_loc = (1.0 - legend_size, 1.0 - legend_size)

        legend_bounds = (*legend_loc, legend_size, legend_size)

    lax = ax.inset_axes(legend_bounds)
    lax.axis("off")
    lax.set_aspect("equal")

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
    lax.text(
        1.02,
        0.5,
        "$+1$",
        ha="left",
        va="center",
        transform=lax.transAxes,
        color=(0.5, 0.5, 0.5),
        size=6,
    )
    lax.text(
        -0.02,
        0.5,
        "$-1$",
        ha="right",
        va="center",
        transform=lax.transAxes,
        color=(0.5, 0.5, 0.5),
        size=6,
    )
    if complexobj:
        lax.text(
            0.5,
            -0.02,
            "$-j$",
            ha="center",
            va="top",
            transform=lax.transAxes,
            color=(0.5, 0.5, 0.5),
            size=6,
        )
        lax.text(
            0.5,
            1.02,
            "$+j$",
            ha="center",
            va="bottom",
            transform=lax.transAxes,
            color=(0.5, 0.5, 0.5),
            size=6,
        )

    # show the overall scale
    if complexobj:
        overall_scale_opts = {"x": 0.85, "y": 0.15, "ha": "left"}
    else:
        overall_scale_opts = {"x": 0.5, "y": -0.15, "ha": "center"}

    lax.text(
        s=f"$\\times {float(max_mag):.3}$",
        va="top",
        **overall_scale_opts,
        transform=lax.transAxes,
        color=(0.5, 0.5, 0.5),
        size=6,
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

        if isinstance(array, (tuple, list)) and all(
            hasattr(x, "shape") for x in array
        ):
            # assume sequence of tensors to plot
            figsize = kwargs.get("figsize", (5, 5))
            rasterize_dpi = kwargs.get("rasterize_dpi", 300)
            nplot = len(array)
            fig, axs = plt.subplots(
                1,
                nplot,
                figsize=figsize,
                squeeze=False,
                sharey=True,
            )
            fig.set_dpi(rasterize_dpi)
            fig.patch.set_alpha(0.0)

            # plot all tensors with same magnitude scale
            kwargs.setdefault("max_mag", max(np.max(np.abs(x)) for x in array))

            # only show legend on last plot
            legend = kwargs.pop("legend", True)

            for i in range(nplot):
                f(
                    array[i],
                    *args,
                    ax=axs[0, i],
                    # only show legend on last plot
                    legend=(legend and i == nplot - 1),
                    show_and_close=False,
                    **kwargs,
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
    magscale="linear",
    alpha_map=True,
    alpha_pow=1 / 2,
    legend=True,
    legend_loc="auto",
    legend_size=0.15,
    legend_bounds=None,
    legend_resolution=3,
    facecolor=None,
    rasterize=4096,
    rasterize_dpi=300,
    figsize=(5, 5),
    ax=None,
):
    """Visualize ``array`` as a 2D colormapped image.

    Parameters
    ----------
    array : array_like or Sequence[array_like]
        A 2D (or 1D) array or sequence of arrays to visualize.
    max_mag : float, optional
        The maximum magnitude to use for the color mapping. If not provided,
        the maximum magnitude in the array will be used.
    magscale : "linear" or float, optional
        How to scale the magnitude of the array values. If "linear", then
        the magnitude is used directly. If a float, then the magnitude is
        raised to this power before being used, which can help to show
        variation among small values.
    alpha_map : bool, optional
        Whether to map the tensor value magnitudes to pixel alpha.
    alpha_pow : float, optional
        The power to raise the magnitude to when mapping to alpha.
    legend : bool, optional
        Whether to show a legend (colorbar). If the array has complex dtype
        then the legend will be a colorwheel.
    legend_loc : str or tuple[float], optional
        Where to place the legend. If "auto", then the legend will be placed
        outside the plot rectangle, otherwise it should be a tuple of
        ``(x, y)`` coordinates in axes space.
    legend_size : float, optional
        The size of the legend, in relation to the size of the plot axes.
    legend_bounds : tuple[float], optional
        The bounds of the legend, as ``(x, y, width, height)`` in axes space.
        If not provided, the bounds will be computed from ``legend_loc`` and
        ``legend_size``.
    legend_resolution : int, optional
        The number of different colors to show in the legend.
    facecolor : str, optional
        The background color of the plot, by default transparent.
    rasterize : int or float, optional
        Whether to rasterize the plot. If a number, then rasterize if the
        number of pixels in the plot is greater than this value.
    rasterize_dpi : float, optional
        The dpi to use when rasterizing.
    figsize : tuple[float], optional
        The size of the figure to create, if ``ax`` is not provided.
    ax : matplotlib.Axis, optional
        The axis to draw to. If not provided, a new figure will be created.
    show_and_close : bool, optional
        If ``True`` (the default) then show and close the figure, otherwise
        return the figure and axis.

    Returns
    -------
    fig : matplotlib.Figure
        The figure containing the plot, or ``None`` if ``ax`` was provided.
    ax : matplotlib.Axis
        The axis or axes containing the plot(s).
    """
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
    ax.imshow(zs, interpolation="nearest", zorder=-1)

    if legend:
        add_visualize_legend(
            ax=ax,
            complexobj=np.iscomplexobj(array),
            max_mag=max_mag,
            legend_inside=False,
            legend_loc=legend_loc,
            legend_size=legend_size,
            legend_bounds=legend_bounds,
            legend_resolution=legend_resolution,
        )

    return fig, ax


@handle_sequence_of_arrays
@show_and_close
@autocorrect_kwargs
def visualize_tensor(
    array,
    spacing_factor=1.0,
    max_projections=None,
    projection_overlap_spacing=1.05,
    angles=None,
    scales=None,
    skew_angle_factor="auto",
    skew_scale_factor=0.05,
    max_mag=None,
    magscale="linear",
    size_map=True,
    size_pow=1 / 2,
    size_scale=1.0,
    alpha_map=True,
    alpha_pow=1 / 2,
    alpha=0.8,
    marker="o",
    linewidths=0,
    show_lattice=True,
    lattice_opts=None,
    compass=False,
    compass_loc="auto",
    compass_size=0.1,
    compass_bounds=None,
    compass_labels=None,
    compass_opts=None,
    legend=True,
    legend_loc="auto",
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
    spacing_factor : float, optional
        How to scale the dimensions relative to each other. If 1.0, then each
        dimension will have the same extent, and smaller dimensions will be
        sparser. If 0.0, the each dimension will have an extent propoertional
        to its size, with matching density.
    max_projections : int, optional
        The maximum number of different projection directions / angles to use.
        If specified and less than the number of dimensions, then multiple
        dimensions will share the same angle but with different scales.
    projection_overlap_spacing : float, optional
        When grouping multiple dimensions to the same angle, how much to
        increase the spacing at each scale so as to emphasize each.
    angles : sequence[float], optional
        An explicit list of angles to use for each direction, in radians, with
        zero pointing straight down. If not provided, then the angles will be
        calculated automatically.
    scales : sequence[float], optional
        An explicit list of scales to use for each direction. If not provided,
        then the scales will be calculated automatically.
    skew_angle_factor : float, optional
        When there are more than two dimensions, a factor to scale the
        rotations by to avoid overlapping data points. If 0.0 then the angles
        will be evenly spaced.
    skew_scale_factor : float, optional
        When there are more than two dimensions, a factor to scale the
        scales by to avoid overlapping data points, that shortens
        non-perpendicular directions.
    max_mag : float, optional
        The maximum magnitude to use for the color mapping. If not provided,
        the maximum magnitude in the array will be used.
    magscale : "linear" or float, optional
        How to scale the magnitude of the array values. If "linear", then
        the magnitude is used directly. If a float, then the magnitude is
        raised to this power before being used, which can help to show
        variation among small values.
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
    marker : str, optional
        The marker to use for the markers.
    linewidths : float, optional
        The linewidth to use for the markers.
    show_lattice : bool, optional
        Show a thin grey line connecting adjacent array coordinate points.
    lattice_opts : dict, optional
        Options to pass to ``maplotlib.Axis.scatter`` for the lattice grid.
    compass : bool, optional
        Whether to show a compass indicating the orientation of each dimension.
    compass_loc : (float, float), optional
        Where to place the compass.
    compass_size : float, optional
        The size of the compass.
    compass_bounds : tuple[float], optional
        Explicit bounds of the compass, as ``(x, y, width, height)``.
    compass_labels : sequence[str], optional
        Explicit labels for the compass, in order of the dimensions.
    compass_opts : dict, optional
        Extra options for the compass arrows.
    legend : bool, optional
        Whether to show a legend (colorbar). If the array has complex dtype
        then the legend will be a colorwheel.
    legend_loc : str or tuple[float], optional
        Where to place the legend. If "auto", then the legend will be placed
        outside the plot rectangle, otherwise it should be a tuple of
        ``(x, y)`` coordinates in axes space.
    legend_size : float, optional
        The size of the legend, in relation to the size of the plot axes.
    legend_bounds : tuple[float], optional
        Explicit bounds of the legend, as ``(x, y, width, height)`` in axes
        space.
    legend_resolution : int, optional
        The number of different colors to show in the legend.
    interleave_projections : bool, optional
        If ``True`` and grouping dimensions, then they are assigned round robin
        fashion rather than blocks. ``False`` matches the behavior of fusing.
    reverse_projections : bool, optional
        Whether to reverse the order of the projections.
    facecolor : str, optional
        The background color of the plot, by default transparent.
    rasterize : int or float, optional
        Whether to rasterize the plot. If a number, then rasterize if the
        size of the array is greater than this value.
    rasterize_dpi : float, optional
        The dpi to use when rasterizing.
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

    if max_projections is None:
        max_projections = array.ndim

    auto_skew = skew_angle_factor == "auto"
    if auto_skew:
        if max_projections == 3:
            skew_angle_factor = 0.3
        else:
            skew_angle_factor = 0.05

    auto_angles = angles is None
    if scales == "equal":
        scales = [1] * array.ndim

    auto_scales = scales is None

    # map each dimension to an angle
    if not auto_angles:
        angles = np.array(angles)
    else:
        # if max_projections == array.ndim, then each dimension has its own
        # angle, if max_projections < array.dim, then we will
        # reuse the same angles, initially round robin distributed
        angles = np.tile(
            np.linspace(0.0, np.pi, max_projections, endpoint=False),
            array.ndim // max_projections + 1,
        )[: array.ndim]

        if not interleave_projections:
            # 'fill up' one angle before moving on, rather than round-robin,
            # doing this matches the behavior of fusing adjacent dimensions
            angles = np.sort(angles)

        def angle_modulate(x):
            return x * (x - np.pi / 2) * (x - x[-1])

        # modulate the angles slightly to avoid overlapping data points
        angles += angle_modulate(angles) * skew_angle_factor

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
                # base size
                (grouped_size[phi] // array.shape[i])
                # shrink non-perpendicular dimensions
                * (1 + skew_scale_factor * np.cos(4 * angles[i]))
                # put extra space between distinct dimensions
                * projection_overlap_spacing ** group_counter[phi]
                # how much to match the first size by
                * max(1, (first_size[phi] - 1)) ** -spacing_factor
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
        s = base_size * (mapped_mag / max_mag) ** size_pow
    else:
        s = base_size

    if show_lattice:
        # put a small grey line on every edge
        ls = []
        for i, isize in fastest_varying:
            other_shape = array.shape[:i] + array.shape[i + 1 :]
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
        lattice_opts.setdefault("color", (0.6, 0.6, 0.6))
        lattice_opts.setdefault(
            "alpha", 0.01 + 2 ** (-(array.size**0.2 + eff_ndim**0.8))
        )
        lattice_opts.setdefault("linewidth", 1)
        lattice_opts.setdefault("zorder", -2)
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
            if compass_loc == "auto":
                if max_projections <= 2:
                    # move compass and legends beyond the plot rectangle, which
                    # will be filled when there are only 2 plot dimensions
                    compass_loc = (-0.05, 1.0 - compass_size + 0.05)
                else:
                    # occupy space within the rectangle
                    compass_loc = (0.0, 1 - compass_size)
            compass_bounds = (*compass_loc, compass_size, compass_size)

        cax = ax.inset_axes(compass_bounds)
        cax.axis("off")
        cax.set_aspect("equal")

        compass_opts = {} if compass_opts is None else dict(compass_opts)
        compass_opts.setdefault("color", (0.5, 0.5, 0.5))
        compass_opts.setdefault("width", 0.002)
        compass_opts.setdefault("length_includes_head", True)

        if compass_labels is None:
            compass_labels = range(len(angles))
        elif compass_labels is False:
            compass_labels = [""] * len(angles)

        for i, phi in enumerate(angles):
            modifier = 1 + skew_scale_factor * np.cos(4 * angles[i])
            dx = np.sin(phi) * group_rank[i] * modifier
            dy = -np.cos(phi) * group_rank[i] * modifier
            cax.arrow(0, 0, dx, dy, **compass_opts)
            cax.text(
                dx,
                dy,
                f" {compass_labels[i]}",
                ha="left",
                va="top",
                color=compass_opts["color"],
                size=6,
                rotation=180 * phi / np.pi - 90,
                rotation_mode="anchor",
            )

    if legend:
        add_visualize_legend(
            ax=ax,
            complexobj=np.iscomplexobj(array),
            max_mag=max_mag,
            legend_inside=max_projections > 2,
            legend_loc=legend_loc,
            legend_size=legend_size,
            legend_bounds=legend_bounds,
            legend_resolution=legend_resolution,
        )

    return fig, ax


def _draw_tensor_tree_branch(
    lines,
    widths,
    colors,
    array,
    x=0.0,
    y=0.0,
    phi_center=np.pi / 2,
    phi_range=np.pi,
    level=1,
    z=None,
    mag=None,
    mag_weight=None,
    linewidth=10,
):
    if not array.shape:
        # stop drawing
        return

    d, *subshape = array.shape
    phis = np.linspace(-0.5, 0.5, 2 * d + 1)[1:-1:2]

    for i in range(d):
        subarray = array[i]
        subphi = phi_center + phi_range * phis[i]

        subx = -level * np.cos(subphi)
        suby = level * np.sin(subphi)

        subz = z[i]
        submag = mag[i]

        _draw_tensor_tree_branch(
            lines,
            widths,
            colors,
            subarray,
            x=subx,
            y=suby,
            phi_center=subphi,
            phi_range=(phi_range / d),
            level=level + 1,
            z=subz,
            mag=submag,
            mag_weight=mag_weight,
            linewidth=linewidth,
        )

        width = linewidth * (np.sum(submag) / mag_weight) ** 0.5

        if not subshape:
            color = subz
        else:
            color = np.mean(subz, axis=tuple(range(subz.ndim - 1)))

        lines.append([(x, y), (subx, suby)])
        widths.append(width)
        colors.append(color)


def visualize_tensor_tree(
    array,
    phi_center=np.pi / 2,
    phi_range=np.pi,
    linewidth=10,
    legend=True,
    legend_loc="top right",
    legend_size=0.15,
    legend_bounds=None,
    legend_resolution=3,
    facecolor=None,
    rasterize=4096,
    rasterize_dpi=300,
    figsize=(5, 5),
    ax=None,
):
    from matplotlib.collections import LineCollection

    z, mag, max_mag = to_colors(
        array,
        magscale=0.5,
        alpha_map=False,
    )
    mag_weight = np.sum(mag)

    lines = []
    widths = []
    colors = []

    _draw_tensor_tree_branch(
        lines=lines,
        widths=widths,
        colors=colors,
        array=array,
        z=z,
        mag=mag,
        mag_weight=mag_weight,
        phi_center=phi_center,
        phi_range=phi_range,
        linewidth=linewidth,
    )

    fig, ax = setup_fig_ax(
        facecolor=facecolor,
        rasterize=rasterize,
        rasterize_dpi=rasterize_dpi,
        figsize=figsize,
        ax=ax,
    )

    lc = LineCollection(
        lines,
        linewidths=widths,
        colors=colors,
        zorder=-1,
        capstyle="round",
    )

    ax.add_collection(lc)
    ax.autoscale()

    if legend:
        add_visualize_legend(
            ax=ax,
            complexobj=np.iscomplexobj(array),
            max_mag=max_mag,
            legend_inside=True,
            legend_loc=legend_loc,
            legend_size=legend_size,
            legend_bounds=legend_bounds,
            legend_resolution=legend_resolution,
        )

    return fig, ax
