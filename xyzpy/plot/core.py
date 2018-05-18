"""
Helper functions for preparing data to be plotted.
"""
# TODO: x-err, upper and lower errors *************************************** #
# TODO: check shape and hint at which dimensions need to be reduced ********* #

import itertools
import functools
import numpy as np
import numpy.ma as ma
import xarray as xr
from .color import convert_colors, xyz_colormaps, get_default_sequential_cm
from .marker import _MARKERS, _SINGLE_LINE_MARKER

PLOTTER_DEFAULTS = {
    # Figure options
    'backend': 'MATPLOTLIB',
    'figsize': (7, 6),
    'axes_loc': None,
    'axes_rloc': None,
    'add_to_axes': None,
    'add_to_xaxes': None,
    'add_to_fig': None,
    'subplot': None,
    'fignum': 1,
    'return_fig': True,
    # Coloring options
    'colors': None,
    'colormap': None,
    'colormap_log': False,
    'colormap_reverse': False,
    # Colorbar options
    'colorbar': None,
    'ctitle': None,
    'vmin': None,
    'vmax': None,
    'colorbar_relative_position': None,
    'colorbar_where': None,
    'colorbar_opts': dict(),
    'method': None,
    'colorbar_color': "black",
    # 'z-axis' options: for legend or colorbar
    'ztitle': None,
    'zlabels': None,
    'zlims': (None, None),
    'zticks': None,
    'legend': None,
    'legend_loc': 'best',
    'legend_where': None,
    'legend_ncol': 1,
    'legend_bbox': None,
    'legend_marker_scale': None,
    'legend_label_spacing': None,
    'legend_column_spacing': 1,
    'legend_frame': False,
    'legend_handlelength': None,
    'legend_reverse': False,
    'legend_marker_alpha': None,
    # x-axis options
    'xtitle': None,
    'xtitle_pad': 5,
    'xlims': None,
    'xticks': None,
    'xtick_labels': None,
    'xticklabels_hide': False,
    'xlog': False,
    'bins': 30,
    # y-axis options
    'ytitle': None,
    'ytitle_pad': 5,
    'ytitle_right': False,
    'ylims': None,
    'yticks': None,
    'yticklabels_hide': False,
    'yticklabels_right': None,
    'ylog': False,
    # Titles and text
    'title': None,
    'panel_label': None,
    'panel_label_loc': (0.05, 0.93),
    'panel_label_color': 'black',
    # Styling options
    'markers': None,
    'markersize': None,
    'marker_alpha': 1.0,
    'lines': True,
    'line_styles': None,
    'line_widths': None,
    'errorbar_capthick': 1,
    'errorbar_capsize': 0,
    'errorbar_linewidth': 0.7,
    'zorders': None,
    'padding': None,
    'vlines': None,
    'hlines': None,
    'span_style': '--',
    'span_width': 1,
    'span_color': "0.5",
    'gridlines': True,
    'gridline_style': (1, 2),
    'ticks_where': ('bottom', 'left', 'top', 'right'),
    # Font options
    'font': ('CMU Serif', 'PT Serif', 'Liberation Serif', 'DejaVu Serif'),
    'fontsize_title': 18,
    'fontsize_ticks': 13,
    'fontsize_xtitle': 18,
    'fontsize_ytitle': 18,
    'fontsize_ztitle': 18,
    'fontsize_zlabels': 13,
    'fontsize_panel_label': 15,
    'math_serif': True,
    # backend options
    'rasterize': False,
    'webgl': False,
}

_PLOTTER_OPTS = list(PLOTTER_DEFAULTS.keys())


def check_excess_dims(ds, var, valid_dims, mode='lineplot'):

    if mode == 'lineplot':
        expanded_valid_dims = list(itertools.chain.from_iterable(
            ds[d].dims for d in valid_dims))

        excess_dims = {d for d, sz in ds[var].sizes .items()
                       if (sz > 1) and (d not in expanded_valid_dims)}
        if excess_dims:
            raise ValueError("Dataset has too many non-singlet dimensions "
                             "- try selection values for the following: "
                             "{}.".format(excess_dims))

    else:
        # TODO: for now, no checking if not lineplot
        pass


class Plotter:
    """
    """

    def __init__(self, ds, x, y, z=None, c=None, y_err=None, x_err=None,
                 **kwargs):
        """
        """
        if isinstance(ds, xr.DataArray):
            self._ds = ds.to_dataset()
        else:
            self._ds = ds

        self.x_coo = x
        self.y_coo = y
        self.z_coo = z
        self.c_coo = c
        self.y_err = y_err
        self.x_err = x_err

        # Figure options
        settings = {**PLOTTER_DEFAULTS, **kwargs}
        for opt in _PLOTTER_OPTS:
            setattr(self, opt, settings.pop(opt))

        # Parse unmatched keywords and error but suggest correct versions
        if len(settings) > 0:
            import difflib
            wrong_opts = list(settings.keys())
            right_opts = [difflib.get_close_matches(opt, _PLOTTER_OPTS, n=3)
                          for opt in wrong_opts]

            msg = ("Option(s) {} not valid.\n Did you mean: {}?"
                   .format(wrong_opts, right_opts))
            print(msg)
            raise ValueError(msg)

        if self.colors and self.c_coo:
            raise ValueError("Cannot specify explicit colors if ``c`` used.")

        # Internal state
        self._data_range_calculated = False
        self._c_cols = []

    # ----------------------------- properties ------------------------------ #

    def _get_ds(self):
        return self._ds

    def _set_ds(self, ds):
        self._ds = ds
        self.prepare_z_vals()
        self.prepare_z_labels()
        self.prepare_xy_vals_lineplot()
        self.calc_plot_range()
        self.update()

    def _del_ds(self):
        self._ds = None

    ds = property(_get_ds, _set_ds, _del_ds,
                  "The dataset used to plot graph.")

    # ------------------------------- methods ------------------------------- #

    def prepare_axes_labels(self):
        """Work out what the axes titles should be.
        """
        self._xtitle = self.x_coo if self.xtitle is None else self.xtitle
        if self.ytitle is None:
            if isinstance(self.y_coo, str):
                self._ytitle = self.y_coo
            else:
                self._ytitle = None
        else:
            self._ytitle = self.ytitle

        if self.c_coo is None:
            self._ctitle = self.z_coo if self.ztitle is None else self.ztitle
        else:
            self._ctitle = self.c_coo if self.ctitle is None else self.ctitle

    def prepare_z_vals(self, mode='lineplot', grid=False):
        """Work out what the 'z-coordinate', if any, should be.

        Parameters
        ----------
        mode : {'lineplot', 'scatter', 'histogram'}
        """
        # TODO: process multi-var errors at same time
        self._multi_var = False

        # Multiple sets of data parametized by z_coo
        if self.z_coo is not None:
            if not grid:
                check_excess_dims(self._ds, self.y_coo,
                                  (self.x_coo, self.z_coo), mode=mode)
            self._z_vals = self._ds[self.z_coo].values

        # Multiple data variables to plot -- just named in list
        elif isinstance(self.y_coo, (tuple, list)):
            for var in self.y_coo:
                if grid:
                    check_excess_dims(self._ds, var, (self.x_coo,), mode=mode)
            self._multi_var = True
            self._z_vals = self.y_coo

        # Multiple data variables to plot -- but for histogram
        elif isinstance(self.x_coo, (tuple, list)):
            self._multi_var = True
            self._z_vals = self.x_coo

        # Single data variable to plot
        else:
            if not grid:
                check_excess_dims(self._ds, self.y_coo,
                                  (self.x_coo,), mode=mode)
            self._z_vals = (None,)

    def prepare_z_labels(self):
        """Work out what the labels for the z-coordinate should be.
        """
        # Manually specified z-labels
        if self.zlabels is not None:
            self._zlbls = iter(self.zlabels)

        # Use z-data (for lineplot only) or multiple y names
        elif (self.z_coo is not None) or self._multi_var:
            self._zlbls = iter(str(z) for z in self._z_vals)

        # No z-labels
        else:
            self._zlbls = itertools.repeat(None)

    def calc_use_legend_or_colorbar(self):
        """Work out whether to use a legend.
        """
        def auto_legend():
            return (1 < len(self._z_vals) <= 10)

        # cspecified, lspecified, many points, colors, colors, c_coo
        if self.colorbar and (self.legend is None):
            self.legend = False if (self.c_coo is None) else auto_legend()
        if self.legend and (self.colorbar is None):
            self.colorbar = False if (self.c_coo is None) else True

        if self.legend is None and self.colorbar is None:
            self._use_legend = auto_legend()
            self._use_colorbar = (((not self._use_legend) and
                                   self.colors is True) or
                                  self.c_coo is not None)
        else:
            self._use_legend = self.legend
            self._use_colorbar = self.colorbar

    def prepare_xy_vals_lineplot(self, mode='lineplot'):
        """Select and flatten the data appropriately to iterate over.
        """

        def gen_xy():
            for i, z in enumerate(self._z_vals):
                das = {}
                data = {}

                # multiple data variables rather than z coordinate
                if self._multi_var:
                    das['x'] = self._ds[self.x_coo]
                    das['y'] = self._ds[z]

                    if (self.y_err is not None) or \
                       (self.x_err is not None) or \
                       (self.c_coo is not None):
                        raise ValueError('Multi-var errors/c not implemented.')

                # z-coordinate to iterate over
                elif z is not None:
                    try:
                        # try positional indexing first, as much faster
                        sub_ds = self._ds[{self.z_coo: i}]
                    except ValueError:
                        # but won't work e.g. on non-dimensions
                        sub_ds = self._ds.loc[{self.z_coo: z}]

                    das['x'] = sub_ds[self.x_coo]
                    das['y'] = sub_ds[self.y_coo]

                    if self.c_coo is not None:
                        if mode == 'lineplot':
                            self._c_cols.append(np.asscalar(
                                sub_ds[self.c_coo].values.flatten()))
                        elif mode == 'scatter':
                            das['c'] = sub_ds[self.c_coo]

                    if self.y_err is not None:
                        das['ye'] = sub_ds[self.y_err]

                    if self.x_err is not None:
                        das['xe'] = sub_ds[self.x_err]

                # nothing to iterate over
                else:
                    das['x'] = self._ds[self.x_coo]
                    das['y'] = self._ds[self.y_coo]

                    if self.c_coo is not None:
                        if mode == 'lineplot':
                            self._c_cols.append(np.asscalar(
                                self._ds[self.c_coo].values.flatten()))
                        elif mode == 'scatter':
                            das['c'] = self._ds[self.c_coo]

                    if self.y_err is not None:
                        das['ye'] = self._ds[self.y_err]

                    if self.x_err is not None:
                        das['xe'] = self._ds[self.x_err]

                for k, da in zip(das, xr.broadcast(*das.values())):
                    data[k] = da.values.flatten()

                # Trim out missing data
                not_null = np.isfinite(data['x'])
                not_null &= np.isfinite(data['y'])

                # TODO: if scatter, broadcast *then* ravel x, y, c?

                data['x'] = data['x'][not_null]
                data['y'] = data['y'][not_null]
                if 'c' in data:
                    data['c'] = data['c'][not_null]
                if 'ye' in data:
                    data['ye'] = data['ye'][not_null]
                if 'xe' in data:
                    data['xe'] = data['xe'][not_null]

                yield data

        self._gen_xy = gen_xy

    def prepare_x_vals_histogram(self):
        """
        """

        def gen_x():
            for z in self._z_vals:
                # multiple data variables rather than z coordinate
                if self._multi_var:
                    x = self._ds[z].values.flatten()

                # z-coordinate to iterate over
                elif z is not None:
                    sub_ds = self._ds.loc[{self.z_coo: z}]
                    x = sub_ds[self.x_coo].values.flatten()

                # nothing to iterate over
                else:
                    x = self._ds[self.x_coo].values.flatten()

                yield {'x': x[np.isfinite(x)]}

        self._gen_xy = gen_x

    def prepare_heatmap_data(self, grid=False):
        """Prepare the data to go into a heatmap.
        """
        # Can only show one dataset
        self._multi_var = False

        self._heatmap_x = self._ds[self.x_coo].values.flatten()
        self._heatmap_y = self._ds[self.y_coo].values.flatten()

        if grid:
            self._zmin = self._ds[self.z_coo].min()
            self._zmax = self._ds[self.z_coo].max()
            return

        self._heatmap_var = ma.masked_invalid(
            self._ds[self.z_coo]
                .squeeze()
                .transpose(self.y_coo, self.x_coo)
                .values)
        self._zmin = self._heatmap_var.min()
        self._zmax = self._heatmap_var.max()

    def calc_color_norm(self):
        import matplotlib as mpl

        self.cmap = xyz_colormaps(self.colormap, reverse=self.colormap_reverse)

        coo = self.z_coo if self.c_coo is None else self.c_coo

        if coo is None:
            return

        # check if real numeric type
        if self._ds[coo].dtype.kind in {'i', 'u', 'f'}:
            self._zmin = self.zlims[0]
            if self._zmin is None:
                self._zmin = self._ds[coo].values.min()
            self._zmax = self.zlims[1]
            if self._zmax is None:
                self._zmax = self._ds[coo].values.max()
        else:
            # no relative coloring possible e.g. for strings
            self._zmin, self._zmax = 0.0, 1.0

        if self.vmin is None:
            self.vmin = self._zmin
        if self.vmax is None:
            self.vmax = self._zmax

        self._color_norm = getattr(
            mpl.colors, "LogNorm" if self.colormap_log else "Normalize")(
                vmin=self.vmin, vmax=self.vmax)

        self.set_mappable()

    def calc_line_colors(self):
        """Helper function for calculating what the colormapped color of each
        line should be.
        """
        self.calc_color_norm()

        # set from 'adjacent' variable
        if self.c_coo is not None:
            rvals = (self._color_norm(z) for z in self._c_cols)
        # set from coordinate
        else:
            if np.isreal(self._z_vals[0]):
                rvals = (self._color_norm(z) for z in self._z_vals)
            else:
                # no relative coloring possible e.g. for strings
                rvals = np.linspace(0, 1, len(self._z_vals))

        # Map relative value to mpl color
        self._cols = (self.cmap(rval) for rval in rvals)
        # Convert colors to correct format
        self._cols = iter(convert_colors(self._cols, outformat=self.backend))

    def prepare_colors(self):
        """Prepare the colors for the lines, based on the z-coordinate.
        """
        # Automatic colors
        if (self.colors is True) or (self.c_coo is not None):
            self.calc_line_colors()
        # Manually specified colors
        elif self.colors:
            self._cols = itertools.cycle(
                convert_colors(self.colors, outformat=self.backend))
        # Use sequential
        else:
            self._cols = get_default_sequential_cm(self.backend)

    def prepare_markers(self):
        """Prepare the markers to be used for each line.
        """
        if self.markers is None:
            # If no lines, always draw markers
            if self.lines is False:
                self.markers = True
            # Else decide on how many points
            else:
                self.markers = len(self._ds[self.x_coo]) <= 51

        # Could add more logic based on number of xpoints?
        if self.markersize is None:
            self._markersize = 5
        else:
            self._markersize = self.markersize

        if self.markers:
            if len(self._z_vals) > 1:
                self._mrkrs = itertools.cycle(_MARKERS[self.backend])
            else:
                self._mrkrs = iter(_SINGLE_LINE_MARKER[self.backend])
        else:
            self._mrkrs = itertools.repeat(None)

    def prepare_line_styles(self):
        """Line widths and styles.
        """
        self._lines = (
            itertools.repeat(" ") if self.lines is False else
            itertools.repeat("solid") if self.line_styles is None else
            itertools.cycle(self.line_styles))

        # Set custom widths for each line
        if self.line_widths is not None:
            self._lws = itertools.cycle(self.line_widths)
        else:
            self._lws = itertools.cycle([1.3])

    def prepare_zorders(self):
        """What order lines appear over one another
        """
        if self.zorders is not None:
            self._zordrs = itertools.cycle(self.zorders)
        else:
            self._zordrs = itertools.cycle([3])

    def calc_data_range(self, force=False):
        """
        """
        if not self._data_range_calculated or force:
            # x data range
            if self.x_coo is None:
                self._data_xmin, self._data_xmax = None, None
            elif isinstance(self.x_coo, str):
                self._data_xmin = float(self._ds[self.x_coo].min())
                self._data_xmax = float(self._ds[self.x_coo].max())
            else:
                self._data_xmin = float(min(self._ds[x].min()
                                            for x in self.x_coo))
                self._data_xmax = float(max(self._ds[x].max()
                                            for x in self.x_coo))

            # y data range
            if self.y_coo is None:
                self._data_ymin, self._data_ymax = None, None
            elif isinstance(self.y_coo, str):
                self._data_ymin = float(self._ds[self.y_coo].min())
                self._data_ymax = float(self._ds[self.y_coo].max())
            else:
                self._data_ymin = float(min(self._ds[y].min()
                                            for y in self.y_coo))
                self._data_ymax = float(max(self._ds[y].max()
                                            for y in self.y_coo))

            self._data_range_calculated = True

    def calc_plot_range(self):
        """Logic for processing limits and padding into plot ranges.
        """
        # Leave as default
        if self.xlims is None and self.padding is None:
            self._xlims = None
        else:
            if self.xlims is not None:
                xmin, xmax = self.xlims
            else:
                self.calc_data_range()
                xmin, xmax = self._data_xmin, self._data_xmax

            if xmin is None and xmax is None:
                self._xlims = None

            # increase plot range if padding specified
            elif self.padding is not None:
                xrnge = xmax - xmin
                self._xlims = (xmin - self.padding * xrnge,
                               xmax + self.padding * xrnge)
            else:
                self._xlims = xmin, xmax

        if self.ylims is None and self.padding is None:
            # Leave as default
            self._ylims = None
        else:
            if self.ylims is not None:
                ymin, ymax = self.ylims
            else:
                self.calc_data_range()
                ymin, ymax = self._data_ymin, self._data_ymax

            if ymin is None and ymax is None:
                self._ylims = None

            # increase plot range if padding specified
            elif self.padding is not None:
                yrnge = ymax - ymin
                self._ylims = (ymin - self.padding * yrnge,
                               ymax + self.padding * yrnge)
            else:
                self._ylims = ymin, ymax


# Abstract Plotters - contain the core preparation required for each plot type

class AbstractLinePlot:

    def prepare_data_single(self):
        self.prepare_axes_labels()
        self.prepare_z_vals(mode='lineplot')
        self.prepare_z_labels()
        self.calc_use_legend_or_colorbar()
        self.prepare_xy_vals_lineplot(mode='lineplot')
        self.prepare_colors()
        self.prepare_markers()
        self.prepare_line_styles()
        self.prepare_zorders()
        self.calc_plot_range()

    def prepare_data_multi_grid(self):
        self.prepare_axes_labels()
        self.prepare_z_vals(mode='lineplot', grid=True)
        self.calc_use_legend_or_colorbar()
        self.calc_color_norm()
        self.calc_data_range()


class AbstractScatter:

    def prepare_data_single(self):
        self.prepare_axes_labels()
        self.prepare_z_vals(mode='scatter')
        self.prepare_z_labels()
        self.calc_use_legend_or_colorbar()
        self.prepare_xy_vals_lineplot(mode='scatter')
        self.prepare_colors()
        self.prepare_markers()
        self.prepare_line_styles()
        self.prepare_zorders()
        self.calc_plot_range()

    def prepare_data_multi_grid(self):
        self.prepare_axes_labels()
        self.prepare_z_vals(mode='scatter', grid=True)
        self.calc_use_legend_or_colorbar()
        self.calc_color_norm()
        self.calc_data_range()


class AbstractHistogram:

    def prepare_data_single(self):
        self.prepare_axes_labels()
        self.prepare_z_vals(mode='histogram')
        self.prepare_z_labels()
        self.calc_use_legend_or_colorbar()
        self.prepare_x_vals_histogram()
        self.prepare_colors()
        self.prepare_line_styles()
        self.prepare_zorders()
        self.calc_plot_range()

    def prepare_data_multi_grid(self):
        self.prepare_axes_labels()
        self.prepare_z_vals(mode='histogram', grid=True)
        self.calc_use_legend_or_colorbar()
        self.calc_color_norm()


class AbstractHeatMap:

    def prepare_data_single(self):
        self.prepare_axes_labels()
        self.prepare_heatmap_data()
        self.calc_use_legend_or_colorbar()
        self.calc_plot_range()

    def prepare_data_multi_grid(self):
        self.prepare_axes_labels()
        self.prepare_heatmap_data(grid=True)
        self.calc_use_legend_or_colorbar()
        self.calc_color_norm()
        self.calc_data_range()


# Helpers for grid plots

def calc_row_col_datasets(ds, row=None, col=None):
    """
    """
    if row is not None:
        rs = ds[row].values
        nr = len(rs)

    if col is not None:
        cs = ds[col].values
        nc = len(cs)

    if row is None:
        return [[ds.loc[{col: c}] for c in cs]], 1, nc

    if col is None:
        return [[ds.loc[{row: r}]] for r in rs], nr, 1

    return [[ds.loc[{row: r, col: c}] for c in cs] for r in rs], nr, nc


def intercept_call_arg(fn):

    @functools.wraps(fn)
    def wrapped_fn(*args, **kwargs):
        call = kwargs.pop('call', True)
        P = fn(*args, **kwargs)

        if call == 'both':
            P()
            return P
        elif call:
            return P()
        else:
            return P

    return wrapped_fn


def prettify(x):

    if np.issubdtype(type(x), np.floating):
        x = "{0:0.4f}".format(x).rstrip('0')
        if x[-1] == '.':
            x += "0"

    return x
