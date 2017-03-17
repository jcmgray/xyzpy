"""
Helper functions for preparing data to be plotted.
"""
# TODO: Error bars ********************************************************** #
# TODO: allow dataArray ***************************************************** #
# TODO: x-err, upper and lower errors *************************************** #

import math
import itertools
import numpy as np
import numpy.ma as ma
import xarray as xr
from .color import convert_colors, xyz_colormaps
from .marker import _MARKERS, _SINGLE_LINE_MARKER

_PLOTTER_DEFAULTS = {
    # Figure options
    'backend': 'MATPLOTLIB',
    'figsize': (7, 6),
    'axes_loc': None,
    'add_to_axes': None,
    'add_to_fig': None,
    'subplot': None,
    'fignum': 1,
    'return_fig': True,
    # Coloring options
    'colors': None,
    'colormap': "xyz",
    'colormap_log': False,
    'colormap_reverse': False,
    # Colorbar options
    'colorbar': False,
    'vmin': None,
    'vmax': None,
    'colorbar_relative_position': None,
    'colorbar_opts': dict(),
    'method': None,
    'colorbar_color': "black",
    # 'z-axis' options: for legend or colorbar
    'ztitle': None,
    'zlabels': None,
    'zlims': (None, None),
    'zticks': None,
    'legend': None,
    'legend_loc': 0,
    'legend_ncol': 1,
    'legend_bbox': None,
    'legend_marker_scale': None,
    'legend_label_spacing': None,
    'legend_column_spacing': 1,
    'legend_frame': False,
    'legend_handlelength': None,
    # x-axis options
    'xtitle': None,
    'xtitle_pad': 5,
    'xlims': None,
    'xticks': None,
    'xticklabels_hide': False,
    'xlog': False,
    # y-axis options
    'ytitle': None,
    'ytitle_pad': 5,
    'ytitle_right': False,
    'ylims': None,
    'yticks': None,
    'yticklabels_hide': False,
    'ylog': False,
    # Titles and text
    'title': None,
    'panel_label': None,
    'panel_label_loc': (0.05, 0.85),
    # Styling options
    'markers': None,
    'markersize': None,
    'lines': True,
    'line_styles': None,
    'line_widths': None,
    'errorbar_capthick': 1,
    'errorbar_capsize': 0,
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
}

_PLOTTER_OPTS = list(_PLOTTER_DEFAULTS.keys())


class LinePlotter:
    def __init__(self, ds, y_coo, x_coo, z_coo=None, y_err=None, **kwargs):
        """
        """
        if isinstance(ds, xr.DataArray):
            self._ds = ds.to_dataset()
        else:
            self._ds = ds
        self.y_coo = y_coo
        self.x_coo = x_coo
        self.z_coo = z_coo
        self.y_err = y_err
        # Figure options
        settings = {**_PLOTTER_DEFAULTS, **kwargs}
        for opt in _PLOTTER_OPTS:
            setattr(self, opt, settings.pop(opt))

        if len(settings) > 0:
            import difflib
            wrong_opts = list(settings.keys())
            right_opts = [difflib.get_close_matches(opt, _PLOTTER_OPTS, n=3)
                          for opt in wrong_opts]

            raise ValueError("Option(s) {} not valid.\n Did you mean: {}?"
                             .format(wrong_opts, right_opts))

        # Internal
        self._data_range_calculated = False

    # ----------------------------- properties ------------------------------ #

    def _get_ds(self):
        return self._ds

    def _set_ds(self, ds):
        self._ds = ds
        self.prepare_z_vals()
        self.prepare_z_labels()
        self.prepare_xy_vals()
        self.calc_plot_range()
        self.update()

    def _del_ds(self):
        self._ds = None

    ds = property(_get_ds, _set_ds, _del_ds,
                  "The dataset used to plot graph.")

    # ------------------------------- methods ------------------------------- #

    def prepare_z_vals(self):
        """Work out what the 'z-coordinate', if any, should be.
        """
        # TODO: process multi-var errors at same time
        self._multi_var = False

        if self.z_coo is not None:
            self._z_vals = self._ds[self.z_coo].values
        elif not isinstance(self.y_coo, str):
            self._multi_var = True
            self._z_vals = self.y_coo
        else:
            self._z_vals = (None,)

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

    def prepare_z_labels(self):
        """Work out what the labels for the z-coordinate should be.
        """
        if self.zlabels is not None:
            self._zlbls = iter(self.zlabels)
        elif self.z_coo is not None or self._multi_var:
            self._zlbls = iter(str(z) for z in self._z_vals)
        else:
            self._zlbls = itertools.repeat(None)

    def prepare_xy_vals(self):
        """Select and flatten the data appropriately to iterate over.
        """

        def gen_xy():
            for z in self._z_vals:
                # multiple data variables rather than z coordinate
                if self._multi_var:
                    x = self._ds[self.x_coo].values.flatten()
                    y = self._ds[z].values.flatten()

                    if self.y_err is not None:
                        raise ValueError('Multi-var errors not implemented.')

                # z-coordinate to iterate over
                elif z is not None:
                    sub_ds = self._ds.loc[{self.z_coo: z}]
                    x = sub_ds[self.x_coo].values.flatten()
                    y = sub_ds[self.y_coo].values.flatten()

                    if self.y_err is not None:
                        ye = sub_ds[self.y_err].values.flatten()

                # nothing to iterate over
                else:
                    x = self._ds[self.x_coo].values.flatten()
                    y = self._ds[self.y_coo].values.flatten()

                    if self.y_err is not None:
                        ye = self._ds[self.y_err].values.flatten()

                # Trim out missing data
                not_null = np.isfinite(x)
                not_null &= np.isfinite(y)

                if self.y_err is not None:
                    yield x[not_null], y[not_null], ye[not_null]
                else:
                    yield x[not_null], y[not_null]

        self._gen_xy = gen_xy

    def prepare_heatmap_data(self):
        """Prepare the data to go into a heatmap.
        """
        self._heatmap_x = self._ds[self.x_coo].values.flatten()
        self._heatmap_y = self._ds[self.y_coo].values.flatten()
        self._heatmap_var = ma.masked_invalid(
            self._ds[self.z_coo]
                .squeeze()
                .transpose(self.y_coo, self.x_coo)
                .values)
        self._zmin = self._heatmap_var.min()
        self._zmax = self._heatmap_var.max()

    def calc_use_legend(self):
        """Work out whether to use a legend.
        """
        if self.legend is None:
            self._lgnd = (1 < len(self._z_vals) <= 10)
        else:
            self._lgnd = self.legend

    def prepare_colors(self):
        """Prepare the colors for the lines, based on the z-coordinate.
        """
        if self.colors is True:
            self.calc_colors()
        elif self.colors:
            self._cols = itertools.cycle(
                convert_colors(self.colors, outformat=self.backend))
        else:
            if self.backend == 'BOKEH':
                from bokeh.palettes import Category10_9
                self._cols = itertools.cycle(Category10_9)
            else:
                self._cols = itertools.repeat(None)

    def calc_colors(self):
        """Helper function for calculating what each color should be.
        """
        cmap = xyz_colormaps(self.colormap)

        try:
            self._zmin = self.zlims[0]
            if self._zmin is None:
                self._zmin = self._ds[self.z_coo].values.min()
            self._zmax = self.zlims[1]
            if self._zmax is None:
                self._zmax = self._ds[self.z_coo].values.max()

            # Relative function
            f = math.log if self.colormap_log else lambda a: a
            # Relative place in range according to function
            rvals = [1 - (f(z) - f(self._zmin)) /
                         (f(self._zmax) - f(self._zmin))
                     for z in self._ds[self.z_coo].values]
        except TypeError:  # no relative coloring possible e.g. for strings
            rvals = np.linspace(0, 1.0, self._ds[self.z_coo].size)

        # Map to mpl colormap, reversing if required
        self._cols = [cmap(1 - rval if self.colormap_reverse else rval)
                      for rval in rvals]
        self._cols = iter(convert_colors(self._cols, outformat=self.backend))

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
            self._data_xmin = float(self._ds[self.x_coo].min())
            self._data_xmax = float(self._ds[self.x_coo].max())

            # y data range
            if self._multi_var:
                self._data_ymin = float(min(self._ds[var].min()
                                            for var in self.y_coo))
                self._data_ymax = float(max(self._ds[var].max()
                                            for var in self.y_coo))
            else:
                self._data_ymin = float(self._ds[self.y_coo].min())
                self._data_ymax = float(self._ds[self.y_coo].max())

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

            # increase plot range if padding specified
            if self.padding is not None:
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

            # increase plot range if padding specified
            if self.padding is not None:
                yrnge = ymax - ymin
                self._ylims = (ymin - self.padding * yrnge,
                               ymax + self.padding * yrnge)
            else:
                self._ylims = ymin, ymax
