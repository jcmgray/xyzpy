"""
Helper functions for preparing data to be plotted.
"""
# TODO: Error bars ********************************************************** #
# TODO: allow dataArray ***************************************************** #
# TODO: x-err, upper and lower errors *************************************** #

import math
import itertools
import numpy as np
import xarray as xr
from .color import convert_colors, _xyz_colormaps
from .marker import _MARKERS, _SINGLE_LINE_MARKER


class LinePlotter:
    def __init__(self, ds, y_coo, x_coo, z_coo=None, y_err=None,
                 engine='MATPLOTLIB',
                 # Figure options
                 figsize=(8, 6),           # absolute figure size
                 axes_loc=None,            # axes location within fig
                 add_to_axes=None,         # add to existing axes
                 add_to_fig=None,          # add plot to an exisitng figure
                 subplot=None,             # make plot in subplot
                 fignum=1,
                 title=None,
                 # Line coloring options
                 colors=None,
                 colormap="xyz",
                 colormap_log=False,
                 colormap_reverse=False,
                 colorbar=False,
                 # Legend options
                 ztitle=None,               # legend title
                 zlabels=None,              # legend labels
                 zlims=(None, None),        # Scaling limits for the colormap
                 legend=None,               # shoow legend or not
                 legend_loc=0,              # legend location
                 legend_ncol=1,             # number of columns in the legend
                 legend_bbox=None,          # Where to anchor the legend to
                 legend_markerscale=None,   # size of the legend markers
                 legend_labelspacing=None,  # vertical spacing
                 legend_columnspacing=3,    # horizontal spacing
                 legend_frame=False,
                 # x-axis options
                 xtitle=None,
                 xtitle_pad=10,           # distance between label and axes
                 xlims=None,              # plotting range on x axis
                 xticks=None,             # where to place x ticks
                 xticklabels_hide=False,  # hide labels but not actual ticks
                 xlog=False,              # logarithmic x scale
                 # y-axis options
                 ytitle=None,
                 ytitle_pad=10,           # distance between label and axes
                 ytitle_right=False,      # draw ytitle on right handside
                 ylims=None,              # plotting range on y-axis
                 yticks=None,             # where to place y ticks
                 yticklabels_hide=False,  # hide labels but not actual ticks
                 ylog=False,              # logarithmic y scale
                 # Shapes
                 markers=None,      # use markers for each plotted point
                 markersize=None,   # size of markers
                 lines=True,
                 line_styles=None,  # iterable of line-styles, e.g. '--'
                 line_widths=None,  # iterable of line-widths
                 zorders=None,      # draw order
                 # Misc options
                 padding=None,           # plot range padding (as fraction)
                 vlines=None,            # vertical line positions to plot
                 hlines=None,            # horizontal line positions to plot
                 span_style='--',        # style of the above lines
                 gridlines=True,         # show gridlines or not
                 gridline_style=(1, 3),  # linestyle of the gridlines
                 font=('Source Sans Pro', 'PT Sans',
                       'Liberation Sans', 'Arial'),
                 fontsize_title=20,
                 fontsize_ticks=16,
                 fontsize_xtitle=20,
                 fontsize_ytitle=20,
                 fontsize_ztitle=20,
                 fontsize_zlabels=18,
                 math_serif=False,       # Use serif fonts for math text
                 return_fig=True):
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
        self.figsize = figsize
        self.axes_loc = axes_loc
        self.add_to_axes = add_to_axes
        self.add_to_fig = add_to_fig
        self.subplot = subplot
        self.fignum = fignum
        self.title = title
        self.colors = colors
        self.colormap = colormap
        self.colormap_log = colormap_log
        self.colormap_reverse = colormap_reverse
        self.colorbar = colorbar
        self.ztitle = ztitle
        self.zlabels = zlabels
        self.zlims = zlims
        self.legend = legend
        self.legend_loc = legend_loc
        self.legend_ncol = legend_ncol
        self.legend_bbox = legend_bbox
        self.legend_markerscale = legend_markerscale
        self.legend_labelspacing = legend_labelspacing
        self.legend_columnspacing = legend_columnspacing
        self.legend_frame = legend_frame
        self.xtitle = xtitle
        self.xtitle_pad = xtitle_pad
        self.xlims = xlims
        self.xticks = xticks
        self.xticklabels_hide = xticklabels_hide
        self.xlog = xlog
        self.ytitle = ytitle
        self.ytitle_pad = ytitle_pad
        self.ytitle_right = ytitle_right
        self.ylims = ylims
        self.yticks = yticks
        self.yticklabels_hide = yticklabels_hide
        self.ylog = ylog
        self.markers = markers
        self.markersize = markersize
        self.lines = lines
        self.line_styles = line_styles
        self.line_widths = line_widths
        self.zorders = zorders
        self.padding = padding
        self.vlines = vlines
        self.hlines = hlines
        self.span_style = span_style
        self.gridlines = gridlines
        self.gridline_style = gridline_style
        self.font = font
        self.fontsize_title = fontsize_title
        self.fontsize_ticks = fontsize_ticks
        self.fontsize_xtitle = fontsize_xtitle
        self.fontsize_ytitle = fontsize_ytitle
        self.fontsize_ztitle = fontsize_ztitle
        self.fontsize_zlabels = fontsize_zlabels
        self.math_serif = math_serif
        self.return_fig = return_fig

        # Internal
        self._data_range_calculated = False

        # Prepare
        self.prepare_z_vals()
        self.prepare_axes_labels()
        self.prepare_z_labels()
        self.calc_use_legend()
        self.prepare_xy_vals()
        self.prepare_colors(engine)
        self.prepare_markers(engine)
        self.prepare_line_styles(engine)
        self.prepare_zorders()
        self.calc_plot_range()

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

    def prepare_z_labels(self):
        """Work out what the labels for the z-coordinate should be.
        """
        if self.zlabels is not None:
            self._zlbls = iter(self.zlabels)
        elif self.z_coo is not None or self._multi_var:
            self._zlbls = iter(str(z) for z in self._z_vals)
        else:
            self._zlbls = itertools.repeat(None)

    def calc_use_legend(self):
        """Work out whether to use a legend.
        """
        if self.legend is None:
            self._lgnd = (1 < len(self._z_vals) <= 10)
        else:
            self._lgnd = self.legend

    def prepare_colors(self, engine):
        """Prepare the colors for the lines, based on the z-coordinate.
        """
        if self.colors is True:
            self.calc_colors(engine=engine)
        elif self.colors:
            self._cols = itertools.cycle(convert_colors(self.colors,
                                                        outformat=engine))
        else:
            if engine == 'BOKEH':
                from bokeh.palettes import Category10_9
                self._cols = itertools.cycle(Category10_9)
            else:
                self._cols = itertools.repeat(None)

    def calc_colors(self, engine):
        """Helper function for calculating what each color should be.
        """
        cmap = _xyz_colormaps(self.colormap)

        try:
            zmin = self.zlims[0]
            if zmin is None:
                zmin = self._ds[self.z_coo].values.min()
            zmax = self.zlims[1]
            if zmax is None:
                zmax = self._ds[self.z_coo].values.max()

            # Relative function
            f = math.log if self.colormap_log else lambda a: a
            # Relative place in range according to function
            rvals = [1 - (f(z) - f(zmin)) / (f(zmax) - f(zmin))
                     for z in self._ds[self.z_coo].values]
        except TypeError:  # no relative coloring possible e.g. for strings
            rvals = np.linspace(0, 1.0, self._ds[self.z_coo].size)

        # Map to mpl colormap, reversing if required
        self._cols = [cmap(1 - rval if self.colormap_reverse else rval)
                      for rval in rvals]
        self._cols = iter(convert_colors(self._cols, outformat=engine))

    def prepare_markers(self, engine):
        """Prepare the markers to be used for each line.
        """
        if self.markers is None:
            # If no lines, always draw markers
            if self.lines is False:
                self.markers = True
            # Else decide on how many points
            else:
                self.markers = len(self._ds[self.x_coo]) <= 51

        if self.markers:
            if len(self._z_vals) > 1:
                self._mrkrs = itertools.cycle(_MARKERS[engine])
            else:
                self._mrkrs = iter(_SINGLE_LINE_MARKER[engine])
        else:
            self._mrkrs = itertools.repeat(None)

    def prepare_line_styles(self, engine):
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
