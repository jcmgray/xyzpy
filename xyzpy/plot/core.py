"""
Helper functions for preparing data to be plotted.
"""
# TODO: Error bars ********************************************************** #
# TODO: allow dataArray ***************************************************** #

import math
import itertools
import numpy as np
from .color import convert_colors, _xyz_colormaps
from .marker import _MARKERS, _SINGLE_LINE_MARKER


class LinePlotter:
    def __init__(self, ds, y_coo, x_coo, z_coo=None,
                 engine='MATPLOTLIB',
                 # Figure options
                 figsize=(8, 6),          # absolute figure size
                 axes_loc=None,           # axes location within fig
                 add_to_axes=None,        # add to existing axes
                 add_to_fig=None,         # add plot to an exisitng figure
                 subplot=None,            # make plot in subplot
                 fignum=1,
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
                 zlims=(None, None),      # Scaling limits for the colormap
                 legend_ncol=1,           # number of columns in the legend
                 legend_bbox=None,        # Where to anchor the legend to
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
                 vlines=None,             # vertical line positions to plot
                 hlines=None,             # horizontal line positions to plot
                 gridlines=True,
                 font=('Source Sans Pro', 'PT Sans',
                       'Liberation Sans', 'Arial'),
                 fontsize_title=20,
                 fontsize_ticks=16,
                 fontsize_xtitle=20,
                 fontsize_ytitle=20,
                 fontsize_ztitle=20,
                 fontsize_zlabels=18,
                 return_fig=True):
        """
        """
        self.ds = ds
        self.y_coo = y_coo
        self.x_coo = x_coo
        self.z_coo = z_coo
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
        self.legend = legend
        self.legend_loc = legend_loc
        self.ztitle = ztitle
        self.zlabels = zlabels
        self.zlims = zlims
        self.legend_ncol = legend_ncol
        self.legend_bbox = legend_bbox
        self.xtitle = xtitle
        self.xtitle_pad = xtitle_pad
        self.xlims = xlims
        self.xticks = xticks
        self.xticklabels_hide = xticklabels_hide
        self.xlog = xlog
        self.ytitle = ytitle
        self.ytitle_pad = ytitle_pad
        self.ylims = ylims
        self.yticks = yticks
        self.yticklabels_hide = yticklabels_hide
        self.ylog = ylog
        self.markers = markers
        self.line_styles = line_styles
        self.line_widths = line_widths
        self.zorders = zorders
        self.padding = padding
        self.vlines = vlines
        self.hlines = hlines
        self.gridlines = gridlines
        self.font = font
        self.fontsize_title = fontsize_title
        self.fontsize_ticks = fontsize_ticks
        self.fontsize_xtitle = fontsize_xtitle
        self.fontsize_ytitle = fontsize_ytitle
        self.fontsize_ztitle = fontsize_ztitle
        self.fontsize_zlabels = fontsize_zlabels
        self.return_fig = return_fig

        # Internal
        self._multi_var = False

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

    def prepare_z_vals(self):
        """
        """
        if self.z_coo is not None:
            self._z_vals = self.ds[self.z_coo].values
        elif not isinstance(self.y_coo, str):
            self._multi_var = True
            self._z_vals = self.y_coo
        else:
            self._z_vals = (None,)

    def prepare_axes_labels(self):
        """
        """
        self._xtitle = self.x_coo if self.xtitle is None else self.xtitle
        if self.ytitle is None and isinstance(self.y_coo, str):
            self._ytitle = self.y_coo
        else:
            self._ytitle = None

    def prepare_xy_vals(self):
        """
        """

        def gen_xy():
            for z in self._z_vals:
                # Select data for current z coord - flatten for singletons
                if self._multi_var:
                    # multiple data variables rather than z coordinate
                    x = self.ds[self.x_coo].values.flatten()
                    y = self.ds[z].values.flatten()
                elif z is not None:
                    # z-coordinate to iterate over
                    x = (self.ds.loc[{self.z_coo: z}][self.x_coo]
                                .values.flatten())
                    y = (self.ds.loc[{self.z_coo: z}][self.y_coo]
                                .values.flatten())
                else:
                    # nothing to iterate over
                    x = self.ds[self.x_coo].values.flatten()
                    y = self.ds[self.y_coo].values.flatten()

                # Trim out missing data
                notnull = ~np.isnan(x) & ~np.isnan(y)
                yield x[notnull], y[notnull]

        self._gen_xy = gen_xy

    def prepare_z_labels(self):
        """
        """
        if self.zlabels is not None:
            self._zlbls = iter(self.zlabels)
        elif self.z_coo is not None or self._multi_var:
            self._zlbls = iter(str(z) for z in self._z_vals)
        else:
            self._zlbls = itertools.repeat(None)

    def calc_use_legend(self):
        if self.legend is None:
            self._lgnd = (1 < len(self._z_vals) <= 10)
        else:
            self._lgnd = self.legend

    def prepare_colors(self, engine):
        """
        """
        if self.colors is True:
            self.calc_colors(engine=engine)
        elif self.colors:
            self._cols = itertools.cycle(convert_colors(self.colors,
                                                        outformat=engine))
        else:
            if engine == 'BOKEH':
                from bokeh.palettes import Dark2_8
                self._cols = itertools.cycle(Dark2_8)
            else:
                self._cols = itertools.repeat(None)

    def calc_colors(self, engine):
        """
        """
        cmap = _xyz_colormaps(self.colormap)

        try:
            zmin = self.zlims[0]
            if zmin is None:
                zmin = self.ds[self.z_coo].values.min()
            zmax = self.zlims[1]
            if zmax is None:
                zmax = self.ds[self.z_coo].values.max()

            # Relative function
            f = math.log if self.colormap_log else lambda a: a
            # Relative place in range according to function
            rvals = [1 - (f(z) - f(zmin)) / (f(zmax) - f(zmin))
                     for z in self.ds[self.z_coo].values]
        except TypeError:  # no relative coloring possible e.g. for strings
            rvals = np.linspace(0, 1.0, self.ds[self.z_coo].size)

        # Map to mpl colormap, reversing if required
        self._cols = [cmap(1 - rval if self.colormap_reverse else rval)
                      for rval in rvals]
        self._cols = iter(convert_colors(self._cols, outformat=engine))

    def prepare_markers(self, engine):
        """
        """
        if self.markers is None:
            self.markers = len(self.ds[self.x_coo]) <= 51

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
        self._lines = (itertools.repeat("-") if self.line_styles is None else
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

    def calc_plot_range(self):
        """Logic for processing limits and padding into plot ranges.
        """
        if self.xlims is None and self.padding is None:
            # Leave as default
            self._xlims = None
        else:
            if self.xlims is not None:
                xmin, xmax = self.xlims
            else:
                xmax, xmin = (self.ds[self.x_coo].max(),
                              self.ds[self.x_coo].min())
                xmin, xmax = float(xmin), float(xmax)
            if self.padding is not None:
                xrnge = xmax - xmin
                self._xlims = (xmin - self.padding * xrnge,
                               xmax + self.padding * xrnge)
            else:
                self._xlims = self.xlims

        if self.ylims is not None or self.padding is not None:
            if self.ylims is not None:
                ymin, ymax = self.ylims
            else:
                if isinstance(self.y_coo, str):
                    ymax, ymin = (self.ds[self.y_coo].max(),
                                  self.ds[self.y_coo].min())
                else:
                    ymax = max(self.ds[var].max() for var in self.y_coo)
                    ymin = min(self.ds[var].min() for var in self.y_coo)
            ymin, ymax = float(ymin), float(ymax)
            if self.padding is not None:
                yrnge = ymax - ymin
                self._ylims = (ymin - self.padding * yrnge,
                               ymax + self.padding * yrnge)
            else:
                self._ylims = self.ylims
        else:
            self._ylims = None
