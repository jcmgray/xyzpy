"""
Functions for plotting datasets nicely.
"""                                  #
# TODO: error bars                                                            #
# TODO: mshow? Remove any auto, context sensitive (use backend etc.)          #
# TODO: custom xtick labels                                                   #
# TODO: annotations, arbitrary text                                           #
# TODO: docs                                                                  #
# TODO: mpl heatmap                                                           #
# TODO: detect zeros in plotting coordinates and adjust padding auto          #


import numpy as np
from ..manage import auto_xyz_ds
from .core import LinePlotter


# ----------------- Main lineplot interface for matplotlib ------------------ #

class LinePlotterMPL(LinePlotter):
    def __init__(self, *args, **kwargs):
        """
        """
        super().__init__(*args, engine='MATPLOTLIB', **kwargs)
        self.prepare_plot()
        self.set_axes_labels()
        self.set_axes_scale()
        self.set_axes_range()
        self.set_spans()
        self.set_gridlines()
        self.set_tick_marks()
        self.plot_lines()
        self.plot_legend()

    def prepare_plot(self):
        """
        """
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        mpl.rc("font", family=self.font)

        # Add a new set of axes to an existing plot
        if self.add_to_fig is not None and self.subplot is None:
            self._fig = self.add_to_fig
            self._axes = self._fig.add_axes((0.4, 0.6, 0.30, 0.25)
                                            if self.axes_loc is None else
                                            self.axes_loc)
        # Add lines to an existing set of axes
        elif self.add_to_axes is not None:
            self._fig = self.add_to_axes
            self._axes = self._fig.get_axes()[0]

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
                                            if self.axes_loc is None else
                                            self.axes_loc)
        self._axes.set_title("" if self.title is None else self.title,
                             fontsize=self.fontsize_title)

    def set_axes_labels(self):
        if self._xtitle:
            self._axes.set_xlabel(self._xtitle, fontsize=self.fontsize_xtitle)
            self._axes.xaxis.labelpad = self.xtitle_pad
        if self._ytitle:
            self._axes.set_ylabel(self._ytitle, fontsize=self.fontsize_ytitle)
            self._axes.yaxis.labelpad = self.ytitle_pad

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
                self._axes.axvline(x, color="0.5", linestyle="dashed")
        if self.hlines is not None:
            for y in self.hlines:
                self._axes.axhline(y, color="0.5", linestyle="dashed")

    def set_gridlines(self):
        """
        """
        if self.gridlines:
            self._axes.set_axisbelow(True)  # ensures gridlines below all
            self._axes.grid(True, color="0.666")

    def set_tick_marks(self):
        """
        """
        import matplotlib as mpl

        if self.xticks is not None:
            self._axes.set_xticks(self.xticks)
            (self._axes.get_xaxis()
             .set_major_formatter(mpl.ticker.ScalarFormatter()))
        if self.yticks is not None:
            self._axes.set_yticks(self.yticks)
            (self._axes.get_yaxis()
             .set_major_formatter(mpl.ticker.ScalarFormatter()))
        if self.xticklabels_hide:
            (self._axes.get_xaxis()
             .set_major_formatter(mpl.ticker.NullFormatter()))
        if self.yticklabels_hide:
            (self._axes.get_yaxis()
             .set_major_formatter(mpl.ticker.NullFormatter()))
        self._axes.tick_params(labelsize=self.fontsize_ticks, direction='out')

    def plot_lines(self):
        """
        """
        for x, y in self._gen_xy():
            col = next(self._cols)

            # add line to axes, with options cycled through
            self._axes.plot(x, y, next(self._lines),
                            c=col,
                            lw=next(self._lws),
                            marker=next(self._mrkrs),
                            markeredgecolor=col,
                            label=next(self._zlbls),
                            zorder=next(self._zordrs))

    def plot_legend(self):
        """Add a legend
        """
        if self._lgnd:
            lgnd = self._axes.legend(title=(self.z_coo if self.ztitle is None
                                            else self.ztitle),
                                     loc=self.legend_loc,
                                     fontsize=self.fontsize_zlabels,
                                     frameon=False,
                                     bbox_to_anchor=self.legend_bbox,
                                     ncol=self.legend_ncol)
            lgnd.get_title().set_fontsize(self.fontsize_ztitle)

    def show(self):
        """
        """
        import matplotlib.pyplot as plt

        if self.return_fig:
            plt.close(self._fig)
            return self._fig


def lineplot(ds, y_coo, x_coo, z_coo=None, return_fig=True, **kwargs):
    """
    """
    p = LinePlotterMPL(ds, y_coo, x_coo, z_coo,
                       return_fig=return_fig, **kwargs)
    return p.show()


def xyz_lineplot(x, y_z, **lineplot_opts):
    """ Take some x-coordinates and an array, convert them to a Dataset
    treating as multiple lines, then send to lineplot. """
    ds = auto_xyz_ds(x, y_z)
    # Plot dataset
    return lineplot(ds, 'y', 'x', 'z', **lineplot_opts)


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
    from xyzpy.plot.color import _xyz_colormaps

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

        ax = fig.add_subplot(*subplot)
        ax.imshow(img, cmap=_xyz_colormaps(colormap), interpolation='nearest',
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
