import collections
import contextlib
import functools
import itertools
import warnings

import numpy as np

from .color import cmoke
from .plotter_matplotlib import (
    add_visualize_legend,
    show_and_close,
    to_colors,
)


@functools.lru_cache(16)
def get_neutral_style(draw_color=(0.5, 0.5, 0.5)):
    return {
        "axes.edgecolor": draw_color,
        "axes.facecolor": (0, 0, 0, 0),
        "axes.grid": True,
        "axes.labelcolor": draw_color,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "figure.facecolor": (0, 0, 0, 0),
        "grid.alpha": 0.1,
        "grid.color": draw_color,
        "legend.frameon": False,
        "text.color": draw_color,
        "xtick.color": draw_color,
        "xtick.minor.visible": True,
        "ytick.color": draw_color,
        "ytick.minor.visible": True,
    }


def use_neutral_style(fn):
    """Decorator to use xyzpy neutral style for a function."""
    import matplotlib as mpl

    @functools.wraps(fn)
    def new_fn(
        *args, use_neutral_style=True, draw_color=(0.5, 0.5, 0.5), **kwargs
    ):
        if not use_neutral_style:
            return fn(*args, **kwargs)

        style = get_neutral_style(draw_color=draw_color)

        with mpl.rc_context(style):
            return fn(*args, **kwargs)

    return new_fn


@contextlib.contextmanager
def neutral_style(draw_color=(0.5, 0.5, 0.5), **kwargs):
    import matplotlib as mpl

    style = {
        **get_neutral_style(draw_color=draw_color),
        **kwargs,
    }

    with mpl.rc_context(style):
        yield


# colorblind palettes by Okabe & Ito: https://siegal.bio.nyu.edu/color-palette/

_COLORS_DEFAULT = (
    "#56B4E9",  # light blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # red
    "#F0E442",  # yellow
    "#CC79A7",  # purple
    "#0072B2",  # dark blue
)
_COLORS_SORTED = (
    "#0072B2",  # dark blue
    "#56B4E9",  # light blue
    "#009E73",  # green
    "#F0E442",  # yellow
    "#E69F00",  # orange
    "#D55E00",  # red
    "#CC79A7",  # purple
)


@functools.lru_cache(6)
def get_default_colormaps(N):
    return (
        cmoke(0.65, hue_shift=-0.05, val2=0.75),
        cmoke(0.13, hue_shift=-0.05, val2=0.80),
        cmoke(0.38, hue_shift=-0.05, val2=0.70),
        cmoke(0.04, hue_shift=-0.05, val2=0.60),
        cmoke(0.24, hue_shift=-0.05, val2=0.90),
        cmoke(0.90, hue_shift=-0.05, val2=0.75),
    )[:N]


def mod_sat(c, mod):
    """Modify the luminosity of rgb color ``c``."""
    from matplotlib.colors import hsv_to_rgb, rgb_to_hsv

    h, s, v = rgb_to_hsv(c[:3])
    return (*hsv_to_rgb((h, mod * s, v)), 1.0)


def auto_colors(N):
    import math

    from matplotlib.colors import LinearSegmentedColormap

    if N < len(_COLORS_DEFAULT):
        return _COLORS_DEFAULT[:N]

    cmap = LinearSegmentedColormap.from_list("okabe-ito", _COLORS_SORTED)

    xs = list(map(cmap, np.linspace(0, 1.0, N)))

    # modulate color saturation with sine to generate local distinguishability
    # ... but only turn on gradually for increasing number of nodes
    sat_mod_period = min(4, N / 7)
    sat_mod_factor = max(0.0, 2 / 3 * math.tanh((N - 7) / 4))

    return [
        mod_sat(
            c, 1 - sat_mod_factor * math.sin(math.pi * i / sat_mod_period) ** 2
        )
        for i, c in enumerate(xs)
    ]


def color_to_colormap(c, **autohue_opts):
    import matplotlib as mpl

    rgb = mpl.colors.to_rgb(c)
    h, s, v = mpl.colors.rgb_to_hsv(rgb)

    return cmoke(h, **autohue_opts)

    # vhi = min(1.0, v + vdiff / 2)
    # vlo = max(0.0, vhi - vdiff)
    # vhi = vlo + vdiff

    # shi = min(1.0, s + sdiff / 2)
    # slo = max(0.0, shi - sdiff)
    # shi = slo + sdiff

    # hsv_i = (h, max(slo, 0.0), min(vhi, 1.0))
    # hsv_f = (h, min(shi, 1.0), max(vlo, 0.0))

    # c1 = mpl.colors.hsv_to_rgb(hsv_i)
    # c2 = mpl.colors.hsv_to_rgb(hsv_f)
    # cdict = {
    #     "red": [(0.0, c1[0], c1[0]), (1.0, c2[0], c2[0])],
    #     "green": [(0.0, c1[1], c1[1]), (1.0, c2[1], c2[1])],
    #     "blue": [(0.0, c1[2], c1[2]), (1.0, c2[2], c2[2])],
    # }
    # return mpl.colors.LinearSegmentedColormap("", cdict)


def get_default_cmap(i, vdiff=0.5, sdiff=0.25):
    return color_to_colormap(_COLORS_DEFAULT[i], vdiff=vdiff, sdiff=sdiff)


def to_colormap(c, **autohue_opts):
    import numbers

    import matplotlib as mpl
    from matplotlib import pyplot as plt

    if isinstance(c, mpl.colors.Colormap):
        return c

    if isinstance(c, numbers.Number):
        return cmoke(c, **autohue_opts)

    try:
        return plt.get_cmap(c)
    except ValueError:
        return color_to_colormap(c, **autohue_opts)


def _make_bold(s):
    return r"$\bf{" + s.replace("_", r"\_") + r"}$"


_LINESTYLES_DEFAULT = (
    "solid",
    (0.0, (3, 1)),
    (0.5, (1, 1)),
    (1.0, (3, 1, 1, 1)),
    (1.5, (3, 1, 3, 1, 1, 1)),
    (2.0, (3, 1, 1, 1, 1, 1)),
)


_MARKERS_DEFAULT = (
    "o",
    "X",
    "v",
    "s",
    "P",
    "D",
    "^",
    "h",
    "*",
    "p",
    "<",
    "d",
    "8",
    ">",
    "H",
)

INFINIPLOTTER_DEFAULTS = dict(
    bins=None,
    bins_density=True,
    aggregate=None,
    aggregate_method="median",
    aggregate_err_range=0.5,
    err=None,
    err_style=None,
    err_kws=None,
    xlink=None,
    color=None,
    colors=None,
    color_order=None,
    color_label=None,
    color_ticklabels=None,
    colormap_start=0.0,
    colormap_stop=1.0,
    hue=None,
    hues=None,
    hue_order=None,
    hue_label=None,
    hue_ticklabels=None,
    palette=None,
    autohue_start=0.25,
    autohue_sweep=1.0,
    autohue_opts=None,
    marker=None,
    markers=None,
    marker_order=None,
    marker_label=None,
    marker_ticklabels=None,
    markersize=None,
    markersizes=None,
    markersize_order=None,
    markersize_label=None,
    markersize_ticklabels=None,
    markeredgecolor="white",
    markeredgecolors=None,
    markeredgecolor_order=None,
    markeredgecolor_label=None,
    markeredgecolor_ticklabels=None,
    linewidth=None,
    linewidths=None,
    linewidth_order=None,
    linewidth_label=None,
    linewidth_ticklabels=None,
    linestyle=None,
    linestyles=None,
    linestyle_order=None,
    linestyle_label=None,
    linestyle_ticklabels=None,
    text=None,
    text_formatter=str,
    text_opts=None,
    col=None,
    col_order=None,
    col_label=None,
    col_ticklabels=None,
    row=None,
    row_order=None,
    row_label=None,
    row_ticklabels=None,
    alpha=1.0,
    join_across_missing=False,
    err_band_alpha=0.1,
    err_bar_capsize=1,
    xlabel=None,
    ylabel=None,
    xlim=None,
    ylim=None,
    zlim=None,
    xscale=None,
    yscale=None,
    zscale=None,
    xbase=10,
    ybase=10,
    xticks=None,
    yticks=None,
    xticklabels=None,
    yticklabels=None,
    vspans=(),
    hspans=(),
    span_color=(0.5, 0.5, 0.5),
    span_alpha=0.5,
    span_linewidth=1,
    span_linestyle=":",
    grid=True,
    grid_which="major",
    grid_alpha=0.1,
    legend=True,
    legend_ncol=None,
    legend_merge="auto",
    legend_reverse=False,
    legend_entries=None,
    legend_labels=None,
    legend_extras=None,
    legend_opts=None,
    label=None,
    title=None,
    ax=None,
    axs=None,
    format_axs=None,
    figsize=None,
    background_color=None,
    height=3,
    width=None,
    hspace=0.12,
    wspace=0.12,
    sharex=True,
    sharey=True,
    kwargs=None,
)

_PLOTTER_OPTS = tuple(INFINIPLOTTER_DEFAULTS.keys())


_HEATMAP_INVALID_KWARGS = {
    "hue",
    "color",
    "marker",
    "markersize",
    "markeredgecolor",
    "linewidth",
    "linestyle",
    "text",
}


class Infiniplotter:
    def __init__(
        self,
        ds,
        x,
        y=None,
        z=None,
        **kwargs,
    ):
        import numpy as np
        from matplotlib import pyplot as plt

        self.x = x
        self.y = y
        self.z = z

        self.is_histogram = self.y is None
        self.is_heatmap = self.z is not None
        self.is_scatter = (not self.is_histogram) and (self.x in ds.data_vars)

        settings = {**INFINIPLOTTER_DEFAULTS, **kwargs}
        for opt in _PLOTTER_OPTS:
            setattr(self, opt, settings.pop(opt))

        # Parse unmatched keywords and error but suggest correct versions
        if len(settings) > 0:
            import difflib

            wrong_opts = list(settings.keys())
            right_opts = [
                difflib.get_close_matches(opt, _PLOTTER_OPTS, n=3)
                for opt in wrong_opts
            ]

            msg = "Option(s) {} not valid.\n Did you mean: {}?".format(
                wrong_opts, right_opts
            )
            print(msg)
            raise ValueError(msg)

        # options to simply pass on to the matplotlib calls
        # use an explicit dict so we can autocorrect other options
        self.kwargs = self.kwargs or {}

        self.autohue_opts = (
            {} if self.autohue_opts is None else dict(self.autohue_opts)
        )

        if self.text is not None:
            self.text_opts = (
                {} if self.text_opts is None else dict(self.text_opts)
            )
            self.text_opts.setdefault("size", 6)
            self.text_opts.setdefault("horizontalalignment", "left")
            self.text_opts.setdefault("verticalalignment", "bottom")
            self.text_opts.setdefault("clip_on", True)

        self.err_kws = {} if self.err_kws is None else dict(self.err_kws)

        # if only one is specified allow it to be either
        if (self.hue is not None) and (self.color is None):
            (self.color, self.color_order, self.colors, self.color_label) = (
                self.hue,
                self.hue_order,
                self.hues,
                self.hue_label,
            )
            self.hue = self.hue_order = self.hues = self.hue_label = None

        # default style options
        self.base_style = {
            "alpha": self.alpha,
            "markersize": 6,
            "color": "#0ca0eb",
            "marker": ".",
            "markeredgecolor": "white",
        }
        # the size of each mapped dimension
        self.sizes = {}
        # the domain (i.e. input) of each mapped dimension
        self.domains = {}
        # the range (i.e. output) of each mappend dimension
        self.values = {}
        # how to label each mapped dimension
        self.labels = {
            self.x: self.x if self.xlabel is None else self.xlabel,
            self.y: self.y if self.ylabel is None else self.ylabel,
        }
        # how to name each point within each mapped dimension
        self.ticklabels = {}

        # work out all the dim mapping information

        def default_colormaps(N):
            if N <= 6:
                hs = get_default_colormaps(N)
            else:
                hs = np.linspace(
                    self.autohue_start,
                    self.autohue_start + self.autohue_sweep,
                    N,
                    endpoint=False,
                )

            self.autohue_opts.setdefault("hue_shift", 0.5 / N)
            return [to_colormap(h, **self.autohue_opts) for h in hs]

        # drop irrelevant variables and dimensions
        ds = ds.drop_vars(
            [
                k
                for k in ds
                if k
                not in (
                    self.x,
                    self.y,
                    self.z,
                    self.err,
                    self.hspans,
                    self.vspans,
                )
            ]
        )

        possible_dims = set()
        if self.x in ds.data_vars:
            possible_dims.update(ds[self.x].dims)
        if self.y in ds.data_vars:
            possible_dims.update(ds[self.y].dims)
        if self.z in ds.data_vars:
            possible_dims.update(ds[self.z].dims)
        self.ds = ds.drop_dims([k for k in ds.dims if k not in possible_dims])

        self.mapped = set()

        self.init_mapped_dim(
            "hue",
            custom_values=(
                [to_colormap(h, **self.autohue_opts) for h in self.hues]
                if self.hues is not None
                else None
            ),
            default_values=default_colormaps,
        )
        self.init_mapped_dim(
            "color",
            custom_values=self.colors,
            default_values=lambda N: np.linspace(
                self.colormap_start, self.colormap_stop, N
            ),
        )

        if (self.hue is not None) and (self.color is not None):
            # need special label
            dim = f"{self.hue}, {self.color}"
            self.labels[dim] = (
                f"{self.labels[self.hue]}, {self.labels[self.color]}"
            )
            self.ticklabels[dim] = {}

        if (self.hue is None) and (self.color is not None):
            # set a global colormap or sequence
            if self.colors is None:
                self.cmap_or_colors = (
                    to_colormap(self.palette, **self.autohue_opts)
                    if self.palette is not None
                    else auto_colors(self.sizes["color"])
                )
            else:
                self.cmap_or_colors = self.values["color"]

        self.init_mapped_dim(
            "marker",
            custom_values=self.markers,
            default_values=itertools.cycle(_MARKERS_DEFAULT),
        )
        self.init_mapped_dim(
            "markersize",
            custom_values=self.markersizes,
            default_values=lambda N: np.linspace(3.0, 9.0, N),
        )
        self.init_mapped_dim(
            "markeredgecolor",
            custom_values=self.markeredgecolors,
            default_values=lambda N: auto_colors(N),
        )

        if self.is_scatter and (self.xlink is None):
            if self.linewidth is not None:
                warnings.warn(
                    "`linewidth` is not supported for scatters, ignoring. "
                    "Set `x` as a coordinate or use `xlink` for a line plot."
                )
            self.linewidth = 0
            if self.linestyle is not None:
                warnings.warn(
                    "`linestyle` is not supported for scatters, ignoring."
                    "Set `x` as a coordinate or use `xlink` for a line plot."
                )
            self.linestyle = ""

        self.init_mapped_dim(
            "linestyle",
            custom_values=self.linestyles,
            default_values=itertools.cycle(_LINESTYLES_DEFAULT),
        )
        self.init_mapped_dim(
            "linewidth",
            custom_values=self.linewidths,
            default_values=lambda N: np.linspace(1.0, 3.0, N),
        )
        self.init_mapped_dim("col")
        self.init_mapped_dim("row")

        # compute which dimensions are not target or mapped dimensions
        self.unmapped = sorted(
            set(self.ds.dims)
            - self.mapped
            - {self.x, self.y, self.z, self.xlink}
        )

        if self.is_scatter and (self.xlink is None):
            # want to vectorize plotting over all unmapped dimensions
            self.ds = self.ds.stack({"__unmapped__": self.unmapped})

        if self.is_histogram:
            # histogram: create y as probability density / counts
            import xarray as xr

            # bin over all unmapped dimensions
            self.ds = self.ds.stack({"__unmapped__": self.unmapped})

            # work out the bin coordinates
            if self.bins is None or isinstance(self.bins, int):
                if self.bins is None:
                    nbins = min(
                        max(3, int(self.ds["__unmapped__"].size ** 0.5)), 50
                    )
                else:
                    nbins = self.bins
                xmin, xmax = self.ds[self.x].min(), self.ds[self.x].max()
                self.bins = np.linspace(float(xmin), float(xmax), nbins + 1)
            elif not isinstance(self.bins, np.ndarray):
                self.bins = np.asarray(self.bins)

            bin_coords = (self.bins[1:] + self.bins[:-1]) / 2

            if self.bins_density:
                self.y = f"prob({self.x})"
            else:
                self.y = f"count({self.x})"

            if self.ylabel is None:
                self.labels[self.y] = self.y
            else:
                self.labels[self.y] = self.ylabel

            ds_binned = (
                xr.apply_ufunc(
                    lambda x: np.histogram(
                        x, bins=self.bins, density=self.bins_density
                    )[0],
                    self.ds[self.x],
                    input_core_dims=[["__unmapped__"]],
                    output_core_dims=[[self.x]],
                    vectorize=True,
                )
                .to_dataset(name=self.y)
                .assign_coords({self.x: bin_coords})
            )

            if isinstance(self.hspans, str):
                ds_binned[self.hspans] = self.ds[self.hspans]
            if isinstance(self.vspans, str):
                ds_binned[self.vspans] = self.ds[self.vspans]

            self.ds = ds_binned
            self.kwargs.setdefault("drawstyle", "steps-mid")

        if self.is_heatmap and self.unmapped:
            if self.aggregate is not True:
                # default to aggregating over all unmapped dimensions, but warn
                warnings.warn(
                    "Heatmap: aggregating over all unmapped dimensions: "
                    f"{self.unmapped}. Set `aggregate=True` to acknowledge "
                    "and disable this warning, or map the dimension(s) to "
                    "`row` or `col`."
                )
                self.aggregate = True

        # get the target data array and possibly aggregate some dimensions
        if self.aggregate:
            agg_into = self.z or self.y

            if self.aggregate is True:
                # select all unmapped dimensions
                self.aggregate = self.unmapped

            # compute data ranges to maybe show spread bars or bands
            if self.aggregate_err_range == "std":
                da_std_mean = self.ds[agg_into].mean(self.aggregate)
                da_std = self.ds[agg_into].std(self.aggregate)

                self.da_ql = da_std_mean - da_std
                self.da_qu = da_std_mean + da_std

            elif self.aggregate_err_range == "stderr":
                da_stderr_mean = self.ds[agg_into].mean(self.aggregate)
                da_stderr_cnt = self.ds[agg_into].notnull().sum(self.aggregate)
                da_stderr = self.ds[agg_into].std(self.aggregate) / np.sqrt(
                    da_stderr_cnt
                )

                self.da_ql = da_stderr_mean - da_stderr
                self.da_qu = da_stderr_mean + da_stderr

            else:
                self.aggregate_err_range = min(
                    max(0.0, self.aggregate_err_range), 1.0
                )
                ql = 0.5 - self.aggregate_err_range / 2.0
                qu = 0.5 + self.aggregate_err_range / 2.0
                self.da_ql = self.ds[agg_into].quantile(ql, self.aggregate)
                self.da_qu = self.ds[agg_into].quantile(qu, self.aggregate)

            # default to showing spread as bands
            if self.err is None:
                self.err = True
            if self.err_style is None:
                self.err_style = "band"

            # main data for central line
            self.ds = getattr(self.ds, self.aggregate_method)(self.aggregate)

        # default to bars if err not taken from aggregating
        if self.err_style is None:
            self.err_style = "bars"

        # all the coordinates we will iterate over
        self.remaining_dims = []
        self.remaining_sizes = []
        for dim, sz in self.ds.sizes.items():
            if dim not in (self.x, self.y, self.z, self.xlink, "__unmapped__"):
                self.remaining_dims.append(dim)
                self.remaining_sizes.append(sz)
        self.ranges = list(map(range, self.remaining_sizes))

        # maybe create the figure and axes
        if self.ax is not None:
            if self.axs is not None:
                raise ValueError("cannot specify both `ax` and `axs`")
            self.axs = np.array([[self.ax]])

        if self.axs is None:
            if self.figsize is None:
                if self.width is None:
                    self.width = self.height
                if self.height is None:
                    self.height = self.width
                self.figsize = (
                    self.width * self.sizes["col"],
                    self.height * self.sizes["row"],
                )

            self.fig, self.axs = plt.subplots(
                self.sizes["row"],
                self.sizes["col"],
                sharex=self.sharex,
                sharey=self.sharey,
                squeeze=False,
                gridspec_kw={"hspace": self.hspace, "wspace": self.wspace},
                figsize=self.figsize,
            )
            if self.background_color is not None:
                self.fig.patch.set_facecolor(self.background_color)
            else:
                self.fig.patch.set_alpha(0.0)
        else:
            self.fig = None

        if (self.fig is not None) and (self.title is not None):
            self.fig.suptitle(self.title)

    def init_mapped_dim(
        self,
        name,
        custom_values=None,
        default_values=None,
    ):
        dim = getattr(self, name)

        if isinstance(dim, list):
            # make hashable
            dim = tuple(dim)

        # handle fused dimensions
        if isinstance(dim, tuple):
            # fused named
            new_dim = ", ".join(dim)
            if new_dim in self.ds.dims:
                # already fused -> nothing to do
                dim = new_dim
            elif all(x in self.ds.dims for x in dim):
                # create a new fused dimension
                self.ds = self.ds.stack({new_dim: dim})
                dim = new_dim
            # else not a valid fused dimensions -> assume constant property

        if (dim is not None) and (dim not in self.ds.dims):
            # attribute is just manually specified, not mapped to dimension
            self.base_style[name] = dim
            self.sizes[name] = 1
            setattr(self, name, None)
            return

        order = getattr(self, f"{name}_order")
        dim_label = getattr(self, f"{name}_label")
        dim_ticklabels = getattr(self, f"{name}_ticklabels")

        if dim is not None:
            if self.is_heatmap and (name in _HEATMAP_INVALID_KWARGS):
                raise ValueError(f"Heatmap: cannot map property `{name}`.")

            if order is not None:
                # select and order along dimension
                self.ds = self.ds.sel({dim: list(order)})

            self.mapped.add(dim)
            self.ds = self.ds.dropna(dim, how="all")

            self.domains[name] = self.ds[dim].values
            self.sizes[name] = len(self.domains[name])
            self.labels[dim] = (
                _make_bold(dim) if dim_label is None else dim_label
            )

            if dim_ticklabels is None:
                dim_ticklabels = {}
            if not isinstance(dim_ticklabels, dict):
                dim_ticklabels = dict(zip(self.domains[name], dim_ticklabels))
            self.ticklabels[dim] = dim_ticklabels

            if custom_values is None:
                if default_values is not None:
                    if callable(default_values):
                        # allow default values to depend on number of values
                        default_values = default_values(self.sizes[name])

                    self.values[name] = tuple(
                        x
                        for x, _ in zip(
                            default_values, range(self.sizes[name])
                        )
                    )
            else:
                self.values[name] = custom_values
        else:
            self.sizes[name] = 1

        setattr(self, name, dim)

    def plot_lines(self):
        """Plot lines of the data."""
        import matplotlib as mpl

        # iterate over and plot all data
        self.handles = {}
        self.split_handles = collections.defaultdict(
            lambda: collections.defaultdict(dict)
        )

        x_is_constant = self.x not in self.ds.data_vars
        if x_is_constant:
            # is a constant coordinate
            xdata = self.ds[self.x].values

        for iloc in itertools.product(*self.ranges):
            # current coordinates
            loc = dict(zip(self.remaining_dims, iloc))

            # get the right set of axes to plot on
            if self.row is not None:
                i_ax = loc[self.row]
            else:
                i_ax = 0
            if self.col is not None:
                j_ax = loc[self.col]
            else:
                j_ax = 0
            ax = self.axs[i_ax, j_ax]

            # map coordinate into relevant styles and keep track of each uniquely
            sub_key = {}
            specific_style = {}

            # need to handle hue and color separately
            if self.color is not None:
                if self.hue is not None:
                    ihue = loc[self.hue]
                    hue_in = self.domains["hue"][ihue]
                    sub_key[self.hue] = hue_in
                    self.cmap_or_colors = self.values["hue"][ihue]

                icolor = loc[self.color]
                color_in = self.domains["color"][icolor]
                if not callable(self.cmap_or_colors):
                    color_out = self.cmap_or_colors[icolor]
                else:
                    color_out = self.cmap_or_colors(
                        self.values["color"][icolor]
                    )

                sub_key[self.color] = color_in
                specific_style["color"] = color_out
                if self.hue is None:
                    legend_dim = self.color
                    legend_in = color_in
                else:
                    legend_dim = ", ".join((self.hue, self.color))
                    legend_in = ", ".join(map(str, (hue_in, color_in)))

                self.split_handles[legend_dim][legend_in]["color"] = color_out
            else:
                legend_dim = None

            for prop in (
                "marker",
                "markersize",
                "markeredgecolor",
                "linewidth",
                "linestyle",
            ):
                dim = getattr(self, prop)
                if dim is not None:
                    idx = loc[dim]
                    prop_in = self.domains[prop][idx]
                    prop_out = self.values[prop][idx]
                    sub_key[dim] = prop_in
                    specific_style[prop] = prop_out

                    if dim in (self.color, self.hue):
                        self.split_handles[legend_dim][legend_in][prop] = (
                            prop_out
                        )
                    else:
                        self.split_handles[dim][prop_in][prop] = prop_out

            # get the masked x and y data
            ds_loc = self.ds.isel(loc)
            mask = ds_loc[self.y].notnull().values

            if not x_is_constant:
                # x also varying
                xdata = ds_loc[self.x].values
                # both x and y must be non-null
                mask &= ds_loc[self.x].notnull().values

            if not np.any(mask):
                # don't plot all null lines
                continue

            if not self.join_across_missing:
                # reset mask
                data_mask = ()
            else:
                data_mask = mask

            xmdata = xdata[data_mask]
            ymdata = ds_loc[self.y].values[data_mask]

            if self.err is not None:
                if (self.err is True) and (self.aggregate is not None):
                    da_ql_loc = self.da_ql.isel(loc)
                    da_qu_loc = self.da_qu.isel(loc)
                    y1 = da_ql_loc.values[data_mask]
                    y2 = da_qu_loc.values[data_mask]
                    yneg = ymdata - y1
                    ypos = y2 - ymdata
                else:
                    yerr_mdata = ds_loc[self.err].values[data_mask]
                    yneg = -yerr_mdata
                    ypos = +yerr_mdata
                    y1 = ymdata + yneg
                    y2 = ymdata + ypos

                if self.err_style == "bars":
                    ax.errorbar(
                        x=xmdata,
                        y=ymdata,
                        yerr=[np.abs(yneg), np.abs(ypos)],
                        fmt="none",
                        capsize=self.err_bar_capsize,
                        **{
                            **self.base_style,
                            **specific_style,
                            **self.err_kws,
                        },
                    )
                elif self.err_style == "band":
                    ax.fill_between(
                        x=xmdata,
                        y1=y1,
                        y2=y2,
                        color=specific_style.get(
                            "color", self.base_style["color"]
                        ),
                        alpha=self.err_band_alpha,
                        **self.err_kws,
                    )

            if self.is_histogram:
                ax.fill_between(
                    x=xmdata,
                    y1=ymdata,
                    y2=0,
                    step={
                        None: None,
                        "default": None,
                        "steps": "pre",
                        "steps-pre": "pre",
                        "steps-mid": "mid",
                        "steps-post": "post",
                    }[self.kwargs.get("drawstyle", None)],
                    color=mpl.colors.to_rgb(
                        specific_style.get("color", self.base_style["color"])
                    ),
                    alpha=self.err_band_alpha,
                )

            plot_opts = {**self.base_style, **specific_style}

            if self.label:
                # override with manual label
                label = self.label
            else:
                label = ", ".join(map(str, sub_key.values()))

            # do the plotting!
            (handle,) = ax.plot(
                xmdata,
                ymdata,
                label=label,
                **plot_opts,
                **self.kwargs,
            )

            # add a text label next to each point
            if self.text is not None:
                # need raw mask for text
                smdata = ds_loc[self.text].values[mask]
                for txx, txy, txs in zip(
                    xdata[mask], ds_loc[self.y].values[mask], smdata
                ):
                    specific_text_opts = {}
                    if "color" not in self.text_opts:
                        # default to line color
                        specific_text_opts["color"] = plot_opts["color"]

                    ax.text(
                        txx,
                        txy,
                        self.text_formatter(txs),
                        **self.text_opts,
                        **specific_text_opts,
                    )

            # only want one legend entry per unique style
            key = frozenset(sub_key.items())
            if key or self.label:
                self.handles.setdefault(key, handle)

            # add spans that depend on location
            for _spans, ax_func in (
                (self.hspans, ax.axhline),
                (self.vspans, ax.axvline),
            ):
                if isinstance(_spans, str):
                    for spn in ds_loc[_spans].values.flat:
                        ax_func(
                            spn,
                            color=self.span_color,
                            alpha=self.span_alpha,
                            linestyle=self.span_linestyle,
                            linewidth=self.span_linewidth,
                        )

        self.do_axes_formatting()

        if self.legend:
            self.create_legend()

    def create_legend(self):
        """Create a legend for lines plot."""
        from matplotlib.lines import Line2D

        legend_opts = {} if self.legend_opts is None else self.legend_opts

        if ("bbox_to_anchor" in legend_opts) or ("loc" in legend_opts):
            # we are manually placing legend, assume its a new one, rather
            # than trying to extend
            try:
                legend_old = self.axs[0, -1].get_legend()
                # later call to legend will overwrite this one: explicitly add
                self.axs[0, -1].add_artist(legend_old)
            except AttributeError:
                pass
            legend_handles = []

        else:
            # try to extend current legend with more entries
            try:
                legend_handles = self.axs[0, -1].get_legend().get_lines()
                # add a space
                legend_handles.append(
                    Line2D([0], [0], markersize=0, linewidth=0, label="")
                )
            except AttributeError:
                legend_handles = []

        if self.legend_merge == "auto":
            self.legend_merge = len(self.mapped) <= 1

        if self.handles and self.legend_merge:
            # show every unique style combination as single legend try

            if self.legend_entries:
                # only keep manually specified legend entries
                remove = set()
                for k in self.handles:
                    for dim, val in k:
                        if dim in self.legend_entries:
                            if val not in self.legend_entries[dim]:
                                remove.add(k)
                for k in remove:
                    del self.handles[k]

            sorters = []
            legend_title = []
            for prop in [
                "hue",
                "color",
                "marker",
                "markersize",
                "markeredgecolor",
                "linewidth",
                "linestyle",
            ]:
                dim = getattr(self, prop)
                dim_order = getattr(self, f"{prop}_order")
                if dim is not None and self.labels[dim] not in legend_title:
                    # check if not in legend_title, as multiple attributes can
                    # be mapped to the same dimension
                    legend_title.append(self.labels[dim])

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
                v
                for _, v in sorted(
                    self.handles.items(),
                    key=legend_sort,
                    reverse=self.legend_reverse,
                )
            )

            if self.legend_ncol is None:
                if self.sizes["color"] == 1 or len(self.handles) <= 10:
                    self.legend_ncol = 1
                else:
                    self.legend_ncol = self.sizes["hue"]

            legend_opts.setdefault("title", ", ".join(legend_title))
            legend_opts.setdefault("ncol", self.legend_ncol)

        elif self.split_handles:
            # separate legend for each style

            if self.legend_entries:
                # only keep manually specified legend entries
                for k, vals in self.legend_entries.items():
                    self.split_handles[k] = {
                        key: val
                        for key, val in self.split_handles[k].items()
                        if key in vals
                    }

            self.base_style["color"] = (0.5, 0.5, 0.5)
            self.base_style["marker"] = ""
            self.base_style["linestyle"] = ""

            ncol = len(self.split_handles)
            nrow = max(map(len, self.split_handles.values()))

            for legend_dim, inputs in self.split_handles.items():
                # add sub title in legend
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        markersize=0,
                        linewidth=0,
                        label=self.labels[legend_dim],
                    )
                )

                key_styles = list(inputs.items())
                if self.legend_reverse:
                    key_styles.reverse()

                for key, style in key_styles:
                    label = self.ticklabels[legend_dim].get(key, str(key))

                    if any("marker" in prop for prop in style):
                        style.setdefault("marker", "o")
                    if any("line" in prop for prop in style):
                        style.setdefault("linestyle", "-")
                    if "color" in style:
                        style.setdefault("marker", ".")
                        style.setdefault("linestyle", "-")

                    legend_handles.append(
                        Line2D(
                            [0],
                            [0],
                            **{**self.base_style, **style},
                            label=label,
                        )
                    )

                if self.legend_ncol is None:
                    npad = nrow - len(inputs)
                else:
                    npad = 1
                for _ in range(npad):
                    # spacing
                    legend_handles.append(
                        Line2D([0], [0], markersize=0, linewidth=0, label="")
                    )

            if self.legend_ncol is None:
                legend_opts.setdefault("ncol", ncol)

            if self.legend_extras is not None:
                for extra in self.legend_extras:
                    if isinstance(extra, dict):
                        extra = Line2D([0], [0], **extra)
                    legend_handles.append(extra)
        else:
            legend_handles = None

        if legend_handles is not None:
            # we have some legend entries

            if self.legend_labels is not None:
                # we have explicit labels for each handle
                for lh, label in zip(
                    # only update the handles that were added by this plot call
                    legend_handles[-len(self.handles) :],
                    self.legend_labels,
                ):
                    lh.set_label(label)

            if (
                legend_opts.get("title", "") == "" and len(legend_handles) == 1
            ) and (self.label is None):
                # promote single legend entry to title
                lh0 = legend_handles[0]
                legend_opts["title"] = lh0.get_label()
                lh0.set_label("")

            lax = self.axs[0, -1]
            legend_opts.setdefault("loc", "upper left")
            legend_opts.setdefault("bbox_to_anchor", (1.0, 1.0))
            legend_opts.setdefault("columnspacing", 1.0)
            legend_opts.setdefault("edgecolor", "none")
            legend_opts.setdefault("framealpha", 0.0)
            lax.legend(handles=legend_handles, **legend_opts)

    def plot_heatmap(self):
        """Plot a heatmap of the data."""
        import matplotlib as mpl

        xdata = self.ds[self.x].values
        ydata = self.ds[self.y].values

        zdata_all = self.ds[self.z].values
        zdata_all = zdata_all[np.isfinite(zdata_all)]

        # set coloring limits
        try:
            zmin, zmax = self.zlim
        except (TypeError, ValueError):
            zmin = zmax = None
        if zmax is None:
            zmax = np.max(zdata_all)
        if zmin is None:
            zmin = np.min(zdata_all)
        max_mag = max(abs(zmax), abs(zmin))

        if self.palette is None:
            self.norm = None
        elif self.zscale == "log":
            self.norm = mpl.colors.LogNorm(vmin=zmin, vmax=zmax)
        elif self.zscale == "symlog":
            self.norm = mpl.colors.SymLogNorm(
                vmin=zmin, vmax=zmax, linthresh=1
            )
        else:
            self.norm = mpl.colors.Normalize(vmin=zmin, vmax=zmax)

        for iloc in itertools.product(*self.ranges):
            # current coordinates
            loc = dict(zip(self.remaining_dims, iloc))

            # get the right set of axes to plot on
            if self.row is not None:
                i_ax = loc[self.row]
            else:
                i_ax = 0
            if self.col is not None:
                j_ax = loc[self.col]
            else:
                j_ax = 0
            ax = self.axs[i_ax, j_ax]

            zdata = self.ds[self.z].isel(loc).transpose(self.y, self.x).values

            # get the masked x and y data
            if self.palette is None:
                C = zdata
                mask = np.isfinite(C)
                zdata = np.empty(C.shape + (4,))
                zdata[mask] = to_colors(
                    C[mask], alpha_pow=0.0, max_mag=max_mag
                )[0]
                zdata[~mask] = (0.5, 0.5, 0.5, 0.5)
            else:
                zdata = (
                    self.ds[self.z].isel(loc).transpose(self.y, self.x).values
                )

            ax.pcolormesh(
                xdata,
                ydata,
                zdata,
                shading="nearest",
                # edgecolors="face",
                rasterized=True,
                cmap=self.palette,
                norm=self.norm,
            )

        self.do_axes_formatting()

        if self.legend:
            if self.palette is None:
                add_visualize_legend(
                    ax=self.axs[0, -1],
                    complexobj=np.iscomplexobj(zdata_all),
                    max_mag=max_mag,
                    legend_loc=(1.1, 0.8),
                    legend_size=0.2,
                    # max_projections=max_projections,
                    # legend_bounds=legend_bounds,
                    # legend_resolution=legend_resolution,
                )
            else:
                cax = self.axs[0, -1].inset_axes((1.1, 0.1, 0.05, 0.8))

                self.fig.colorbar(
                    mpl.cm.ScalarMappable(norm=self.norm, cmap=self.palette),
                    cax=cax,
                    orientation="vertical",
                    label=_make_bold(self.z),
                )

    def do_axes_formatting(self):
        if (self.fig is None) and (not self.format_axs):
            return

        # perform axes level formatting
        from matplotlib.ticker import (
            AutoMinorLocator,
            LogLocator,
            NullFormatter,
            ScalarFormatter,
            StrMethodFormatter,
        )

        for (i, j), ax in np.ndenumerate(self.axs):
            # only change this stuff if we created the figure
            title = []
            if self.col is not None:
                title.append(
                    f"{self.labels[self.col]}={self.domains['col'][j]}"
                )
            if self.row is not None:
                title.append(
                    f"{self.labels[self.row]}={self.domains['row'][i]}"
                )
            if title:
                title = ", ".join(title)
                ax.text(
                    0.5,
                    1.0,
                    title,
                    transform=ax.transAxes,
                    horizontalalignment="center",
                    verticalalignment="bottom",
                )

            # only label outermost plot axes
            if i + 1 == self.sizes["row"]:
                ax.set_xlabel(self.labels[self.x])
            if j == 0:
                ax.set_ylabel(self.labels[self.y])

            # set some nice defaults
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)

            if self.grid:
                ax.grid(True, which=self.grid_which, alpha=self.grid_alpha)
                ax.set_axisbelow(True)

            if self.xlim is not None:
                ax.set_xlim(self.xlim)
            if self.ylim is not None:
                ax.set_ylim(self.ylim)

            if self.xscale is not None:
                ax.set_xscale(self.xscale)
            if self.yscale is not None:
                ax.set_yscale(self.yscale)

            for scale, base, ticks, ticklabels, axis in [
                (
                    self.xscale,
                    self.xbase,
                    self.xticks,
                    self.xticklabels,
                    ax.xaxis,
                ),
                (
                    self.yscale,
                    self.ybase,
                    self.yticks,
                    self.yticklabels,
                    ax.yaxis,
                ),
            ]:
                if scale == "log":
                    axis.set_major_locator(LogLocator(base=base, numticks=6))
                    if base != 10:
                        if isinstance(base, int):
                            axis.set_major_formatter(
                                StrMethodFormatter("{x:.0f}")
                            )
                        else:
                            axis.set_major_formatter(ScalarFormatter())
                    if base < 3:
                        subs = [1.5]
                    else:
                        subs = np.arange(2, base)
                    axis.set_minor_locator(LogLocator(base=base, subs=subs))
                    axis.set_minor_formatter(NullFormatter())
                elif scale == "symlog":
                    # TODO: choose some nice defaults
                    pass
                else:
                    axis.set_minor_locator(AutoMinorLocator(5))

                if ticks is not None:
                    axis.set_ticks(ticks, labels=ticklabels)

            for _spans, ax_func, h_or_v in (
                (self.hspans, ax.axhline, "h"),
                (self.vspans, ax.axvline, "v"),
            ):
                if not isinstance(_spans, str):
                    for line in _spans:
                        if isinstance(_spans, dict):
                            import matplotlib.transforms as transforms

                            # label each span using key from dict
                            spanlabel = line
                            # actual line location is value in dict
                            line = _spans[spanlabel]
                            trans = transforms.blended_transform_factory(
                                ax.transAxes, ax.transData
                            )

                            if h_or_v == "h":
                                _span_label_coo = (0.0, line)
                                va = "bottom"
                            else:
                                _span_label_coo = (line, 1.0)
                                va = "top"

                            ax.text(
                                *_span_label_coo,
                                spanlabel,
                                ha="left",
                                va=va,
                                transform=trans,
                                color=self.span_color,
                            )

                        ax_func(
                            line,
                            color=self.span_color,
                            alpha=self.span_alpha,
                            linestyle=self.span_linestyle,
                            linewidth=self.span_linewidth,
                        )


@show_and_close
@use_neutral_style
def infiniplot(
    ds,
    x,
    y=None,
    z=None,
    **kwargs,
):
    """Helper class for the infiniplot functionality.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to plot.
    x : str
        Name of the x coordinate.
    y : str, optional
        Name of the y coordinate. If not specified, histogram mode is activated
        and the values of ``x`` are binned to produce a density or frequency to
        use as the y-variable.
    z : str, optional
        Name of the z coordinate. If specified this turns on the heatmap mode.
    bins : int or array_like, optional
        If in histogram mode, specify either the number of bins to use or the
        bin edges. If not specified, a default number of bins is automatically
        chosen based on the number of data points.
    bins_density : bool, optional
        If in histogram mode, whether to plot the density (True) or frequency
        (False) of the data. Default is True.
    aggregate : str or Sequence[str], optional
        If specified, aggregate over the given dimension(s) using
        ``aggregate_method`` (by default 'median'). If `True` aggregate over
        all unmapped dimensions. If in heatmap mode, this is automatically set
        to `True`, since only one plot can be shown per axis.
    aggregate_method : str, optional
        If ``aggregate`` is specified, the method to use for aggregation. Any
        option available as a method on a DataArray can be used, e.g. 'mean',
        'median', 'max'. Default is 'median'.
    aggregate_err_range : float or str, optional
        If ``aggregate`` is specified, the range of the error bars or bands to
        show. The options are:

        - ``'std'``: show the standard deviation of the data
        - ``'stderr'``: show the standard error of the mean
        - float: show the given quantile range, e.g. 0.5 for the interquartile
            range

    err : str, optional
        If specified, a data variable to use for error bars or bands. This
        overrides any derived from ``aggregate``.
    err_style : str, optional
        If specified, the style of error to show. The options are:

        - ``'bars'``: show error bars
        - ``'band'``: show error bands

    err_kws : dict, optional
        Additional keyword arguments to pass to the error plotting function.
    xlink : str, optional
        If specified, the name of a dimension to use for linking the x-axis.
        Used when you are plotting a variable rather than coordinate as ``x``,
        but want to link each sweep of values as a line.
    color : str, optional
        If specified, the name of a dimension to use for mapping the color or
        intensity of each line. If ``hue`` is also specified, this controls the
        intensity of the color. If not a dimension, this is used as a constant
        color for all lines.
    colors : sequence, optional
        An explicit sequence of colors to use for the color-mapped dimension.
    color_order : sequence, optional
        An explicit order of values to use for the color-mapped dimension.
    color_label : str, optional
        An alternate label to use for the color-mapped dimension.
    color_ticklabels : dict or sequence, optional
        A mapping from values to tick labels to use for the color-mapped
        dimension.
    colormap_start : float, optional
        If using a palette, the starting value of the colormap to use, e.g. 0.2
        would skip the first 20% of the colormap.
    colormap_stop : float, optional
        If using a palette, the stopping value of the colormap to use, e.g. 0.9
        would skip the last 10% of the colormap.
    hue : str, optional
        If specified, the name of a dimension to use for mapping the color or
        hue of each line. If ``color`` is also specified, this controls the hue
        of the color. If not a dimension, this is used as a constant hue for
        all lines.
    hues : sequence, optional
        An explicit sequence of hues to use for the hue-mapped dimension.
    hue_order : sequence, optional
        An explicit order of values to use for the hue-mapped dimension.
    hue_label : str, optional
        An alternate label to use for the hue-mapped dimension.
    hue_ticklabels : dict or sequence, optional
        A mapping from values to tick labels to use for the hue-mapped
        dimension.
    palette : str, sequence, or colormap, optional
        If specified, the name of a colormap, or an actual colormap, to use for
        mapping the color or hue of each line. If both ``color`` and ``hue``
        are specified, you can supply a sequence of palettes here, with ``hue``
        controlling which palette, and ``color`` controlling the intensity
        within the palette.
    autohue_start : float, optional
        If not using a palette, the starting hue to use for automatically
        generating a sequence of hues.
    autohue_sweep : float, optional
        If not using a palette, the sweep of hues to use for automatically
        generating a sequence of hues.
    autohue_opts : dict, optional
        Additional keyword arguments to pass to the automatic hue generator -
        see {func}`xyzpy.color.cmoke`.
    marker : str, optional
        If specified, the name of a dimension to use for mapping the marker
        style of each line. If not a dimension, this is used as a constant
        marker style for all lines.
    markers : sequence, optional
        An explicit sequence of markers to use for the marker-mapped dimension.
    marker_order : sequence, optional
        An explicit order of values to use for the marker-mapped dimension.
    marker_label : str, optional
        An alternate label to use for the marker-mapped dimension.
    marker_ticklabels : dict or sequence, optional
        A mapping from values to tick labels to use for the marker-mapped
        dimension.
    markersize : str, optional
        If specified, the name of a dimension to use for mapping the marker
        size of each line. If not a dimension, this is used as a constant
        marker size for all lines.
    markersizes : sequence, optional
        An explicit sequence of marker sizes to use for the markersize-mapped
        dimension.
    markersize_order : sequence, optional
        An explicit order of values to use for the markersize-mapped dimension.
    markersize_label : str, optional
        An alternate label to use for the markersize-mapped dimension.
    markersize_ticklabels : dict or sequence, optional
        A mapping from values to tick labels to use for the markersize-mapped
        dimension.
    markeredgecolor : str, optional
        If specified, the name of a dimension to use for mapping the marker
        edge color of each line. If not a dimension, this is used as a constant
        marker edge color for all lines.
    markeredgecolors : sequence, optional
        An explicit sequence of marker edge colors to use for the
        markeredgecolor-mapped dimension.
    markeredgecolor_order : sequence, optional
        An explicit order of values to use for the markeredgecolor-mapped
        dimension.
    markeredgecolor_label : str, optional
        An alternate label to use for the markeredgecolor-mapped dimension.
    markeredgecolor_ticklabels : dict or sequence, optional
        A mapping from values to tick labels to use for the
        markeredgecolor-mapped dimension.
    linewidth : str, optional
        If specified, the name of a dimension to use for mapping the line
        width of each line. If not a dimension, this is used as a constant
        line width for all lines.
    linewidths : sequence, optional
        An explicit sequence of line widths to use for the linewidth-mapped
        dimension.
    linewidth_order : sequence, optional
        An explicit order of values to use for the linewidth-mapped dimension.
    linewidth_label : str, optional
        An alternate label to use for the linewidth-mapped dimension.
    linewidth_ticklabels :
        A mapping from values to tick labels to use for the linewidth-mapped
        dimension.
    linestyle : str, optional
        If specified, the name of a dimension to use for mapping the line
        style of each line. If not a dimension, this is used as a constant
        line style for all lines.
    linestyles : sequence, optional
        An explicit sequence of line styles to use for the linestyle-mapped
        dimension.
    linestyle_order : sequence, optional
        An explicit order of values to use for the linestyle-mapped dimension.
    linestyle_label : str, optional
        An alternate label to use for the linestyle-mapped dimension.
    linestyle_ticklabels : dict or sequence, optional
        A mapping from values to tick labels to use for the linestyle-mapped
        dimension.
    text : str, optional
        If specified, the name of a dimension to use for mapping text
        annotations to each line.
    text_formatter : callable, optional
        A function to use to format data entries to text annotations. Default
        is ``str``.
    text_opts : dict, optional
        Additional keyword arguments to pass to the text plotting function.
    col : str, optional
        If specified, the name of a dimension to use for mapping the subplot
        column of each line.
    col_order : sequence, optional
        An explicit order of values to use for the col-mapped dimension.
    col_label : str, optional
        An alternate label to use for the col-mapped dimension.
    col_ticklabels : dict or sequence, optional
        A mapping from values to tick labels to use for the col-mapped
        dimension.
    row : str, optional
        If specified, the name of a dimension to use for mapping the subplot
        row of each line.
    row_order : sequence, optional
        An explicit order of values to use for the row-mapped dimension.
    row_label : str, optional
        An alternate label to use for the row-mapped dimension.
    row_ticklabels : dict or sequence, optional
        A mapping from values to tick labels to use for the row-mapped
        dimension.
    alpha : float, optional
        Global alpha value to use for all lines.
    join_across_missing : bool, optional
        If True, join lines across missing (NaN) data. Default is False.
    err_band_alpha : float, optional
        Alpha value to use for error bands.
    err_bar_capsize : float, optional
        Size of the caps on error bars.
    xlabel : str, optional
        Alternate label to use for the x-axis.
    ylabel : str, optional
        Alternate label to use for the y-axis.
    xlim : tuple, optional
        Limits to use for the x-axis.
    ylim : tuple, optional
        Limits to use for the y-axis.
    xscale : str, optional
        Scale to use for the x-axis, e.g. 'log'.
    yscale : str, optional
        Scale to use for the y-axis, e.g. 'log'.
    zscale : str, optional
        Scale to use for a heatmap color dimension, e.g. 'log'.
    xbase : float, optional
        If ``xscale=='log'``, the log base to use for the x-axis.
    ybase : float, optional
        If ``yscale=='log'``, the log base to use for the y-axis.
    xticks : sequence[float], optional
        Manual sequence of x-values to use for ticks.
    yticks : sequence[float], optional
        Manual sequence of y-values to use for ticks.
    xticklabels : sequence[str], optional
        Manual sequence of x-tick labels to use, requires and should be the
        same length as ``xticks``.
    yticklabels : sequence[str], optional
        Manual sequence of y-tick labels to use, requires and should be the
        same length as ``yticks``.
    vspans : sequence[float], optional
        Sequence of x-values to use for vertical spans.
    hspans : sequence[float], optional
        Sequence of y-values to use for horizontal spans.
    span_color : str or tuple, optional
        Color to use for spans.
    span_alpha : float, optional
        Alpha value to use for spans.
    span_linewidth : float, optional
        Line width to use for spans.
    span_linestyle : str, optional
        Line style to use for spans.
    grid : bool, optional
        Whether to show grid lines.
    grid_which : str, optional
        Which grid lines to show, either 'major' or 'minor'.
    grid_alpha : float, optional
        Alpha value to use for grid lines.
    legend : bool, optional
        Whether to show a legend.
    legend_ncol : int, optional
        Number of columns to use for the legend.
    legend_merge : bool, optional
        If ``True``, combinations of different mapped properties are merged
        into list of every combination.
    legend_reverse : bool, optional
        If ``True``, reverse the order of the legend entries.
    legend_entries : sequence, optional
        An explicit sequence of legend entries to use.
    legend_labels : sequence, optional
        An explicit sequence of legend labels to use.
    legend_extras : sequence, optional
        An explicit sequence of extra legend items to add.
    legend_opts : dict, optional
        Additional keyword arguments to pass to the legend plotting function.
    title : str, optional
        A title to use for the plot.
    axs : sequence[sequence[matplotlib.Axes]], optional
        An explicit array of axes to use for the plot, it should have at least
        as many rows and columns as there are mapped dimensions.
    ax : matplotlib.Axes, optional
        Shortcut for supplying a single axes to use for the plot, can only
        supply if there is a single row and column.
    format_axs : bool, optional
        Whether to format the axes to use the neutral xyzpy style.
    figsize : tuple, optional
        Size of the figure to use if creating one (ax is axs is None). If not
        specified it is automatically computed based on the number of rows and
        columns.
    height : float, optional
        Height of each subplot. Default is 3.
    width : float, optional
        Width of each subplot. If not specified, it is automatically set to
        match ``height``. Default is None.
    hspace : float, optional
        Spacing between subplots vertically. Default is 0.12.
    wspace : float, optional
        Spacing between subplots horizontally. Default is 0.12.
    sharex : bool, optional
        Whether to share the x-axis between subplots. Default is True.
    sharey : bool, optional
        Whether to share the y-axis between subplots. Default is True.
    kwargs : dict, optional
        Additional keyword arguments to pass to the main plotting function.

    Returns
    -------
    fig : matplotlib.Figure
        Figure containing the plot (None if ``ax`` or ``axs`` is specified).
    axs : sequence[sequence[matplotlib.Axes]]
        Array of axes containing the plot.
    """
    p = Infiniplotter(ds, x, y, z, **kwargs)
    if z is not None:
        p.plot_heatmap()
    else:
        p.plot_lines()
    return p.fig, p.axs
