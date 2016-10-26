"""
Functions for plotting datasets nicely.
"""
# TODO: unify all options with lineplot                                       #
# TODO: names                                                                 #

import functools
import itertools
from ..manage import auto_xyz_ds
from .core import _process_plot_range, _prepare_data_and_styles


@functools.lru_cache(1)
def init_plotly_nb():
    """Cache this so it doesn't happen over and over again.
    """
    from plotly.offline import init_notebook_mode
    init_notebook_mode()


def ishow(figs, nb=True, **kwargs):
    """Show multiple plotly figures in notebook or on web.
    """
    if isinstance(figs, (list, tuple)):
        fig_main = figs[0]
        for fig in figs[1:]:
            fig_main["data"] += fig["data"]
    else:
        fig_main = figs
    if nb:
        from plotly.offline import iplot as plot
        init_plotly_nb()
    else:
        from plotly.plotly import plot
    plot(fig_main, **kwargs)


# --------------------------------------------------------------------------  #
# Line Plots                                                                  #
# --------------------------------------------------------------------------  #

def ilineplot(ds, y_coo, x_coo, z_coo=None,
              figsize=(6, 5),          # absolute figure size
              nb=True,
              title=None,
              # Line coloring options
              colors=False,
              colormap="xyz",
              colormap_log=False,
              colormap_reverse=False,
              # Legend options
              legend=None,
              legend_ncol=None,       # XXX: unused
              ztitle=None,            # XXX: unused
              zlabels=None,           # legend labels
              # x-axis options
              xtitle=None,
              xlims=None,
              xticks=None,
              xlog=False,
              # y-axis options
              ytitle=None,
              ylims=None,
              yticks=None,
              ylog=False,
              # Line markers, styles and positions
              markers=None,
              # TODO linewidths
              # TODO linestyles
              # Misc options
              padding=None,
              vlines=None,
              hlines=None,
              gridlines=True,
              font='Source Sans Pro',
              fontsize_title=20,
              fontsize_ticks=16,
              fontsize_xtitle=20,
              fontsize_ytitle=20,
              fontsize_ztitle=20,     # XXX: unused by plotly
              fontsize_legend=18,
              return_fig=False,
              **kwargs):
    """Take a dataset and plot onse of its variables as a function of two
    coordinates using plotly.
    """
    # TODO: list of colors, send to calc_colors
    # TODO: mouse scroll zoom

    from plotly.graph_objs import Scatter, Margin

    z_vals, cols, zlabels, gen_xy = _prepare_data_and_styles(
        ds=ds, y_coo=y_coo, x_coo=x_coo, z_coo=z_coo, zlabels=zlabels,
        colors=colors, colormap=colormap, colormap_log=colormap_log,
        colormap_reverse=colormap_reverse, engine='PLOTLY')

    # Decide on using markers, and set custom markers and line-styles
    if markers is None:
        markers = len(ds[x_coo]) <= 51
    if markers and len(z_vals) > 1:
        mrkrs = itertools.cycle(range(44))
    else:
        mrkrs = itertools.repeat(None)

    # Strip out probable latex signs "$"
    title, xtitle, ytitle, ztitle = (
        (s.translate({ord("$"): ""}) if isinstance(s, str) else s)
        for s in (title, xtitle, ytitle, ztitle))

    def gen_traces():
        for x, y in gen_xy():
            col = next(cols)

            yield Scatter({
                'x': x,
                'y': y,
                'name': next(zlabels),
                'mode': 'lines+markers' if markers else 'lines',
                'marker': {
                    'color': col,
                    'symbol': next(mrkrs),
                    'line': {
                        'color': col,
                    },
                },
            })

    traces = list(gen_traces())

    if vlines is None:
        vlines = []
    if hlines is None:
        hlines = []
    lines = ([{'type': 'line',
               'layer': 'below',
               'line': {'color': 'rgb(128, 128, 128)',
                        'width': 1.0,
                        'dash': 'dot'},
               'xref': 'x', 'x0': lx, 'x1': lx,
               'yref': 'paper', 'y0': 0, 'y1': 1} for lx in vlines] +
             [{'type': 'line',
               'layer': 'below',
               'line': {'color': 'rgb(128, 128, 128)',
                        'width': 1.0,
                        'dash': 'dot'},
               'yref': 'y', 'y0': ly, 'y1': ly,
               'xref': 'paper', 'x0': 0, 'x1': 1} for ly in hlines])

    auto_no_legend = (legend is False or len(z_vals) > 10 or len(z_vals) == 1)
    xlims, ylims = _process_plot_range(xlims, ylims, ds, x_coo, y_coo, padding)

    layout = {
        'title': title,
        'titlefont': {
            'size': fontsize_title,
        },
        'width': figsize[0] * 100 + 100,
        'height': figsize[1] * 100,
        'margin': Margin(autoexpand=True, l=60, r=80, b=50, t=30, pad=0),
        'xaxis': {
            'showline': True,
            'showgrid': gridlines,
            'title': x_coo if xtitle is None else xtitle,
            'mirror': 'ticks',
            'ticks': 'outside',
            'tickvals': xticks if xticks is not None else None,
            'range': xlims,
            'type': 'log' if xlog else 'linear',
            'tickfont': {
                'size': fontsize_ticks,
            },
            'titlefont': {
                'size': fontsize_xtitle,
            },
        },
        'yaxis': {
            'showline': True,
            'showgrid': gridlines,
            'title': y_coo if ytitle is None else ytitle,
            'range': ylims,
            'mirror': 'ticks',
            'ticks': 'outside',
            'tickvals': yticks if yticks is not None else None,
            'type': 'log' if ylog else 'linear',
            'tickfont': {
                'size': fontsize_ticks,
            },
            'titlefont': {
                'size': fontsize_ytitle,
            },
        },
        'showlegend': legend or not auto_no_legend,
        'legend': {
            'font': {
                'size': fontsize_legend,
            },
        },
        'shapes': lines,
        'font': {
            'family': font,
        },
    }

    fig = {'data': traces, 'layout': layout}
    if return_fig:
        return fig
    ishow(fig, nb=nb, **kwargs)


def xyz_ilineplot(x, y_z, **ilineplot_opts):
    """ Take some x-coordinates and an array, convert them to a Dataset
    treating as multiple lines, then send to ilineplot. """
    ds = auto_xyz_ds(x, y_z)
    return ilineplot(ds, 'y', 'x', 'z', **ilineplot_opts)


# --------------------------------------------------------------------------- #
# Other types of plot                                                         #
# --------------------------------------------------------------------------- #

def iheatmap(ds, data_name, x_coo, y_coo,
             colormap="Portland",
             go_dict={}, ly_dict={},
             nb=True,
             return_fig=False,
             **kwargs):
    """Automatic 2D-Heatmap plot using plotly.
    """
    # TODO: automatic aspect ratio? aspect_ratio='AUTO'
    from plotly.graph_objs import Heatmap
    traces = [Heatmap({"z": (ds[data_name]
                             .dropna(x_coo, how="all")
                             .dropna(y_coo, how="all")
                             .squeeze()
                             .transpose(y_coo, x_coo)
                             .data),
                       "x": ds.coords[x_coo].data,
                       "y": ds.coords[y_coo].data,
                       "colorscale": colormap,
                       "colorbar": {"title": data_name},
                       **go_dict})]
    layout = {"height": 600,
              "width": 650,
              "xaxis": {"showline": True,
                        "mirror": "ticks",
                        "ticks": "outside",
                        "title": x_coo},
              "yaxis": {"showline": True,
                        "mirror": "ticks",
                        "ticks": "outside",
                        "title": y_coo},
              **ly_dict}
    fig = {"data": traces, "layout": layout}
    if return_fig:
        return fig
    ishow(fig, nb=nb, **kwargs)


def iscatter(x, y, cols=None, xlog=False, ylog=False, nb=True,
             return_fig=False, ly_dict={}, **kwargs):
    from plotly.graph_objs import Scatter, Marker
    mkr = Marker({"color": cols, "opacity": 0.9,
                  "colorscale": "Portland", "showscale": cols is not None})
    traces = [Scatter({"x": x, "y": y, "mode": "markers", "marker": mkr})]
    layout = {"width": 700, "height": 700, "showlegend": False,
              "xaxis": {"showline": True, "mirror": "ticks",
                        "ticks": "outside",
                        "type": "log" if xlog else "linear"},
              "yaxis": {"showline": True, "mirror": "ticks",
                        "ticks": "outside",
                        "type": "log" if ylog else "linear"}, **ly_dict}
    fig = {"data": traces, "layout": layout}
    if return_fig:
        return fig
    ishow(fig, nb=nb, **kwargs)


def ihist(xs, nb=True, go_dict={}, ly_dict={}, return_fig=False,
          **kwargs):
    """ Histogram plot with plotly. """
    from plotly.graph_objs import Histogram
    traces = [Histogram({"x": x, **go_dict}) for x in xs]
    layout = {"width": 750, "height": 600,
              "xaxis": {"showline": True, "mirror": "ticks",
                        "ticks": "outside"},
              "yaxis": {"showline": True, "mirror": "ticks",
                        "ticks": "outside"}, **ly_dict}
    fig = {"data": traces, "layout": layout}
    if return_fig:
        return fig
    ishow(fig, nb=nb, **kwargs)


def visualize_matrix(a, colormap="Greys", nb=True, return_fig=False, **kwargs):
    from plotly.graph_objs import Heatmap, Margin
    m, n = a.shape
    traces = [Heatmap({"z": -abs(a),
                       "colorscale": colormap,
                       "showscale": False})]
    layout = {"height": 500, "width": 500,
              'margin': Margin(autoexpand=True, l=30, r=30,
                               b=30, t=30, pad=0),
              "xaxis": {"range": (-1/2, m - 1/2),
                        "zeroline": False,
                        "showline": False,
                        "autotick": True,
                        "ticks": "",
                        "showticklabels": False},
              "yaxis": {"range": (n - 1/2, -1/2),
                        "zeroline": False,
                        "showline": False,
                        "autotick": True,
                        "ticks": "",
                        "showticklabels": False}}
    fig = {"data": traces, "layout": layout}
    if return_fig:
        return fig
    ishow(fig, nb=nb, **kwargs)
