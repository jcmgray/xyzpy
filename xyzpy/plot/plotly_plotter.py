"""
Functions for plotting datasets nicely.
"""
# TODO: unify options with xmlinplot                                          #
# TODO: plotly hlines, vlines                                                 #
# TODO: names                                                                 #
# TODO: reverse color                                                         #
# TODO: logarithmic color                                                     #
# TODO: not working currently?                                                #

import itertools
import numpy as np
from ..manage import auto_xyz_ds
from .color import calc_colors


def ishow(figs, nb=True, **kwargs):
    """ Show multiple plotly figures in notebook or on web. """
    if isinstance(figs, (list, tuple)):
        fig_main = figs[0]
        for fig in figs[1:]:
            fig_main["data"] += fig["data"]
    else:
        fig_main = figs
    if nb:
        from plotly.offline import init_notebook_mode
        from plotly.offline import iplot as plot
        init_notebook_mode()
    else:
        from plotly.plotly import plot
    plot(fig_main, **kwargs)


# --------------------------------------------------------------------------  #
# Line Plots                                                                  #
# --------------------------------------------------------------------------  #

def ilineplot(ds, y_coo, x_coo, z_coo=None,
              return_fig=False,
              figsize=(8, 6),          # absolute figure size
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
              # x-axis options
              xtitle=None,
              xlims=None,
              xlog=False,
              # y-axis options
              ytitle=None,
              ylims=None,
              ylog=False,
              # Line markers, styles and positions
              markers=None,
              # Misc options
              vlines=[],
              hlines=[],
              gridlines=True,
              font='Source Sans Pro',
              fontsize_title=20,
              fontsize_ticks=16,
              fontsize_xtitle=20,
              fontsize_ytitle=20,
              fontsize_ztitle=20,     # XXX: unused by plotly
              fontsize_legend=18,
              # TODO padding
              # TODO linewidths
              # TODO linestyles
              # TODO style gridlines
              **kwargs):
    """ Take a dataset and plot onse of its variables as a function of two
    coordinates using plotly. """
    # TODO: list of colors, send to calc_colors
    # TODO: mouse scroll zoom

    from plotly.graph_objs import Scatter

    # Work out whether to iterate over multiple lines
    if z_coo is not None:
        z_vals = ds[z_coo].data
    else:
        z_vals = (None,)

    # Color lines
    if colors is True:
        cols = iter(calc_colors(ds, z_coo, plotly=True,
                                colormap=colormap,
                                log_scale=colormap_log,
                                reverse=colormap_reverse))
    elif colors is False:
        cols = itertools.repeat(None)
    else:
        cols = itertools.cycle(colors)

    # Decide on using markers, and set custom markers and line-styles
    markers = (len(ds[y_coo]) <= 51) if markers is None else markers
    if markers:
        if len(z_vals) > 1:
            mrkrs = itertools.cycle(range(44))
        else:
            mrkrs = iter([None])
    else:
        mrkrs = itertools.repeat(None)

    def gen_traces():
        for z in z_vals:
            # Select data for current z coord - flatten for singletons
            sds = ds.loc[{z_coo: z}] if z is not None else ds
            x = sds[x_coo].data.flatten()
            y = sds[y_coo].data.flatten()

            # Trim out missing data
            notnull = ~np.isnan(x) & ~np.isnan(y)
            x, y = x[notnull], y[notnull]

            col = next(cols)

            yield Scatter({
                'x': x,
                'y': y,
                'name': str(z),
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

    layout = {
        'title': title,
        'titlefont': {
            'size': fontsize_title,
        },
        'width': figsize[0] * 100,
        'height': figsize[1] * 100,
        'xaxis': {
            'showline': True,
            'showgrid': gridlines,
            'title': x_coo if xtitle is None else xtitle,
            'mirror': 'ticks',
            'ticks': 'outside',
            'range': xlims if xlims is not None else None,
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
            'range': ylims if ylims is not None else None,
            'mirror': 'ticks',
            'ticks': 'outside',
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
    """ Automatic 2D-Heatmap plot using plotly. """
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


def plot_matrix(a, colormap="Greys", nb=True, return_fig=False, **kwargs):
    from plotly.graph_objs import Heatmap
    traces = [Heatmap({"z": -abs(a),
                       "colorscale": colormap,
                       "showscale": False})]
    layout = {"height": 500, "width": 500,
              "xaxis": {"autorange": True,
                        "zeroline": False,
                        "showline": False,
                        "autotick": True,
                        "ticks": "",
                        "showticklabels": False},
              "yaxis": {"autorange": True,
                        "zeroline": False,
                        "showline": False,
                        "autotick": True,
                        "ticks": "",
                        "showticklabels": False}}
    fig = {"data": traces, "layout": layout}
    if return_fig:
        return fig
    ishow(fig, nb=nb, **kwargs)
