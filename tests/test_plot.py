from pytest import fixture, mark
import numpy as np
import xarray as xr
import matplotlib

from xyzpy import (
    lineplot,
    auto_lineplot,
    ilineplot,
    auto_ilineplot,
    visualize_matrix
)
from xyzpy.plot.color import convert_colors


matplotlib.use('Template')


@fixture
def dataset_3d():
    x = [1, 2, 3, 4, 5, 6, 8]
    z = [10, 20, 40, 80]
    d = np.random.rand(7, 4)
    ds = xr.Dataset()
    ds["x"] = x
    ds["z"] = z
    ds["y"] = (("x", "z"), d)
    return ds


@fixture
def dataset_heatmap():
    x = np.linspace(10, 20, 11)
    y = np.linspace(20, 40, 21)
    xx, yy = np.meshgrid(x, y)
    c = np.cos(((xx**2 + yy**2)**0.5) / 2)
    s = np.sin(((xx**2 + yy**2)**0.5) / 2)
    ds = xr.Dataset(
        coords={'x': x,
                'y': y},
        data_vars={'c': (('y', 'x'), c),
                   's': (('y', 'x'), s)})
    return ds.where(ds.y < 35)


@fixture
def dataset_4d():
    x = np.linspace(10, 20, 11)
    y = np.linspace(20, 40, 21)
    phi = np.linspace(-0.1, 0.1, 3)
    xx, yy, phis = np.meshgrid(x, y, phi)
    c = np.cos(((xx**2 + yy**2)**0.5) / 2 + phis)
    s = np.sin(((xx**2 + yy**2)**0.5) / 2 + phis)
    ds = xr.Dataset(
        coords={'x': x,
                'y': y,
                'phi': phi},
        data_vars={'c': (('y', 'x', 'phi'), c),
                   's': (('y', 'x', 'phi'), s)})
    return ds


@fixture
def dataset_5d():
    x = np.linspace(10, 20, 11)
    y = np.linspace(20, 40, 21)
    phi = np.linspace(-0.5, 0.5, 3)
    A = [1, 2]
    xx, yy, phis, AA = np.meshgrid(x, y, phi, A)
    c = AA * np.cos(((xx**2 + yy**2)**0.5) / 2 + phis)
    s = AA * np.sin(((xx**2 + yy**2)**0.5) / 2 + phis)
    ds = xr.Dataset(
        coords={'x': x,
                'y': y,
                'phi': phi,
                'A': A},
        data_vars={'c': (('y', 'x', 'phi', 'A'), c),
                   's': (('y', 'x', 'phi', 'A'), s)})
    return ds


@fixture
def dataset_scatter():
    x = np.random.randn(100, 100)
    y = np.random.randn(100, 100)
    z = np.random.randn(100)

    a = np.linspace(0, 3, 100)
    b = np.linspace(0, 3, 100)

    ds = xr.Dataset(
        coords={'a': a,
                'b': b},
        data_vars={'x': (('a', 'b'), x),
                   'y': (('a', 'b'), y),
                   'z': ('b', z)})
    return ds.where(ds.z > 0.5)


# --------------------------------------------------------------------------- #
# TEST COLORS                                                                 #
# --------------------------------------------------------------------------- #

class TestConvertColors:
    def test_simple(self):
        cols = [(1, 0, 0, 1), (0, 0.5, 0, 0.5)]
        new_cols = list(convert_colors(cols, outformat='BOKEH'))
        assert new_cols == [(255, 0, 0, 1), (0, 127, 0, 0.5)]


# --------------------------------------------------------------------------- #
# TEST PLOTTERS                                                               #
# --------------------------------------------------------------------------- #

@mark.parametrize("plot_fn", [lineplot, ilineplot])
class TestCommonInterface:
    @mark.parametrize("colors", [True, False, None])
    @mark.parametrize("markers", [True, False, None])
    def test_works_2d(self,
                      plot_fn,
                      dataset_3d,
                      colors,
                      markers):
        plot_fn(dataset_3d, "x", "y", "z", return_fig=True,
                colors=colors,
                markers=markers)

    @mark.parametrize("colormap", ['xyz', 'viridis'])
    @mark.parametrize("colormap_log", [True, False])
    @mark.parametrize("colormap_reverse", [True, False])
    @mark.parametrize("string_z_coo", [True, False])
    def test_color_options(self,
                           plot_fn,
                           dataset_3d,
                           string_z_coo,
                           colormap,
                           colormap_log,
                           colormap_reverse):
        if string_z_coo:
            dataset_3d = dataset_3d.copy(deep=True)
            dataset_3d['z'] = ['a', 'b', 'c', 'd']
        plot_fn(dataset_3d, "x", "y", "z", return_fig=True,
                colors=True,
                colormap=colormap,
                colormap_log=colormap_log,
                colormap_reverse=colormap_reverse)

    @mark.parametrize("markers", [True, False, None])
    def test_works_1d(self,
                      plot_fn,
                      dataset_3d,
                      markers):
        plot_fn(dataset_3d.loc[{"z": 40}], "x", "y", return_fig=True,
                markers=markers)

    @mark.parametrize("padding", [None, 0.1])
    @mark.parametrize("xlims", [None, (0., 10.)])
    @mark.parametrize("ylims", [None, (-1., 1.)])
    def test_plot_range(self,
                        dataset_3d,
                        plot_fn,
                        padding,
                        xlims,
                        ylims):
        plot_fn(dataset_3d, "x", "y", "z", return_fig=True,
                padding=padding,
                xlims=xlims,
                ylims=ylims)

    @mark.parametrize("xlog", [True, False])
    @mark.parametrize("ylog", [True, False])
    @mark.parametrize("xticks", [None, (2, 3,)])
    @mark.parametrize("yticks", [None, (0.2, 0.3,)])
    @mark.parametrize("vlines", [None, (2, 3,)])
    @mark.parametrize("hlines", [None, (0.2, 0.3,)])
    def test_ticks_and_lines(self,
                             dataset_3d,
                             plot_fn,
                             xlog,
                             ylog,
                             xticks,
                             yticks,
                             vlines,
                             hlines):
        plot_fn(dataset_3d, "x", "y", "z", return_fig=True,
                xticks=xticks,
                yticks=yticks,
                vlines=vlines,
                hlines=hlines)


class TestLinePlot:

    def test_auto_lineplot(self):
        x = [1, 2, 3]
        y = [[4, 5, 6], [7, 8, 9]]
        auto_lineplot(x, y)

    def test_multi_plot_4d(self, dataset_4d):
        dataset_4d.xyz.lineplot('x', 'c', 'y', row='phi')
        dataset_4d.xyz.lineplot('x', 'c', 'y', col='phi')

    @mark.parametrize("colors", [False, True])
    def test_multi_plot_5d(self, dataset_5d, colors):
        kws = {'colors': colors}
        dataset_5d.xyz.lineplot('x', 'c', 'y', row='phi', col='A', **kws)
        dataset_5d.xyz.lineplot('x', 'c', 'y', col='phi', row='A', **kws)


class TestILinePlot:

    def test_auto_ilineplot(self):
        x = [1, 2, 3]
        y = [[4, 5, 6], [7, 8, 9]]
        auto_ilineplot(x, y)

    def test_multi_plot_4d(self, dataset_4d):
        dataset_4d.xyz.ilineplot('x', 'c', 'y', row='phi')
        dataset_4d.xyz.ilineplot('x', 'c', 'y', col='phi')

    @mark.parametrize("colors", [False, True])
    def test_multi_plot_5d(self, dataset_5d, colors):
        kws = {'colors': colors}
        dataset_5d.xyz.ilineplot('x', 'c', 'y', row='phi', col='A', **kws)
        dataset_5d.xyz.ilineplot('x', 'c', 'y', col='phi', row='A', **kws)


class TestHeatmap:
    def test_simple(self, dataset_heatmap):
        dataset_heatmap.xyz.heatmap('x', 'y', 'c', return_fig=True)

    def test_multi_plot_4d(self, dataset_4d):
        dataset_4d.xyz.heatmap('x', 'y', 'c', row='phi')
        dataset_4d.xyz.heatmap('x', 'y', 'c', col='phi')

    def test_multi_plot_5d(self, dataset_5d):
        dataset_5d.xyz.heatmap('x', 'y', 'c', row='phi', col='A')
        dataset_5d.xyz.heatmap('x', 'y', 'c', col='phi', row='A')


class TestIHeatmap:
    def test_simple(self, dataset_heatmap):
        dataset_heatmap.xyz.iheatmap('x', 'y', 'c', return_fig=True)

    def test_multi_plot_4d(self, dataset_4d):
        dataset_4d.xyz.iheatmap('x', 'y', 'c', row='phi')
        dataset_4d.xyz.iheatmap('x', 'y', 'c', col='phi')

    def test_multi_plot_5d(self, dataset_5d):
        dataset_5d.xyz.iheatmap('x', 'y', 'c', row='phi', col='A')
        dataset_5d.xyz.iheatmap('x', 'y', 'c', col='phi', row='A')


class TestScatter:

    def test_normal(self, dataset_scatter):
        dataset_scatter.xyz.scatter('x', 'y')

    def test_multi_plot_4d(self, dataset_4d):
        dataset_4d.xyz.scatter('x', 'y', c='c', row='phi')
        dataset_4d.xyz.scatter('x', 'y', c='c', col='phi')

    def test_multi_plot_5d(self, dataset_5d):
        dataset_5d.xyz.scatter('x', 'y', c='c', row='phi', col='A')
        dataset_5d.xyz.scatter('x', 'y', c='c', col='phi', row='A')


class TestIScatter:

    def test_normal(self, dataset_scatter):
        dataset_scatter.xyz.iscatter('x', 'y')

    def test_multi_plot_4d(self, dataset_4d):
        dataset_4d.xyz.iscatter('x', 'y', c='c', row='phi')
        dataset_4d.xyz.iscatter('x', 'y', c='c', col='phi')

    def test_multi_plot_5d(self, dataset_5d):
        dataset_5d.xyz.iscatter('x', 'y', c='c', row='phi', col='A')
        dataset_5d.xyz.iscatter('x', 'y', c='c', col='phi', row='A')


class TestHistogram:

    def test_normal(self, dataset_3d):
        dataset_3d.xyz.histogram('y', z='z')

    def test_multi_hist(self, dataset_heatmap):
        dataset_heatmap.xyz.histogram(('c', 's'))

    def test_multi_plot_4d(self, dataset_4d):
        dataset_4d.xyz.histogram(('c', 's'), row='phi')
        dataset_4d.xyz.histogram(('c', 's'), col='phi')

    def test_multi_plot_5d(self, dataset_5d):
        dataset_5d.xyz.histogram(('c', 's'), row='phi', col='A')
        dataset_5d.xyz.histogram(('c', 's'), col='phi', row='A')


class TestVisualizeMatrix():

    def test_single(self):
        x = np.random.randn(10, 10)
        visualize_matrix(x)
