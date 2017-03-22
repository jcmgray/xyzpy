"""
Helper functions for generating color spectrums.
"""
import itertools
import matplotlib.cm as cm
from .xyz_cmaps import _XYZ_CMAPS


def _COLORS_MATPLOTLIB_TO_BOKEH(cols):
    for col in cols:
        yield (int(255 * col[0]), int(255 * col[1]), int(255 * col[2]), col[3])


_COLOR_CONVERT_METHODS = {
    ('MATPLOTLIB', 'BOKEH'): _COLORS_MATPLOTLIB_TO_BOKEH,
}


def convert_colors(cols, outformat, informat='MATPLOTLIB'):
    """Convert lists of colors between formats
    """
    if informat == outformat:
        return cols
    return _COLOR_CONVERT_METHODS[(informat, outformat)](cols)


def xyz_colormaps(name):
    """Custom-defined colormaps
    """
    # Custom xyzpy colormaps
    if name in _XYZ_CMAPS:
        return _XYZ_CMAPS[name]

    # special cases with name conflicts
    if name in {'inferno', 'coolwarm', 'blues'}:
        return getattr(cm, name)

    try:
        import colorcet
        return colorcet.cm[name]
    except (ImportError, KeyError):
        pass

    try:
        import cmocean
        return getattr(cmocean.cm, name)
    except (ImportError, AttributeError):
        pass

    # matplotlib colormaps
    return getattr(cm, name)


def get_default_sequential_cm(engine='MATPLOTLIB'):
    if engine == 'BOKEH':
        from bokeh.palettes import Category10_9
        return itertools.cycle(Category10_9)
    else:
        return itertools.cycle(rgb + (1.,) for rgb in cm.Vega10.colors)
