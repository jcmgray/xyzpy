"""
Helper functions for generating color spectrums.
"""
import math
import itertools

import matplotlib.cm as cm
from matplotlib.colors import Colormap, BASE_COLORS, CSS4_COLORS, to_rgba

from .xyz_cmaps import _XYZ_CMAPS


def _COLORS_MATPLOTLIB_TO_BOKEH(cols):
    for col in cols:
        yield (int(255 * col[0]), int(255 * col[1]), int(255 * col[2]), col[3])


_COLOR_CONVERT_METHODS = {
    ('MATPLOTLIB', 'BOKEH'): _COLORS_MATPLOTLIB_TO_BOKEH,
}


def convert_colors_string_to_tuple(cols):
    for col in cols:
        if isinstance(col, str):
            if col in BASE_COLORS:
                yield to_rgba(BASE_COLORS[col])
            elif col in CSS4_COLORS:
                yield to_rgba(CSS4_COLORS[col])
            else:
                raise ValueError("Color specifier {} "
                                 "not recognized".format(col))
        else:
            yield to_rgba(col)


def convert_colors(cols, outformat, informat='MATPLOTLIB'):
    """Convert lists of colors between formats
    """
    # make sure cols are numeric tuples first
    cols = convert_colors_string_to_tuple(cols)

    if informat == outformat:
        return cols
    return _COLOR_CONVERT_METHODS[(informat, outformat)](cols)


def xyz_colormaps(name=None, reverse=False):
    """Custom-defined colormaps
    """
    if isinstance(name, Colormap):
        if reverse:
            return name.reversed()
        else:
            return name

    try:
        import colorcet
        found_colorcet = True
    except (ImportError, KeyError):
        found_colorcet = False
    try:
        import cmocean
    except (ImportError, KeyError):
        pass

    # the default colormap
    if name is None:
        if found_colorcet:
            name = 'rainbow'
        else:
            name = 'xyz'

    # specify some aliases
    name = {
        'rainbow_pink': 'rainbow_bgyrm_35_85_c71',
        'rainbow_pink_r': 'rainbow_bgyrm_35_85_c71_r',
        'isolum_pink': 'isoluminant_cm_70_c39',
        'isolum_pink_r': 'isoluminant_cm_70_c39_r',
    }.get(name, name)

    # TODO: make this more general - not reliant on '_r' versions of cmaps
    if reverse:
        if name[-2:] != '_r':
            name = name + '_r'
        # 'un'-reverse
        else:
            name = name[:-2]

    # Custom xyzpy colormaps
    if name in _XYZ_CMAPS:
        return _XYZ_CMAPS[name]

    # special cases with name conflicts -->  prefer matplotlib
    if name not in {'inferno', 'coolwarm', 'blues'}:
        try:
            return colorcet.cm[name]
        except (NameError, KeyError):
            pass
        try:
            return getattr(cmocean.cm, name)
        except (NameError, AttributeError):
            pass

    # matplotlib colormaps
    return getattr(cm, name)


def get_default_sequential_cm(engine='MATPLOTLIB'):
    if engine == 'BOKEH':
        from bokeh.palettes import Category10_9
        return itertools.cycle(Category10_9)
    else:
        return itertools.cycle(rgb + (1.,) for rgb in cm.tab10.colors)


def cimple(
    hue,
    sat1=0.4,
    sat2=1.0,
    val1=0.95,
    val2=0.25,
    hue_shift=0.0,
    name='cimple',
    auto_adjust_sat=0.2,
):
    """Creates a color map for a single hue.
    """
    import matplotlib as mpl

    # account for the fact that yellows appear much less saturated
    sat1 += auto_adjust_sat * math.cos(hue * math.pi - 0.15)**8

    c1 = mpl.colors.hsv_to_rgb((hue % 1.0, sat1, val1))
    c2 = mpl.colors.hsv_to_rgb(((hue + hue_shift) % 1.0, sat2, val2))
    cdict = {
        'red': [(0.0, c1[0], c1[0]), (1.0, c2[0], c2[0])],
        'green': [(0.0, c1[1], c1[1]), (1.0, c2[1], c2[1])],
        'blue': [(0.0, c1[2], c1[2]), (1.0, c2[2], c2[2])],
    }
    return mpl.colors.LinearSegmentedColormap(name, cdict)


def cimple_bright(
    hue,
    sat1=0.8,
    sat2=0.9,
    val1=0.97,
    val2=0.3,
    hue_shift=0.0,
    name='cimple_bright',
):
    """Creates a color map for a single hue, with bright defaults.
    """
    return cimple(hue, sat1, sat2, val1, val2, hue_shift, name)
