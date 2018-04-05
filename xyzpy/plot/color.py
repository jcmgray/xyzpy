"""
Helper functions for generating color spectrums.
"""
import itertools
import matplotlib.cm as cm
from matplotlib.colors import BASE_COLORS, CSS4_COLORS, to_rgba
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
