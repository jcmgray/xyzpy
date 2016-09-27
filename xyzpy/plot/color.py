"""
Helper functions for generating color spectrums.
"""
# TODO: Transparency                                                          #

from math import log
import itertools
import numpy as np
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap


def _COLORS_MATPLOTLIB_TO_PLOTLY(cols):
    for col in cols:
        if len(col) == 3:  # append alpha value
            col = itertools.chain(col, (1,))
        yield "rgba" + str(tuple(int(255*rgb) for rgb in col))


_conversion_methods = {
    ('MATPLOTLIB', 'PLOTLY'): _COLORS_MATPLOTLIB_TO_PLOTLY}


def convert_colors(cols, outformat, informat='MATPLOTLIB'):
    """Convert lists of colors between formats
    """
    if informat == outformat:
        return cols
    return _conversion_methods[(informat, outformat)](cols)


def _xyz_colormaps(name):
    """ Custom-defined colormaps """
    cmaps = {
        'xyz': {'red':   ((0.00,    0/255,   0/255),
                          (0.36,   44/255,  44/255),
                          (0.50,  238/255, 238/255),
                          (0.64,  245/255, 245/255),
                          (1.00,  223/255, 223/255)),
                'green': ((0.00,   52/255,  52/255),
                          (0.36,  190/255, 190/255),
                          (0.50,  215/255, 215/255),
                          (0.64,  170/255, 170/255),
                          (1.00,    0/255,   0/255)),
                'blue':  ((0.00,  161/255, 161/255),
                          (0.36,  140/255, 140/255),
                          (0.50,    0/255,   0/255),
                          (0.64,    0/255,   0/255),
                          (1.00,   67/255,  67/255))},
        #    0.00    #0034a1    rgb(0,    52, 161)
        #    0.36    #2cbe8c    rgb(44,  190, 140)
        #    0.50    #eed700    rgb(238, 215,   0)
        #    0.64    #F5AA00    rgb(245, 170,   0)
        #    1.00    #DF0043    rgb(223,   0,  67)
        'neon': {'red':   ((0.00,    0/255,   0/255),
                           (0.25,  118/255, 118/255),
                           (0.50,  255/255, 255/255),
                           (0.75,  255/255, 255/255),
                           (1.00,  224/255, 224/255)),
                 'green': ((0.00,  152/255, 152/255),
                           (0.25,  215/255, 215/255),
                           (0.50,  193/255, 193/255),
                           (0.75,  117/255, 117/255),
                           (1.00,    0/255,   0/255)),
                 'blue':  ((0.00,  255/255, 255/255),
                           (0.25,  202/255, 202/255),
                           (0.50,    0/255,   0/255),
                           (0.75,  117/255, 117/255),
                           (1.00,  115/255, 115/255))}
        #    0.00    #0098ff    rgb(  0, 152, 255)
        #    0.25    #76d7ca    rgb(118, 215, 202)
        #    0.50    #ffc100    rgb(255, 193,   0)
        #    0.75    #ff7575    rgb(255, 117, 117)
        #    1.00    #e00073    rgb(224,   0, 115)
    }
    return LinearSegmentedColormap(name, cmaps[name])


def calc_colors(ds, z_coo, colormap="viridis",
                log_scale=False, reverse=False, outformat='MATPLOTLIB'):
    """ Calculate colors for a set of lines given their relative position
    in the range of `z_coo`.

    Parameters
    ----------
        ds: xarray dataset
        z_coo: coordinate describing the range of lines
        colormap: which matplotlib colormap style to use
        log_scale: find relative logarithmic position
        reverse: reverse the relative ordering
        plotly: modify string for plotly compatibility

    Returns
    -------
        list of colors corresponding to each line in `z_coo`.
    """
    try:
        cmap = _xyz_colormaps(colormap)
    except KeyError:
        cmap = getattr(cm, colormap)

    try:
        zmin, zmax = ds[z_coo].values.min(), ds[z_coo].values.max()
        # Relative function
        f = log if log_scale else lambda a: a
        # Relative place in range according to function
        rvals = [1 - (f(z)-f(zmin))/(f(zmax)-f(zmin))
                 for z in ds[z_coo].values]
    except TypeError:  # no relative coloring possible e.g. for strings
        rvals = np.linspace(0, 1.0, ds[z_coo].size)

    # Map to mpl colormap, reversing if required
    cols = [cmap(1 - rval if reverse else rval) for rval in rvals]
    cols = convert_colors(cols, outformat)

    return cols
