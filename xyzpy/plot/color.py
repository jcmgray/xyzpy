"""
Helper functions for generating color spectrums.
"""

import itertools
import math

import numpy as np

import matplotlib as mpl
import matplotlib.cm as cm
from matplotlib.colors import (
    BASE_COLORS,
    CSS4_COLORS,
    Colormap,
    LinearSegmentedColormap,
    hsv_to_rgb,
    to_rgba,
)

from .xyz_cmaps import _XYZ_CMAPS


def _COLORS_MATPLOTLIB_TO_BOKEH(cols):
    for col in cols:
        yield (int(255 * col[0]), int(255 * col[1]), int(255 * col[2]), col[3])


_COLOR_CONVERT_METHODS = {
    ("MATPLOTLIB", "BOKEH"): _COLORS_MATPLOTLIB_TO_BOKEH,
}


def convert_colors_string_to_tuple(cols):
    for col in cols:
        if isinstance(col, str):
            if col in BASE_COLORS:
                yield to_rgba(BASE_COLORS[col])
            elif col in CSS4_COLORS:
                yield to_rgba(CSS4_COLORS[col])
            else:
                raise ValueError(
                    "Color specifier {} not recognized".format(col)
                )
        else:
            yield to_rgba(col)


def convert_colors(cols, outformat, informat="MATPLOTLIB"):
    """Convert lists of colors between formats"""
    # make sure cols are numeric tuples first
    cols = convert_colors_string_to_tuple(cols)

    if informat == outformat:
        return cols
    return _COLOR_CONVERT_METHODS[(informat, outformat)](cols)


def xyz_colormaps(name=None, reverse=False):
    """Custom-defined colormaps"""
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
            name = "rainbow"
        else:
            name = "xyz"

    # specify some aliases
    name = {
        "rainbow_pink": "rainbow_bgyrm_35_85_c69",
        "rainbow_pink_r": "rainbow_bgyrm_35_85_c69_r",
        "isolum_pink": "isoluminant_cm_70_c39",
        "isolum_pink_r": "isoluminant_cm_70_c39_r",
    }.get(name, name)

    # TODO: make this more general - not reliant on '_r' versions of cmaps
    if reverse:
        if name[-2:] != "_r":
            name = name + "_r"
        # 'un'-reverse
        else:
            name = name[:-2]

    # Custom xyzpy colormaps
    if name in _XYZ_CMAPS:
        return _XYZ_CMAPS[name]

    # special cases with name conflicts -->  prefer matplotlib
    if name not in {"inferno", "coolwarm", "blues"}:
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


def get_default_sequential_cm(engine="MATPLOTLIB"):
    if engine == "BOKEH":
        from bokeh.palettes import Category10_9

        return itertools.cycle(Category10_9)
    else:
        return itertools.cycle(rgb + (1.0,) for rgb in cm.tab10.colors)


def cimple(
    hue,
    sat1=0.4,
    sat2=1.0,
    val1=0.95,
    val2=0.35,
    hue_shift=0.0,
    name="cimple",
    auto_adjust_sat=0.2,
):
    """Creates a color map for a single hue."""
    # account for the fact that yellows appear much less saturated
    sat1 += auto_adjust_sat * math.cos(hue * math.pi - 0.15) ** 8

    c1 = hsv_to_rgb((hue % 1.0, sat1, val1))
    c2 = hsv_to_rgb(((hue + hue_shift) % 1.0, sat2, val2))
    cdict = {
        "red": [(0.0, c1[0], c1[0]), (1.0, c2[0], c2[0])],
        "green": [(0.0, c1[1], c1[1]), (1.0, c2[1], c2[1])],
        "blue": [(0.0, c1[2], c1[2]), (1.0, c2[2], c2[2])],
    }
    return LinearSegmentedColormap(name, cdict)


def cimple_bright(
    hue,
    sat1=0.8,
    sat2=0.9,
    val1=0.97,
    val2=0.3,
    hue_shift=0.0,
    name="cimple_bright",
):
    """Creates a color map for a single hue, with bright defaults."""
    return cimple(hue, sat1, sat2, val1, val2, hue_shift, name)


def cimluv(
    hue,
    hue_shift=0.0,
    sat1=1.00,
    sat2=0.50,
    val1=0.80,
    val2=0.30,
    N=30,
    reverse=False,
):
    """Creates a color map for single hue, using HSLuv color space."""
    import hsluv
    import numpy as np

    pos = np.linspace(0.0, 1.0, N)
    hue1 = hue - hue_shift / 2
    hue2 = hue + hue_shift / 2
    hues = np.linspace(360 * hue1, 360 * hue2, N)
    sats = np.linspace(100 * sat1, 100 * sat2, N)
    luvs = np.linspace(100 * val1, 100 * val2, N)
    cdict = {"red": [], "green": [], "blue": []}
    for p, h, s, l in zip(pos, hues, sats, luvs):
        r, g, b = hsluv.hsluv_to_rgb((h, s, l))
        cdict["red"].append((p, r, r))
        cdict["green"].append((p, g, g))
        cdict["blue"].append((p, b, b))

    if reverse:
        for key in cdict:
            cdict[key].reverse()

    return LinearSegmentedColormap("cimluv", cdict)


def oklch_to_oklab(l, c, h):
    h_rad = np.radians(h)
    a = c * np.cos(h_rad)
    b = c * np.sin(h_rad)
    return (l, a, b)


def oklab_to_xyz(l, a, b):
    l_prime = l + 0.3963377774 * a + 0.2158037573 * b
    m_prime = l - 0.1055613458 * a - 0.0638541728 * b
    s_prime = l - 0.0894841775 * a - 1.2914855480 * b

    l_linear = l_prime**3
    m_linear = m_prime**3
    s_linear = s_prime**3

    x = (
        1.227013851103521 * l_linear
        - 0.5577999806518222 * m_linear
        + 0.2812561489664678 * s_linear
    )
    y = (
        -0.04058017842328059 * l_linear
        + 1.1122568696168302 * m_linear
        - 0.0716766786656012 * s_linear
    )
    z = (
        -0.0763812845057069 * l_linear
        - 0.4214819784180127 * m_linear
        + 1.586163220440795 * s_linear
    )
    return (x, y, z)


def xyz_to_linear_rgb(x, y, z):
    r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z
    g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z

    r = np.clip(r, 0.0, 1.0)
    g = np.clip(g, 0.0, 1.0)
    b = np.clip(b, 0.0, 1.0)
    return (r, g, b)


def linear_rgb_to_srgb(r, g, b):
    def gamma(u):
        return np.where(
            u <= 0.0031308,
            12.92 * u,
            1.055 * (u ** (1 / 2.4)) - 0.055,
        )

    sr = gamma(r)
    sg = gamma(g)
    sb = gamma(b)
    return (sr, sg, sb)


def oklch_to_rgb(l, c, h):
    l_ok, a_ok, b_ok = oklch_to_oklab(l, c, h)
    x, y, z = oklab_to_xyz(l_ok, a_ok, b_ok)
    r_linear, g_linear, b_linear = xyz_to_linear_rgb(x, y, z)
    sr, sg, sb = linear_rgb_to_srgb(r_linear, g_linear, b_linear)
    sr = np.clip(sr, 0.0, 1.0)
    sg = np.clip(sg, 0.0, 1.0)
    sb = np.clip(sb, 0.0, 1.0)
    return (sr, sg, sb)


def srgb_to_linear_rgb(sr, sg, sb):
    def inverse_gamma(u):
        if u <= 0.04045:
            return u / 12.92
        else:
            return ((u + 0.055) / 1.055) ** 2.4

    r_linear = inverse_gamma(sr)
    g_linear = inverse_gamma(sg)
    b_linear = inverse_gamma(sb)
    return (r_linear, g_linear, b_linear)


def linear_rgb_to_xyz(r, g, b):
    x = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    return (x, y, z)


def xyz_to_oklab(x, y, z):
    l = 0.8189330101 * x + 0.3618667424 * y - 0.1288597137 * z
    m = 0.0329845436 * x + 0.9293118715 * y + 0.0361456387 * z
    s = 0.0482003018 * x + 0.2643662691 * y + 0.6338517070 * z

    l_prime = l ** (1 / 3)
    m_prime = m ** (1 / 3)
    s_prime = s ** (1 / 3)

    L = (
        0.2104542553 * l_prime
        + 0.7936177850 * m_prime
        - 0.0040720468 * s_prime
    )
    a = (
        1.9779984951 * l_prime
        - 2.4285922050 * m_prime
        + 0.4505937099 * s_prime
    )
    b = (
        0.0259040371 * l_prime
        + 0.7827717662 * m_prime
        - 0.8086757660 * s_prime
    )
    return (L, a, b)


def oklab_to_oklch(L, a, b):
    c = math.sqrt(a**2 + b**2)
    h_rad = math.atan2(b, a)
    h_deg = math.degrees(h_rad) % 360.0
    return (L, c, h_deg)


def rgb_to_oklch(sr, sg, sb):
    r_linear, g_linear, b_linear = srgb_to_linear_rgb(sr, sg, sb)
    x, y, z = linear_rgb_to_xyz(r_linear, g_linear, b_linear)
    L, a, b = xyz_to_oklab(x, y, z)
    l, c, h = oklab_to_oklch(L, a, b)
    return (l, c, h)


def cmoke(
    hue,
    hue_shift=0.0,
    sat1=0.36,
    sat2=0.50,
    val1=0.38,
    val2=0.93,
    N=51,
    reverse=False,
):
    """Creates a color map for single hue, using OKLCH color space."""
    import numpy as np

    if isinstance(hue, (list, tuple)):
        hue1, hue2 = hue
    else:
        hue1 = hue - hue_shift / 2
        hue2 = hue + hue_shift / 2

    hs = np.linspace(360 * hue1, 360 * hue2, N)
    ls = np.linspace(val1, val2, N)
    cs = np.linspace(0.37 * sat1, 0.37 * sat2, N)

    pos = np.linspace(0.0, 1.0, N)
    cdict = {"red": [], "green": [], "blue": []}

    for p, l, c, h in zip(pos, ls, cs, hs):
        r, g, b = oklch_to_rgb(l, c, h)
        cdict["red"].append((p, r, r))
        cdict["green"].append((p, g, g))
        cdict["blue"].append((p, b, b))

    cmap = LinearSegmentedColormap("cmoke", cdict)

    if reverse:
        cmap = cmap.reversed()

    return cmap


doublerainbow = cmoke((1.8, -0.1), val1=0.2)
doublerainbow_r = doublerainbow.reversed()

mpl.colormaps.register(cmap=doublerainbow, name="doublerainbow")
mpl.colormaps.register(cmap=doublerainbow_r, name="doublerainbow_r")
