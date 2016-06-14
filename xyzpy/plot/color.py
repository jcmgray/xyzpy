from math import log


def calc_colors(ds, z_coo, colormap="viridis",
                log_scale=False, reverse=False, plotly=True):
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
        list of colors corresponding to each line in `z_coo`. """
    import matplotlib.cm as cm
    cmap = getattr(cm, colormap)

    zmin, zmax = ds[z_coo].values.min(), ds[z_coo].values.max()

    # Relative function
    f = log if log_scale else lambda a: a

    # Relative place in range according to function
    rvals = [1 - (f(z)-f(zmin))/(f(zmax)-f(zmin)) for z in ds[z_coo].values]

    # Map to mpl colormap, reversing if required
    cols = [cmap(1 - rval if reverse else rval) for rval in rvals]

    # Add string modifier if using for plotly
    if plotly:
        cols = ["rgba" + str(col) for col in cols]

    return cols
