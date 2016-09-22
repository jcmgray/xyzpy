def _process_plot_range(xlims, ylims, ds, x_coo, y_coo, padding):
    """Logic for processing limits and padding into plot ranges.
    """
    if xlims is  None and padding is None:
        xlims = None
    else:
        if xlims is not None:
            xmin, xmax = xlims
        else:
            xmax, xmin = ds[x_coo].max(), ds[x_coo].min()
        if padding is not None:
            xrnge = xmax - xmin
            xlims = (xmin - padding * xrnge, xmax + padding * xrnge)

    if ylims is not None or padding is not None:
        if ylims is not None:
            ymin, ymax = ylims
        else:
            ymax, ymin = ds[y_coo].max(), ds[y_coo].min()
        if padding is not None:
            yrnge = ymax - ymin
            ylims = (ymin - padding * yrnge, ymax + padding * yrnge)
    else:
        ylims = None

    return xlims, ylims
