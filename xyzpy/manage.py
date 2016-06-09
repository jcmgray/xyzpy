""" Manage datasets --- loading, saving, merging etc. """

import numpy as np
import xarray as xr


def xrsmoosh(*dss, accept_new=False):
    """ Aggregates xarray Datasets and DataArrays """
    # TODO: rename --> aggregate, look into, part_align -> concat.
    ds = dss[0]
    for new_ds in dss[1:]:
        # First make sure both datasets have the same variables
        for data_var in new_ds.data_vars:
            if data_var not in ds.data_vars:
                ds[data_var] = np.nan
        # Expand both to have same dimensions, padding with NaN
        ds, new_ds = xr.align(ds, new_ds, join="outer")
        # Fill NaNs one way or the other w.r.t. accept_new
        ds = new_ds.fillna(ds) if accept_new else ds.fillna(new_ds)
    return ds


def xrload(file_name, engine="h5netcdf", load_to_mem=True,
           create_new=True):
    """ Loads a xarray dataset. """
    try:
        try:
            ds = xr.open_dataset(file_name, engine=engine)
        except AttributeError as e1:
            if "object has no attribute" in str(e1):
                ds = xr.open_dataset(file_name, engine="netcdf4")
            else:
                raise e1
        if load_to_mem:
            ds.load()
            ds.close()
    except (RuntimeError, OSError) as e2:
        if "o such" in str(e2) and create_new:
            ds = xr.Dataset()
        else:
            raise e2
    return ds


def xrsave(ds, file_name, engine="h5netcdf"):
    """ Saves a xarray dataset. """
    # TODO: look for "." and append .xyz if not found
    ds.to_netcdf(file_name, engine=engine)


def xrgroupby_to_dim(ds, dim):
    """ Convert a grouped coordinate to dimension. """
    def gen_ds():
        for val, d in ds.groupby(dim):
            del d[dim]  # delete grouped labels
            d[dim] = [val]
            d, = xr.broadcast(d)
            yield d

    return xrsmoosh(*gen_ds())
