""" Manage datasets --- loading, saving, merging etc. """

# TODO: Only aggregate Null data -------------------------------------------- #
# TODO: add singlet dimensions (for all or given vars) ---------------------- #

import numpy as np
import xarray as xr


def _auto_add_extension(file_name, engine):
    if "." not in file_name:
        extension = ".h5" if engine == "h5netcdf" else ".nc"
        file_name += extension
    return file_name


def xrsave(ds, file_name, engine="h5netcdf"):
    """ Saves a xarray dataset.

    Parameters
    ----------
        ds: Dataset to save
        file_name: name of file to save to
        engine: engine used to save file

    Returns
    -------
        None
    """
    file_name = _auto_add_extension(file_name, engine)
    ds.to_netcdf(file_name, engine=engine)


def xrload(file_name, engine="h5netcdf", load_to_mem=True, create_new=False):
    """
    Loads a xarray dataset.

    Parameters
    ----------
        file_name: name of file
        engine: engine used to load file
        load_to_mem: once opened, load from disk to memory
        create_new: if no file exists make a blank one

    Returns
    -------
        ds: loaded Dataset
    """
    file_name = _auto_add_extension(file_name, engine)
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


def aggregate(*dss, accept_new=False):
    """ Aggregates xarray Datasets and DataArrays """
    # TODO: overwrite option, rather than accept_new, raise error if not
    # TODO: rename --> aggregate, look into, part_align -> concat.
    # TODO: check if result var is all non-nan and could be all same dtype

    if accept_new:
        dss = tuple(reversed(dss))

    ds = dss[0]
    for new_ds in dss[1:]:
        # First make sure both datasets have the same variables
        for data_var in new_ds.data_vars:
            if data_var not in ds.data_vars:
                ds[data_var] = np.nan
        # Expand both to have same dimensions, padding with NaN
        ds, _ = xr.align(ds, new_ds, join="outer")
        # assert all(ds.loc[new_ds.coords].isnull())
        # Fill NaNs one way or the other w.r.t. accept_new
        ds = ds.fillna(new_ds)
    return ds


xrsmoosh = aggregate


def auto_xyz_ds(x, y_z):
    """ Automatically turn an array into a `xarray` dataset """
    # Infer dimensions to coords mapping
    y_z = np.array(np.squeeze(y_z), ndmin=2)
    if np.size(x) == y_z.shape[0]:
        y_z = np.transpose(y_z)
    n_y = y_z.shape[0]
    # Turn into dataset
    ds = xr.Dataset(coords={'x': x, 'z': np.arange(n_y)})
    ds['y'] = (('z', 'x'), y_z)
    return ds


def xrgroupby_to_dim(ds, dim):
    """ Convert a grouped coordinate to dimension. """
    def gen_ds():
        for val, d in ds.groupby(dim):
            del d[dim]  # delete grouped labels
            d[dim] = [val]
            d, = xr.broadcast(d)
            yield d

    return aggregate(*gen_ds())
