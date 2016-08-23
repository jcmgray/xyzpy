""" Manage datasets --- loading, saving, merging etc. """

# TODO: Only aggregate Null data -------------------------------------------- #
# TODO: add singlet dimensions (for all or given vars) ---------------------- #

import numpy as np
import xarray as xr
from xarray.ufuncs import logical_not


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


def are_conflicting(ds1, ds2):
    """ Check whether two (aligned) datasets have any conflicting values. """
    both_not_null = logical_not(ds1.isnull() | ds2.isnull())
    return not ds1.where(both_not_null).equals(ds2.where(both_not_null))


def aggregate(*dss, overwrite=False, accept_newer=False):
    """ Aggregates xarray Datasets and DataArrays

    Parameters
    ----------
        *dss:
        overwrite:
        accept_newer:

    Returns
    -------
        ds: singlet Dataset containing data from all `dss`
    """
    # TODO: check if result var is all non-nan and could be all same dtype

    dss = iter(dss)
    ds = next(dss)

    for new_ds in dss:
        # Expand both to have same coordinates, padding with NaN
        ds, new_ds = xr.align(ds, new_ds, join="outer")

        # Check no data-loss will occur if overwrite not set
        if not overwrite and are_conflicting(ds, new_ds):
            raise ValueError("Conflicting values in datasets. "
                             "Consider setting `overwrite=True`.")

        # Fill out missing values in initial dataset
        ds = new_ds.fillna(ds) if accept_newer else ds.fillna(new_ds)

        # Add completely missing data_variables
        for var_name in new_ds.data_vars:
            if var_name not in ds.data_vars:
                ds[var_name] = new_ds[var_name]

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
