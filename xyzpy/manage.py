""" Manage datasets --- loading, saving, merging etc. """

# TODO: add singlet dimensions (for all or given vars) ---------------------- #

import os
from glob import glob

import numpy as np
import xarray as xr
import joblib


_DEFAULT_FN_CACHE_PATH = '__xyz_cache__'


def cache_to_disk(fn=None, *, cachedir=_DEFAULT_FN_CACHE_PATH, **kwargs):
    """Cache this function to disk, using joblib.
    """
    mem = joblib.Memory(cachedir=cachedir, verbose=0, **kwargs)

    # bare decorator
    if fn:
        return mem.cache(fn)

    # decorator called with kwargs
    else:
        def cache_to_disk_decorator(fn):
            return mem.cache(fn)
        return cache_to_disk_decorator


_engine_extensions = {
    'h5netcdf': '.h5',
    'netcdf4': '.nc',
    'joblib': '.dmp',
    'zarr': '.zarr',
}


def auto_add_extension(file_name, engine):
    """Make sure a file name has an extension that reflects its
    file type.
    """
    if not any(ext in file_name for ext in _engine_extensions.values()):
        extension = _engine_extensions[engine]
        file_name += extension
    return file_name


def save_ds(ds, file_name, engine="h5netcdf", **kwargs):
    """Saves a xarray dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to save.
    file_name : str
        Name of the file to save to.
    engine : {'h5netcdf', 'netcdf4', 'joblib', 'zarr'}, optional
        Engine used to save file with.

    Returns
    -------
        None
    """
    file_name = auto_add_extension(file_name, engine)

    # Parse out 'bad' netcdf types (only attributes -> values not so important)
    if engine not in {'joblib', 'zarr'}:
        for attr, val in ds.attrs.items():
            if val is None:
                ds.attrs[attr] = "None"
            if val is True:
                ds.attrs[attr] = "True"
            if val is False:
                ds.attrs[attr] = "False"

    if engine == 'joblib':
        joblib.dump(ds, file_name, **kwargs)
    elif engine == 'zarr':
        ds.to_zarr(file_name, **kwargs)
    else:
        ds.to_netcdf(file_name, engine=engine, **kwargs)


def load_ds(file_name,
            engine="h5netcdf",
            load_to_mem=None,
            create_new=False,
            chunks=None,
            **kwargs):
    """Loads a xarray dataset. Basically ``xarray.open_dataset`` with some
    different defaults and convenient behaviour.

    Parameters
    ----------
    file_name : str
        Name of file to open.
    engine : {'h5netcdf', 'netcdf4', 'joblib', 'zarr'}, optional
        Engine used to load file.
    load_to_mem : bool, optional
        Ince opened, load from disk into memory. Defaults to ``True`` if
        ``chunks=None``.
    create_new : bool, optional
        If no file exists make a blank one.
    chunks : int or dict
        Passed to ``xarray.open_dataset`` so that data is stored using
        ``dask.array``.


    Returns
    -------
    ds : xarray.Dataset
        Loaded Dataset.
    """
    file_name = auto_add_extension(file_name, engine)

    if not os.path.exists(file_name) and create_new:
        return xr.Dataset()

    if engine == 'joblib':
        return joblib.load(file_name, **kwargs)

    if (load_to_mem is None) and (chunks is None):
        load_to_mem = True
    else:
        if load_to_mem and (chunks is not None):
            raise ValueError("``chunks`` redundant if ``load_to_mem`` given.")
        else:
            load_to_mem = False

    if engine == 'zarr':
        ds = xr.open_zarr(file_name, chunks=chunks, **kwargs)

    else:
        opts = dict(engine=engine, chunks=chunks, **kwargs)

        try:
            ds = xr.open_dataset(file_name, **opts)
        except AttributeError as e1:
            if "object has no attribute" in str(e1) and engine == 'h5netcdf':
                opts['engine'] = 'netcdf4'
                ds = xr.open_dataset(file_name, **opts)
            else:
                raise e1

    if load_to_mem:
        ds.load()
        ds.close()

    return ds


def save_merge_ds(ds, fname, overwrite=None, **kwargs):
    """Save dataset ``ds``, but check for an existing dataset with that name
    first, and if it exists, merge the two before saving.

    Parameters
    ----------
    ds : xarray.Dataset
        The dataset to save.
    fname : str
        The file name.
    overwrite : {None, False, True}, optional
        How to merge the dataset with the existing dataset.

            - None: the datasets will be merged in there are no conflicts
            - False: data will be taken from old dataset if conflicting
            - True: data will be taken from new dataset if conflicting
    """

    # check for the existing file
    if os.path.exists(fname):
        old_ds = load_ds(fname)
    else:
        old_ds = xr.Dataset()

    # merge the new and old
    if overwrite is True:
        new_ds = ds.combine_first(old_ds)
    elif overwrite is False:
        new_ds = old_ds.combine_first(ds)
    else:
        new_ds = xr.merge([old_ds, ds])

    # write to disk
    save_ds(new_ds, fname, **kwargs)


def trimna(obj):
    """Drop values across all dimensions for which all values are NaN.
    """
    trimmed_obj = obj.copy()
    for d in obj.dims:
        trimmed_obj = trimmed_obj.dropna(d, how='all')
    return trimmed_obj


def sort_dims(ds):
    """Make sure all the dimensions on all the variables appear in the
    same order.
    """
    dim_list = tuple(ds.dims)

    for var in ds.data_vars:
        var_dims = (d for d in dim_list if d in ds[var].dims)
        ds[var] = ds[var].transpose(*var_dims)


def post_fix(ds, postfix):
    """Add ``postfix`` to the name of every data variable in ``ds``.
    """
    return ds.rename({old: old + '_' + postfix for old in ds.data_vars})


def check_runs(obj, dim='run', var=None, sel=()):
    """Print out information about the range and any missing values for an
    integer dimension.

    Parameters
    ----------
        obj : xarray object
            Data to check.
        dim : str (optional)
            Dimension to check, defaults to 'run'.
        var : str (optional)
            Subselect this data variable first.
        sel : mapping (optional)
            Subselect these other coordinates first.
    """
    try:
        if sel:
            obj = obj.loc[sel]
        if var:
            obj = obj[var]

        obj = trimna(obj)
        obj = obj.dropna(dim, how='all')
        obj = obj[dim].values
    except KeyError:
        print("* NO DATA *")
        return

    if obj.size == 0:
        print("* NO DATA *")
        return

    if 'int' not in str(obj.dtype):
        raise TypeError("check_runs can only check integer dimesions.")

    xmin, xmax = obj.min(), obj.max() + 1
    xmiss_start = obj[:-1][obj[1:] != obj[:-1] + 1] + 1
    xmiss_end = obj[1:][obj[1:] != obj[:-1] + 1]
    xmissing = list(zip(xmiss_start, xmiss_end))
    msg = "RANGE:  {:>7} -> {:<7} | TOTAL:  {:^7}".format(xmin, xmax, len(obj))
    if xmissing:
        msg += " | MISSING: {}".format(xmissing)
    print(msg)


def auto_xyz_ds(x, y_z=None):
    """Automatically turn an array into a `xarray` dataset. Transpose ``y_z``
    if necessary to automatically match dimension sizes.

    Parameters
    ----------
    x : array_like
        The x-coordinates.
    y_z : array_like, optional
        The y-data, possibly varying with coordinate z.
    """
    # assume non-coordinated - e.g. for auto_histogram
    if y_z is None:
        dims = ['yzwvu'[i] for i in range(x.ndim)]
        return xr.Dataset(data_vars={'x': (dims, x)})

    # Infer dimensions to coords mapping
    y_z = np.array(np.squeeze(y_z), ndmin=2)
    x = np.asarray(x)
    if np.size(x) == y_z.shape[0]:
        y_z = np.transpose(y_z)
    n_y = y_z.shape[0]

    # Turn into dataset
    if x.ndim == 2:
        ds = xr.Dataset(data_vars={'y': (['z', '_x'], y_z),
                                   'x': (['z', '_x'], x)})
    else:
        ds = xr.Dataset(coords={'x': x, 'z': np.arange(n_y)},
                        data_vars={'y': (['z', 'x'], y_z)})
    return ds


def merge_sync_conflict_datasets(base_name, engine='h5netcdf',
                                 combine_first=False):
    """Glob files based on `base_name`, merge them, save this new dataset if
    it contains new info, then clean up the conflicts.

    Parameters
    ----------
        base_name : str
            Base file name to glob on - should include '*'.
        engine : str , optional
            Load and save engine used by xarray.
    """
    fnames = glob(base_name)
    if len(fnames) < 2:
        print('Nothing to do - need multiple files to merge.')
        return

    # make sure first filename is the shortest -> assumed original
    fnames.sort(key=len)

    print("Merging:\n{}\ninto ->\n{}\n".format(fnames, fnames[0]))

    def load_dataset(fname):
        return load_ds(fname, engine=engine)

    datasets = list(map(load_dataset, fnames))

    # combine all the conflicts
    if combine_first:
        full_dataset = datasets[0]
        for ds in datasets[1:]:
            full_dataset = full_dataset.combine_first(ds)
    else:
        full_dataset = xr.merge(datasets)

    # save new dataset?
    if full_dataset.identical(datasets[0]):
        # nothing to do
        pass
    else:
        save_ds(full_dataset, fnames[0], engine=engine)

    # clean up conflicts
    for fname in fnames[1:]:
        os.remove(fname)


def save_df(df, name, engine='pickle', key='df', **kwargs):
    """Save a dataframe to disk.
    """
    meth = "to_{}".format(engine)
    if engine == 'hdf':
        kwargs['key'] = key
    if engine == 'csv':
        kwargs.setdefault('index', False)
    getattr(df, meth)(name, **kwargs)


def load_df(name, engine='pickle', key='df', **kwargs):
    """Load a dataframe from disk.
    """
    import pandas as pd
    func = "read_{}".format(engine)
    if engine == 'hdf':
        kwargs['key'] = key
    return getattr(pd, func)(name)
