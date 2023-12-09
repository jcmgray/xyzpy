# Installation

`xyzpy` is available on both [pypi](https://pypi.org/project/xyzpy/) and [conda-forge](https://anaconda.org/conda-forge/xyzpy). While `xyzpy` is itself pure python, the recommended distribution would be [mambaforge](https://github.com/conda-forge/miniforge#mambaforge) for installing the various optional dependencies.

**Installing with `pip`:**
```bash
pip install xyzpy
```

**Installing with `conda`:**
```bash
conda install -c conda-forge xyzpy
```

**Installing with `mambaforge`:**
```bash
mamba install xyzpy
```

```{hint}
Mamba is a faster version of `conda`, and the -forge distritbution comes pre-configured with only the `conda-forge` channel, which further simplifies and speeds up installing dependencies.
```

**Installing the latest version directly from github:**

If you want to checkout the latest version of features and fixes, you can install directly from the github repository:
```bash
pip install -U git+https://github.com/jcmgray/xyzpy.git
```

**Installing a local, editable development version:**

If you want to make changes to the source code and test them out, you can install a local editable version of the package:
```bash
git clone https://github.com/jcmgray/xyzpy.git
pip install --no-deps -U -e xyzpy/
```

## Dependencies

`xyzpy` is itself a pure python package, built atop the following libraries:

- [numpy](http://www.numpy.org/) - *ndarrays*
- [xarray](http://xarray.pydata.org/en/latest/) - *labelled ndarrays*
- [joblib](https://joblib.readthedocs.io/en/latest/index.html) - *serialization and parallel processing*
- [tqdm](https://tqdm.github.io) - *progress bars*
- [pandas](https://pandas.pydata.org/) - *dataframes*

and the optional plotting functionality is provided by:

- [matplotlib](https://matplotlib.org/) - *plotting*
- [bokeh](https://bokeh.pydata.org/en/latest/) - *interactive plotting*