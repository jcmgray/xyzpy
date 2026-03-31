# Installation

`xyzpy` is available on both [pypi](https://pypi.org/project/xyzpy/) and
[conda-forge](https://anaconda.org/conda-forge/xyzpy). While `xyzpy` is
pure python itself, the preferred way to install it is with
[pixi](https://pixi.sh), which creates isolated and reproducible environments
that can mix packages from [`conda-forge`](https://conda-forge.org/) (the
default) and also [`pypi`](https://pypi.org/).

**Installing with `pixi` (preferred):**
```bash
pixi init xyzpy-project
cd xyzpy-project
pixi add xyzpy
```

**Installing with `pip`:**
```bash
pip install xyzpy
# or
uv pip install xyzpy
```
It is recommended to use [`uv`](https://docs.astral.sh/uv/) to install and
manage purely pypi based environments.

**Installing with `conda` / `mamba`:**
```bash
conda install -c conda-forge xyzpy
```
[`miniforge`](https://github.com/conda-forge/miniforge) is the recommended way
to manage and install a conda-based environment.


**Installing the latest version directly from github:**

If you want to checkout the latest version of features and fixes, you can
install directly from the github repository:
```bash
pip install -U git+https://github.com/jcmgray/xyzpy.git
```

**Installing a local, editable development version:**

If you want to make changes to the source code and test them out, you can
install a local editable version of the package:
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