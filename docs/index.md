# Welcome to xyzpy's documentation!

[![tests](https://github.com/jcmgray/xyzpy/actions/workflows/tests.yml/badge.svg)](https://github.com/jcmgray/xyzpy/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/jcmgray/xyzpy/branch/main/graph/badge.svg?token=Q5evNiuT9S)](https://codecov.io/gh/jcmgray/xyzpy)
[![Docs](https://readthedocs.org/projects/xyzpy/badge/?version=latest)](https://xyzpy.readthedocs.io)
[![PyPI](https://img.shields.io/pypi/v/xyzpy?color=teal)](https://pypi.org/project/xyzpy/)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/xyzpy/badges/version.svg)](https://anaconda.org/conda-forge/xyzpy)
[![Pixi Badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/prefix-dev/pixi/main/assets/badge/v0.json)](https://pixi.sh)

---

[xyzpy](https://github.com/jcmgray/xyzpy) is a Python library for efficiently generating, manipulating, and plotting data with a lot of dimensions, of the type that often occurs in numerical simulations. It stands wholly atop the labelled N-dimensional array library [xarray](http://xarray.pydata.org/en/stable/). The project's documentation is hosted on [readthedocs](http://xyzpy.readthedocs.io/).

The aim is to take the pain and errors out of generating and exploring data with a high number of possible parameters. This means:

- You don't have to write super nested for loops
- You don't have to remember which arrays/dimensions belong to which variables/parameters
- You don't have to parallelize over or distribute runs yourself
- You don't have to worry about loading, saving and merging disjoint data
- You don't need to guess when a set of runs is going to finish
- You don't have to write batch submission scripts or leave the notebook to use SGE, PBS or SLURM

As well as the ability to automatically parallelize over runs, {mod}`xyzpy` provides the {class}`Crop` object that allows runs and results to be written to disk, these can then be run by any process with access to the files - e.g. a batch system such as SGE, PBS or SLURM - or just serve as a convenient persistent progress mechanism.

Once your data has been aggregated into a {class}`xarray.Dataset` or {class}`pandas.DataFrame`, there exist many powerful visualization tools such as [seaborn](https://seaborn.pydata.org/), [altair](https://altair-viz.github.io/), and [holoviews](https://holoviews.org/#) / [hvplot](https://hvplot.holoviz.org/).

To these, `xyzpy` also adds a simple 'oneliner' interface for interactively plotting the data using [bokeh](https://bokeh.pydata.org/en/latest/), or for static, publication-ready figures using [matplotlib](https://matplotlib.org/), whilst being able to see the dependence on up to 4 dimensions at once.

## Overview

The following guides introduce the main parts of ``xyzpy``:

```{toctree}
:caption: Guides
:maxdepth: 2
installation
generate
gen_parallel
plotting
plotting-new
visualization
utilities
```


## Quick-start

```ipython

In [1]: import xyzpy as xyz
   ...: import time as time

In [2]: @xyz.label(var_names=['sum', 'diff'])
   ...: def sumdiff(a, b):
   ...:     time.sleep(0.5)
   ...:     return a + b, a - b
   ...:

In [3]: combos = {'a': range(1, 10), 'b': range(23, 27)}

In [4]: sumdiff.run_combos(combos, parallel=True)
100%|###########################################| 36/36 [00:06<00:00,  5.33it/s]
Out[4]:
<xarray.Dataset>
Dimensions:  (a: 9, b: 4)
Coordinates:
    * a        (a) int64 1 2 3 4 5 6 7 8 9
    * b        (b) int64 23 24 25 26
Data variables:
    sum      (a, b) int64 24 25 26 27 25 26 27 28 26 ... 31 32 33 34 32 33 34 35
    diff     (a, b) int64 -22 -23 -24 -25 -21 -22 ... -17 -18 -14 -15 -16 -17
```


## Detailed Examples

These following examples are generated from the notebooks in ``docs/examples``.
They demonstrate more complete usage or advanced features of ``xyzpy``.

```{toctree}
:maxdepth: 2
index_examples
```


## Development

```{toctree}
:caption: Development
:maxdepth: 2
changelog
GitHub Repository <https://github.com/jcmgray/xyzpy>
```