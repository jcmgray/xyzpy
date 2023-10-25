![xyzpy logo](https://github.com/jcmgray/xyzpy/blob/main/docs/_static/xyzpy-logo-title.png)

[![tests](https://github.com/jcmgray/xyzpy/actions/workflows/tests.yml/badge.svg)](https://github.com/jcmgray/xyzpy/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/jcmgray/xyzpy/branch/main/graph/badge.svg?token=Q5evNiuT9S)](https://codecov.io/gh/jcmgray/xyzpy)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/ba896d74c4954dd58da01df30c7bf326)](https://app.codacy.com/gh/jcmgray/xyzpy/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Docs](https://readthedocs.org/projects/xyzpy/badge/?version=latest)](https://xyzpy.readthedocs.io)
[![PyPI](https://img.shields.io/pypi/v/xyzpy?color=teal)](https://pypi.org/project/xyzpy/)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/xyzpy/badges/version.svg)](https://anaconda.org/conda-forge/xyzpy)

-------------------------------------------------------------------------------

[`xyzpy`](https://github.com/jcmgray/xyzpy) is python library for efficiently generating, manipulating and plotting data with a lot of dimensions, of the type that often occurs in numerical simulations. It stands wholly atop the labelled N-dimensional array library [`xarray`](http://xarray.pydata.org). The project's documentation is hosted on [readthedocs](http://xyzpy.readthedocs.io).

The aim is to take the pain and errors out of generating and exploring data with a high number of possible parameters. This means:

- you don't have to write super nested for loops
- you don't have to remember which arrays/dimensions belong to which variables/parameters
- you don't have to parallelize over or distribute runs yourself
- you don't have to worry about loading, saving and merging disjoint data
- you don't have to guess when a set of runs is going to finish
- you don't have to write batch submission scripts or leave the notebook to use SGE, PBS or SLURM

As well as the ability to automatically parallelize over runs, ``xyzpy``
provides the ``Crop`` object that allows runs and results to be written to disk,
these can then be run by any process with access to the files - e.g. a batch system
such as SGE, PBS or SLURM - or just serve as a convenient persistent progress mechanism.

Once your data has been aggregated into a ``xarray.Dataset`` or ``pandas.DataFrame``
there exists many powerful visualization tools such as
[`seaborn`](https://seaborn.pydata.org), [`altair`](https://altair-viz.github.io) and
[`holoviews`](https://holoviews.org) / [`hvplot`](https://hvplot.holoviz.org).
To these ``xyzpy`` adds also a simple 'oneliner' interface for interactively plotting the data
using [`bokeh`](https://bokeh.pydata.org), or for static, publication ready figures
using [`matplotlib`](https://matplotlib.org), whilst being able to see the dependence on
up to 4 dimensions at once.

![example](https://github.com/jcmgray/xyzpy/blob/main/docs/ex_simple.png)

Please see the [docs](http://xyzpy.readthedocs.io) for more information.

