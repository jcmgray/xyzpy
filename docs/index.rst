.. xyzpy documentation master file, created by
   sphinx-quickstart on Thu Mar  8 23:59:43 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to xyzpy's documentation!
=================================

.. image:: https://travis-ci.org/jcmgray/xyzpy.svg?branch=master
  :target: https://travis-ci.org/jcmgray/xyzpy
.. image:: https://codecov.io/gh/jcmgray/xyzpy/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/jcmgray/xyzpy
.. image:: https://img.shields.io/lgtm/grade/python/g/jcmgray/xyzpy.svg
  :target: https://lgtm.com/projects/g/jcmgray/xyzpy/
  :alt: LGTM Grade
.. image:: https://readthedocs.org/projects/xyzpy/badge/?version=latest
  :target: http://xyzpy.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status

----------------------------------------------------------------------------------

`xyzpy <https://github.com/jcmgray/xyzpy>`__ is python library for efficiently generating, manipulating and plotting data with a lot of dimensions, of the type that often occurs in numerical simulations. It stands wholly atop the labelled N-dimensional array library `xarray <http://xarray.pydata.org/en/stable/>`__. The project is hosted on `github <https://github.com/jcmgray/xyzpy>`__, please do submit any issues or PRs there.

The aim is to take the pain and errors out of generating and exploring data with a high number of possible parameters. This means:

- you don't have to write super nested for loops
- you don't have to remember which arrays/dimensions belong to which variables/parameters
- you don't have to parallelize over or distribute runs yourself
- you don't have to worry about loading, saving and merging disjoint data
- you don't need to guess when a set of runs is going to finish

As well as the ability to automatically parallelize over runs, ``xyzpy`` provides the :class:`~xyzpy.Crop` object that allows runs and results to be written to disk, these can then be run by any process with access to the files - e.g. a batch system - or just serve as a convenient persistent progress mechanism.

In terms of post-processing, as well as all the power of `xarray <http://xarray.pydata.org/en/stable/>`__, ``xyzpy`` adds uneven step differentiation and error propagation, filtering and interpolation - along any axis just specified by name.

The aim of the plotting functionality is to keep the same interface between interactively plotting the data using `bokeh <https://bokeh.pydata.org/en/latest/>`__, and static, publication ready figures using `matplotlib <https://matplotlib.org/>`__, whilst being able to see the dependence on up to 4 dimensions at once.


Overview
--------

The following guides introduce the main parts of ``xyzpy``:

.. toctree::
  :numbered:
  :maxdepth: 2

  generate
  gen_parallel
  manipulate
  plotting
  utilities


Quick-start
-----------

.. code-block:: ipython

    In [1]: import xyzpy as xyz
       ...: import time as time

    In [2]: def sumdiff(a, b):
       ...:     time.sleep(0.5)
       ...:     return a + b, a - b
       ...:

    In [3]: runner = xyz.Runner(sumdiff, var_names=['sum', 'diff'])
       ...: combos = {'a': range(1, 10), 'b': range(23, 27)}

    In [4]: runner.run_combos(combos, parallel=True)
    100%|###########################################| 36/36 [00:04<00:00,  7.96it/s]
    Out[4]:
    <xarray.Dataset>
    Dimensions:  (a: 9, b: 4)
    Coordinates:
      * a        (a) int64 1 2 3 4 5 6 7 8 9
      * b        (b) int64 23 24 25 26
    Data variables:
        sum      (a, b) int64 24 25 26 27 25 26 27 28 26 ... 31 32 33 34 32 33 34 35
        diff     (a, b) int64 -22 -23 -24 -25 -21 -22 ... -17 -18 -14 -15 -16 -17


Examples
--------

These following examples are generated from the notebooks in ``docs/examples``. They demonstrate more complete usage or advanced features of ``xyzpy``.

.. toctree::
  :maxdepth: 1

  examples/basic output example
  examples/complex output example
  examples/farming example
  examples/crop example
  examples/dask distributed example


Installation
------------

``xzypy`` is itself a pure python package and can be found on `pypi <https://pypi.org/project/xyzpy/>`_. The core dependencies are:

- `numpy <http://www.numpy.org/>`__
- `xarray <http://xarray.pydata.org/en/latest/>`__ - *labelled ndarrays*
- `joblib <https://joblib.readthedocs.io/en/latest/index.html>`__ - *serialization and parallel processing*
- `tqdm <https://tqdm.github.io>`__ - *progress bars*

Processing functions like filtering and differentiating require:

- `scipy <https://www.scipy.org/>`__
- `numba <http://numba.pydata.org/numba-doc/latest/index.html>`__ - *compiled gufuncs*

and the plotting functionality is provided by:

- `matplotlib <https://matplotlib.org/>`__
- `bokeh <https://bokeh.pydata.org/en/latest/>`__ - *interactive plotting*

The recommended way of installing these is using the `conda <https://conda.io/miniconda.html>`__ package manager. A development version of ``xyzpy`` can be installed directly from github using the command:

.. code-block:: bash

    pip install -U git+https://github.com/jcmgray/xyzpy.git


Release Notes
-------------

.. toctree::
  :maxdepth: 2

  changelog


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
