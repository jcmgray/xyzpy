.. xyzpy documentation master file, created by
   sphinx-quickstart on Thu Mar  8 23:59:43 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to xyzpy's documentation!
=================================

.. image:: https://dev.azure.com/xyzpy-org/xyzpy/_apis/build/status/jcmgray.xyzpy?branchName=develop
  :target: https://dev.azure.com/xyzpy-org/xyzpy
  :alt: Azure CI
.. image:: https://codecov.io/gh/jcmgray/xyzpy/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/jcmgray/xyzpy
  :alt: Code Coverage
.. image:: https://img.shields.io/lgtm/grade/python/g/jcmgray/xyzpy.svg
  :target: https://lgtm.com/projects/g/jcmgray/xyzpy/
  :alt: LGTM Grade
.. image:: https://readthedocs.org/projects/xyzpy/badge/?version=latest
  :target: http://xyzpy.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status

-------------------------------------------------------------------------------

`xyzpy <https://github.com/jcmgray/xyzpy>`__ is python library for efficiently
generating, manipulating and plotting data with a lot of dimensions, of the
type that often occurs in numerical simulations. It stands wholly atop the
labelled N-dimensional array library `xarray <http://xarray.pydata.org/en/stable/>`__.
The project's documentation is hosted on `readthedocs <http://xyzpy.readthedocs.io/>`__.

The aim is to take the pain and errors out of generating and exploring data
with a high number of possible parameters. This means:

- you don't have to write super nested for loops
- you don't have to remember which arrays/dimensions belong to which variables/parameters
- you don't have to parallelize over or distribute runs yourself
- you don't have to worry about loading, saving and merging disjoint data
- you don't need to guess when a set of runs is going to finish
- you don't have to write batch submission scripts or leave the notebook to use SGE, PBS or SLURM

As well as the ability to automatically parallelize over runs, ``xyzpy``
provides the ``Crop`` object that allows runs and results to be written to disk,
these can then be run by any process with access to the files - e.g. a batch system
such as SGE, PBS or SLURM - or just serve as a convenient persistent progress mechanism.

Once your data has been aggregated into a ``xarray.Dataset`` or ``pandas.DataFrame``
there exists many powerful visualization tools such as
`seaborn <https://seaborn.pydata.org/>`_, `altair <https://altair-viz.github.io/>`_, and
`holoviews <https://holoviews.org/#>`_ / `hvplot <https://hvplot.holoviz.org/>`_.
To these ``xyzpy`` adds also a simple 'oneliner' interface for interactively plotting the data
using `bokeh <https://bokeh.pydata.org/en/latest/>`__, or for static, publication ready figures
using `matplotlib <https://matplotlib.org/>`__, whilst being able to see the dependence on
up to 4 dimensions at once.


Overview
--------

The following guides introduce the main parts of ``xyzpy``:

.. toctree::
  :maxdepth: 2

  index_guides


Quick-start
-----------

.. code-block:: ipython

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


Detailed Examples
-----------------

These following examples are generated from the notebooks in ``docs/examples``.
They demonstrate more complete usage or advanced features of ``xyzpy``.

.. toctree::
  :maxdepth: 2

  index_examples


Installation
------------

``xzypy`` is itself a pure python package and can be found on `pypi <https://pypi.org/project/xyzpy/>`_,
and now `conda-forge <https://conda-forge.org/>`_ (the recommended installation method).
The core dependencies are:

- `numpy <http://www.numpy.org/>`__
- `xarray <http://xarray.pydata.org/en/latest/>`__ - *labelled ndarrays*
- `joblib <https://joblib.readthedocs.io/en/latest/index.html>`__ - *serialization and parallel processing*
- `tqdm <https://tqdm.github.io>`__ - *progress bars*

and the optional plotting functionality is provided by:

- `matplotlib <https://matplotlib.org/>`__
- `bokeh <https://bokeh.pydata.org/en/latest/>`__ - *interactive plotting*

The recommended way of installing these is also using the
`conda <https://conda.io/miniconda.html>`__ package manager.
A development version of ``xyzpy`` can be installed directly from github using the command:

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
