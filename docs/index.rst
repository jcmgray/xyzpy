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
.. image:: https://api.codacy.com/project/badge/Grade/7085feb3f47c4c509559778be5eb6a60
  :target: https://www.codacy.com/app/jcmgray/xyzpy?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=jcmgray/xyzpy&amp;utm_campaign=Badge_Grade
.. image:: https://landscape.io/github/jcmgray/xyzpy/master/landscape.svg?style=flat
  :target: https://landscape.io/github/jcmgray/xyzpy/master
  :alt: Code Health
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

.. toctree::
  :numbered:
  :maxdepth: 2

  generate
  gen_parallel
  manipulate
  plot


Examples
--------

.. toctree::
  :maxdepth: 1

  ex_basic
  ex_complex
  ex_farming
  ex_pool
  ex_crop


Installation
------------

``xzypy`` is a puthon python package. The core dependencies are:

- `numpy <http://www.numpy.org/>`__
- `xarray <http://xarray.pydata.org/en/latest/>`__ - labelled ndarrays

Processing functions like filtering and differentiating require:

- `scipy <https://www.scipy.org/>`__
- `numba <http://numba.pydata.org/numba-doc/latest/index.html>`__ - compiled gufuncs

and the plotting functionality is provided by:

- `matplotlib <https://matplotlib.org/>`__
- `bokeh <https://bokeh.pydata.org/en/latest/>`__ - interactive plotting

The recommended way of installing these is using the `conda <https://conda.io/miniconda.html>`__ package manager. ``xyzpy`` itself can be installed directly from github using the command:

.. code-block:: bash

    pip install -U git+https://github.com/jcmgray/xyzpy.git


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
