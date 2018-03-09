.. xyzpy documentation master file, created by
   sphinx-quickstart on Thu Mar  8 23:59:43 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to xyzpy's documentation!
=================================

`xyzpy <https://github.com/jcmgray/xyzpy>`_ is python library for efficiently generating, manipulating and plotting data with a lot of dimensions. It stands wholly atop the labelled N-dimensional array library `xarray <http://xarray.pydata.org/en/stable/>`_.

The aim is to take the pain and errors out of generating and exploring data with a high number of possible parameters. This means:

    - you don't have to write super nested for loops
    - you don't have to remember which arrays/dimensions belong to which variables/parameters
    - you don't have to parallelize over or distribute runs yourself
    - you don't have to worry about loading, saving and merging disjoint data
    - you don't need to guess when a set of runs is going to finish

In terms of processing, as well as all the power of `xarray <http://xarray.pydata.org/en/stable/>`_, ``xyzpy`` adds uneven step differentiation and error propagation, filtering and interpolation, along any axis just specified by name.

The aim of the plotting functionality is to keep the same interface between interactively plotting the data using `bokeh <https://bokeh.pydata.org/en/latest/>`_, and static, publication quality plotting using `matplotlib <https://matplotlib.org/>`_, whilst being able to see the dependence on up to 4 dimensions at once.


.. rubric:: Overview of each area:

.. toctree::
   :maxdepth: 2

   generate
   manipulate
   plot


.. rubric:: Complete list of modules, functions and classes:

.. toctree::
   :maxdepth: 1

   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
