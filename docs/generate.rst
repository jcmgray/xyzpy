===============
Generating Data
===============

The idea of ``xyzpy`` is to ease the some of the pain generating data with a large parameter space.
The central aim being that, once you know what a single run of a function looks like, it should be as easy as saying, "run these combinations of parameters, now run these particular cases" with everything automatically aggregated into a fully self-described dataset.

Combos & Cases
--------------

The main backend function is :func:`~xyzpy.combo_runner`, which in its simplest form takes a function, say:

.. code-block:: python

    def foo(a, b, c):
        return a, b, c

and ``combos`` of the form:

.. code-block:: python

    combos = [
        ('a', [1, 2, 3]),
        ('b', ['x', 'y', 'z']),
        ('c', [True, False]),
    ]

and generates a nested (here 3 dimensional) array of all the outputs of ``foo`` with the ``3 * 3 * 2 = 18`` combinations of input arguments:

.. code-block:: python

    >>> combo_runner(foo, combos)
    100%|##########| 18/18 [00:00<00:00, 88508.17it/s]
    ((((1, 'x', True), (1, 'x', False)),
      ((1, 'y', True), (1, 'y', False)),
      ((1, 'z', True), (1, 'z', False))),
     (((2, 'x', True), (2, 'x', False)),
      ((2, 'y', True), (2, 'y', False)),
      ((2, 'z', True), (2, 'z', False))),
     (((3, 'x', True), (3, 'x', False)),
      ((3, 'y', True), (3, 'y', False)),
      ((3, 'z', True), (3, 'z', False))))

Note the progress bar shown. If the function was slower (generally the target case for ``xyzpy``), this would show the remaining time before completion.

There is also :func:`~xyzpy.case_runner` for running isolated cases:

.. code-block:: python

    >>> cases = [(4, 'z', False), (5, 'y', True)]
    >>> case_runner(foo, fn_args=('a', 'b', 'c'), cases=cases)
    100%|##########| 2/2 [00:00<00:00, 5111.89it/s]
    ((4, 'z', False), (5, 'y', True))

Alone, these are not super useful, but the rest of the functionality is built on them.


Describing the function - ``Runner``
------------------------------------

To automatically put the generated data into a labelled :class:`xarray.Dataset` you need to describe your function using the :class:`~xyzpy.Runner` class. In the simplest case this is just a matter of naming the outputs:

.. code-block:: python

    >>> runner = Runner(foo, var_names=['a_out', 'b_out', 'c_out'])
    >>> runner.run_combos(combos)
    100%|##########| 18/18 [00:00<00:00, 36720.56it/s]
    <xarray.Dataset>
    Dimensions:  (a: 3, b: 3, c: 2)
    Coordinates:
      * a        (a) int64 1 2 3
      * b        (b) <U1 'x' 'y' 'z'
      * c        (c) bool True False
    Data variables:
        a_out    (a, b, c) int64 1 1 1 1 1 1 2 2 2 2 2 2 3 3 3 3 3 3
        b_out    (a, b, c) <U1 'x' 'x' 'y' 'y' 'z' 'z' 'x' 'x' 'y' 'y' 'z' 'z' ...
        c_out    (a, b, c) bool True False True False True False True False True ...

The output dataset is also stored in ``runner.last_ds`` and, as can be seen, is completely labelled - see `xarray <https://xarray.pydata.org/>`__ for details of the myriad functionality this allows. See also the :ref:`Basic Output Example` for a more complete example.

Various other arguments to :class:`~xyzpy.Runner` allow: 1) constant arguments to be specified, 2) for each variable to have its own dimensions and 3) to specify the coordinates of those dimensions.
See the :ref:`Structured Output with Julia Set Example`, for how to describe structured data.

If the function itself returns a :class:`xarray.Dataset`, then just use ``var_names=None`` and all the outputs will be concatenated together automatically.


Aggregating data - ``Harvester``
--------------------------------

A common scenario when running simulations is the following:

1. Generate some data
2. Save it to disk
3. Generate a different set of data (maybe after analysis of the first set)
4. Load the old data
5. Merge the new data with the old data
6. Save the new combined data
7. Repeat

The aim of the :class:`~xyzpy.Harvester` is to automate that process. A :class:`~xyzpy.Harvester` is instantiated with a :class:`~xyzpy.Runner` instance and, optionally, a ``data_name``. If a ``data_name`` is given, then every time a round of combos/cases is generated, it will be automatically synced with a on-disk dataset of that name. Either way, the harvester will aggregate all runs into the ``full_ds`` attribute.

.. code-block:: python

    >>> harvester = Harvester(runner, data_name='foo.h5')
    >>> harvester.harvest_combos(combos)
    100%|##########| 18/18 [00:00<00:00, 18540.64it/s]

Which, because it didn't exist yet, created the file ``data_name``:

.. code-block:: bash

    $ ls *.h5
    foo.h5

:meth:`xyzpy.Harvester.harvest_combos` calls :meth:`xyzpy.Runner.run_combos` itself - this doesn't need to be done seperately.

Now we can run a second set of different combos:

.. code-block:: python

    >>> # if we are using a runner, combos can be supplied as a dict
    >>> combos2 = {
    ...     'a': [4, 5, 6],
    ...     'b': ['w', 'v'],
    ...     'c': [True, False],
    ... }
    >>> harvester.harvest_combos(combos2)
    100%|##########| 12/12 [00:00<00:00, 31635.23it/s]

Now we can check the total dataset containing all combos and cases run so far:

    >>> harvester.full_ds
    <xarray.Dataset>
    Dimensions:  (a: 6, b: 5, c: 2)
    Coordinates:
      * a        (a) int64 1 2 3 4 5 6
      * b        (b) object 'v' 'w' 'x' 'y' 'z'
      * c        (c) int8 1 0
    Data variables:
        a_out    (a, b, c) float64 nan nan nan nan 1.0 1.0 1.0 1.0 1.0 1.0 nan ...
        b_out    (a, b, c) object nan nan nan nan 'x' 'x' 'y' 'y' 'z' 'z' nan ...
        c_out    (a, b, c) float64 nan nan nan nan 1.0 0.0 1.0 0.0 1.0 0.0 nan ...

Note that, since the different runs were disjoint, missing values have automatically been filled in with ``nan`` values - see :func:`xarray.merge`. The on-disk dataset now contains both runs.


Summary
-------

  1. :func:`~xyzpy.combo_runner` is the core function which outputs a nested tuple and contains the parallelization logic and progress display etc.

  2. :class:`~xyzpy.Runner` and :meth:`xyzpy.Runner.run_combos` are used to describe the function's output and perform a single set of runs yielding a :class:`~xarray.Dataset`. These internally call :func:`~xyzpy.combo_runner`.

  3. :class:`~xyzpy.Harvester` and :meth:`xyzpy.Runner.harvest_combos` are used to perform many sets of runs, continuously merging the results into one larger :class:`~xarray.Dataset` - ``Harvester.full_ds``, probably synced to disk. These internally call :meth:`xyzpy.Runner.run_combos`.

In general, you would only generate data with one of these methods at once - see the full demonstrations in :ref:`Examples`.
