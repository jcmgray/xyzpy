Basic Output Example
====================

Imagine we want to explore the function :func:`scipy.special.eval_jacobi`. It takes four different arguments and we want to get a feel for what each does.

First we wrap it in a :class:`~xyzpy.Runner` object, that encapsulates how to run all the different combinations of its arguments and automatically labels the output.

.. code-block:: python

    >>> from xyzpy import *
    >>> from from scipy.special import eval_jacobi

    >>> def jacobi(x, n, alpha, beta):
    ...     return eval_jacobi(n, alpha, beta, x)

    >>> r = Runner(jacobi, var_names='Pn(x)')

This is as simple as it gets, the function ``jacobi`` has one output variable, which we are calling ``'Pn(x)'``.

Now let's define all the different values we want to try for each argument (the function actually vectorizes over ``x`` so this is overkill, but serves as a good demonstration):

.. code-block:: python

    >>> import numpy as np
    >>> combos = {
    ...     'x': np.linspace(0, 1, 101),
    ...     'n': [1, 2, 4, 8, 16],
    ...     'alpha': np.linspace(0, 2, 3),
    ...     'beta': np.linspace(0, 1, 5),
    ... }

Now, let's run the function for every combination of the above parameters:

.. code-block:: python

    >>> r.run_combos(combos)
    100%|##########| 7575/7575 [00:00<00:00, 166777.00it/s]
    <xarray.Dataset>
    Dimensions:  (alpha: 3, beta: 5, n: 5, x: 101)
    Coordinates:
      * x        (x) float64 0.0 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 ...
      * n        (n) int64 1 2 4 8 16
      * alpha    (alpha) float64 0.0 1.0 2.0
      * beta     (beta) float64 0.0 0.25 0.5 0.75 1.0
    Data variables:
        Pn(x)    (x, n, alpha, beta) float64 0.0 -0.125 -0.25 -0.375 -0.5 0.5 ...

The resulting dataset is stored in ``r.last_ds`` and is an automatically labelled n-dimensional :class:`xarray.Dataset`. Let's plot what we have, showing the effect of all four dimensions:

.. code-block:: python

    >>> r.last_ds.xyz.ilineplot(
    ...     x='x', y='Pn(x)', z='beta', col='n', row='alpha', ylims=(-2, 2),
    ...     colors=True, colormap='rainbow_r', colorbar=True,
    ... )

.. image:: ex_simple.png
