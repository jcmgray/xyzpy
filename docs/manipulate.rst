===============
Processing Data
===============

`xarray <xarray.pydata.org>`__ itself is very powerful and contains most functionality required to process data. On top of this ``xyzpy`` adds a few potentially helpful functions using the 'gufunc' capabilities of `numba <https://numba.pydata.org/numba-doc/latest/user/vectorize.html>`__. These are all 1D functions that can be called through the registered (upon import) 'accessor' attribute ``xarray.Dataset.xyz`` or with the functional versions listed below which have ``xr_`` prepended.

1. Differentiation, not requiring equal steps (fornberg's method and a lagrange
   interpolation method):

       - :func:`~xyzpy.xr_diff_u` - differentiate with unevenly spaced data
       - :func:`~xyzpy.xr_diff_u_err` - propagage errors for above method
       - :func:`~xyzpy.xr_diff_fornberg`

2. Wiener, Butterworth or Bessel filtering (using scipy).

       - :func:`~xyzpy.xr_filtfilt_butter`
       - :func:`~xyzpy.xr_filtfilt_bessel`
       - :func:`~xyzpy.xr_filter_wiener`

3. Interpolation (using scipy):

       - :func:`~xyzpy.xr_interp`
       - :func:`~xyzpy.xr_pchip`

4. Fitting (using scipy ``unispline`` or polynomials).

       - :func:`~xyzpy.xr_unispline`
       - :func:`~xyzpy.xr_polyfit`

5. max/min Coordinate finding.

       - :func:`~xyzpy.xr_idxmax`
       - :func:`~xyzpy.xr_idxmin`


One workflow the combination of these allow, for example, is to find the locations of steepest descent/ascent in noisy data: (i) filter or fit data, (ii) differentiate smooth data (iii) find coordinate min/max of gradients.
