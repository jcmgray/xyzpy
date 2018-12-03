Changelog
=========

Release notes for ``xyzpy``.


.. _whats-new.0.2.5:

v0.2.5 (unreleased)
--------------------

**Breaking changes**

- (:issue:`5`) ``combo_runner`` key argument ``pool`` renamed to ``executor``

**Enhancements**

- (:issue:`5`) Support ``multiprocessing.pool`` in ``combo_runner``
- Document timing and estimation utilities
- Use ``loky`` as the default parallel executor

**Bug fixes:**

- make sure ``Crop._batch_remainder`` synced with disk.
- update pytest marking parametrizations to xfail for recent pytest
- compatibility updates for dask and numba
- fix farming example which wasn't appearing



.. _whats-new.0.2.4:

v0.2.4 (1st November 2018)
--------------------------

**Bug fixes:**

- Various campatibility fixes for plotting functionality



.. _whats-new.0.2.3:

v0.2.3 (4th October 2018)
-------------------------

**Enhancements:**

- add :class:`~xyzpy.Timer`
- add :func:`~xyzpy.benchmark`
- add :class:`~xyzpy.Benchmarker`
- add :class:`~xyzpy.RunningStatistics`
- add :func:`~xyzpy.estimate_from_repeats`

**Bug fixes:**

- various fixes to batch growing and :class:`~xyzpy.Crop`
- various fixes to plotting



.. _whats-new.0.2.2:

v0.2.2 (7th June 2018)
----------------------

**Enhancements:**

- allow ``case_runner`` to return ``Dataset``

**Bug fixes:**

- (:issue:`1`) make ``numba`` an optional dependency




.. _whats-new.0.2.1:

v0.2.1 (27th May 2018)
----------------------

**Bug fixes:**

- docs updates
- distribute crop bathes more evenly
