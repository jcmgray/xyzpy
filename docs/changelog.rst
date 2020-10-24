Changelog
=========

Release notes for ``xyzpy``.


.. _whats-new.0.3.2:

v0.3.2 (Unreleased)
--------------------------

**Enhancements**

- Generalize (and deprecate) :meth:`xyzpy.Crop.qsub_grow` to :meth:`xyzpy.Crop.grow_cluster` (:pull:`10`)
- Add SLURM support to :meth:`xyzpy.Crop.grow_cluster` (:pull:`10`)
- Add PBS support to :meth:`xyzpy.Crop.grow_cluster`
- Fix PBS crop submission for job arrays of size 1
- Add :func:`xyzpy.save_merge_ds` for manually aggregating datasets to disk
- Add ``allow_incomplete=True`` option to :meth:`xyzpy.Crop.reap` for gathering data even if the crop is not fully grown (:issue:`7`)
- Make new :class:`~xyzpy.Crop` instances by default automatically load information from disk if they have been already prepared/sown (:issue:`7`)
- Automatically load Crops in the current (or specified) directory with :func:`xyzpy.load_crops`.
- Add `'joblib'` and `'zarr'` as possible engines for saving and loading datasets
- Add utility :func:`xyzpy.getsizeof` to quite accurately get a python objects size
- Keep a running track of covariance using :class:`~xyzpy.utils.RunningCovariance`.


.. _whats-new.0.3.1:

v0.3.1 (25th January 2019)
--------------------------

**Bug fixes:**

- Make sure license is included in sdist/wheel distributions (:pull:`6`)


.. _whats-new.0.3.0:

v0.3.0 (21st January 2019)
--------------------------

**Breaking changes**

- Changed plot option ``markersize -> marker_size`` to match other keywords.

**Enhancements**

- New :class:`~xyzpy.Sampler` object - sparsely sample ``combos`` into a ``pandas.DataFrame``
- Decorate functions directly into ``Runner`` instances using :func:`~xyzpy.label`


.. _whats-new.0.2.5:

v0.2.5 (3rd December 2018)
--------------------------

**Breaking changes**

- (:issue:`5`) ``combo_runner`` key argument ``pool`` renamed to ``executor``

**Enhancements**

- (:issue:`5`) Support ``multiprocessing.pool`` in ``combo_runner``
- Document timing and estimation utilities
- Use ``loky`` as the default parallel executor
- plotting: add `xjitter` and `yjitter`

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
