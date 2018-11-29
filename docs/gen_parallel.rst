=================================
Parallel & Distributed Generation
=================================


Parallel Generation - ``num_workers`` and ``pool``
--------------------------------------------------

Running a function for many different parameters theoretically allows perfect parallelization since each run is independent. ``xyzpy`` can automatically handle this in a number of different ways:

1. Supply ``parallel=True`` when calling ``Runner.run_combos(...)`` or ``Harvester.harvest_combos(...)`` etc. This spawns a ``ProcessExecutorPool`` with the same number of workers as logical cores.

2. Supply ``num_workers=...`` instead to explicitly control how any workers are used. Since for many numeric codes threading is controlled by the environement variable ``$OMP_NUM_THREADS`` you generally want the product of this and ``num_workers`` to be equal to the number of cores.

3. Supply ``executor=...`` to use any custom parallel pool-executor like object (e.g. a ``dask.distributed`` client or ``mpi4py`` pool) which has a ``submit``/``apply_async`` method, and yields futures with  a ``result``/``get`` method. More specifically, this covers pools with an API matching either ``concurrent.futures`` or an ``ipyparallel`` view. Pools from ``multiprocessing.pool`` are also explicitly handled.

4. Use a :class:`~xyzpy.Crop` to write combos to disk, which can then be 'grown' persistently by any computers with access to the filesystem, such as distributed cluster - see below.


Batched / Distributed generation - ``Crop``
-------------------------------------------

Running combos using the disk as a persistence mechanism requires one more object - the :class:`~xyzpy.Crop`. These can be instantiated directly or, generally more convenient, from a parent ``Runner`` or ``Harvester``:
:meth:`xyzpy.Runner.Crop` and :meth:`xyzpy.Harvester.Crop`. Using a ``Crop`` requires a number of steps:

1. Creation with:

    * a unique ``name`` to call this set of runs
    * a ``fn`` if not creating from an ``Harvester`` or ``Runner``
    * other optional settings such as ``batchsize`` controlling how many runs to group into one.

2. **'Sow'**. Use :meth:`xyzpy.Crop.sow_combos` to write ``combos`` into batches on disk.

3. **'Grow'**. Grow each batch. This can be done a number of ways:

    * In another different process, navigate to the same directory and run for example ``python -c "import xyzpy; c = xyzpy.Crop(name=...); xyzpy.grow(i, crop=c)"`` to grow the ith batch of crop with specified name. See :func:`~xyzpy.grow` for other options. This could manually be put in a script to run on a batch system.

    * Use :meth:`xyzpy.Crop.qsub_grow` - experimental! This automatically generates and submits a script using qsub. See its options and :func:`xyzpy.Crop.gen_qsub_script` for the template script.

    * Use :meth:`xyzpy.Crop.grow` or :meth:`xyzpy.Crop.grow_missing` to complete some or all of the batches locally. This can be useful to a) finish up a few missing/errored runs b) run all the combos with persistent progress, so that one can restart the runs at a completely different time/ with updated functions etc.

4. Watch the progress. ``Crop.__repr__`` will show how many batches have been completed of the total sown.

5. **'Reap'**. Once all the batches have completed, run ``Crop.reap()`` to collect the results and remove the batches' temporary directory. If the crop originated from a ``Runner`` or ``Harvester``, the data will be labelled, merged and saved accordingly.


See the full demonstrations in :ref:`Examples`.
