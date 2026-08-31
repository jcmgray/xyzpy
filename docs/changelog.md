# Changelog

Release notes for `xyzpy`.


(whats-new-1-3-5)=
## v1.3.5 (2026-08-31)

**Enhancements:**

- Add top-level `xyz.plot` for directly plotting array-like y-data or x/y pairs, including multiple series and per-series x-values. Plotting options are passed through to {func}`~xyzpy.infiniplot`.

**Bug fixes:**

- Default to importing `cloudpickle` directly rather than `joblib.externals.cloudpickle`, which recent `joblib` versions no longer vendor. The old location is still used as a fallback.

**Other:**

- Add `cloudpickle` as an explicit dependency, and demote `dask` to a test dependency, since it is only needed for the optional chunked/lazy `xarray` loading path.


(whats-new-1-3-4)=
## v1.3.4 (2026-04-30)

**Enhancements:**

- Expose `parent_dir` on {meth}`~xyzpy.Harvester.cultivate` and {func}`~xyzpy.cultivate` so the on-disk crop folder can be placed somewhere other than the current working directory.

**Bug fixes:**

- Fix {func}`~xyzpy.parse_into_cases` raising `IndexError` (or silently returning wrong cases) when both `combos` and `cases` are supplied alongside a `ds` whose internal dimension order differs from the case-keys-then-combo-keys insertion order. Per-variable indexers are now built in the variable's own dim order, and dims not present in `ds` are now treated as new coordinate locations rather than crashing.


(whats-new-1-3-3)=
## v1.3.3 (2026-04-30)

**Bug fixes:**

- Fix [cultivate](#farming.cultivate) silently ignoring `cases`.


(whats-new-1-3-2)=
## v1.3.2 (2026-04-30)

**Bug fixes:**

- Fix `xyzpy_grow` not being shipped in the built wheel/sdist after the v1.3.1 entry-point relocation, which caused `xyzpy-grow` to fail with `ModuleNotFoundError: No module named 'xyzpy_grow'` on the conda-forge feedstock CI. The hatchling build now uses `force-include` to ship the top-level `xyzpy_grow.py` alongside the `xyzpy` package.


(whats-new-1-3-1)=
## v1.3.1 (2026-04-29)

**Bug fixes:**

- Fix `xyzpy-grow --num-threads N` having no effect on numpy / BLAS / OpenMP threading. The CLI entry point lived inside the `xyzpy` package, so importing it ran `xyzpy/__init__.py` (which eagerly imports `xarray` / `numpy`) before `main()` could set `OMP_NUM_THREADS` etc. — the env vars were assigned too late. The entry point is now a top-level `xyzpy_grow` module, so the threading env vars land before any numerical library is imported. The recursive subprocess invocation in {meth}`~xyzpy.Crop.grow_subprocess` was updated to match.


(whats-new-1-3-0)=
## v1.3.0 (2026-03-30)

**New features:**

- Add {func}`~xyzpy.cultivate` for handling the entire crop lifecycle (annotate, sow, grow, reap) in one function.
- Add {meth}`~xyzpy.Crop.grow_subprocess` for running batches in isolated subprocesses with resource control — supports `gpus` (GPU device pooling via `CUDA_VISIBLE_DEVICES`), `affinities` (CPU pinning via `taskset`), `log` (save stdout/stderr per batch), `num_workers`, `num_threads`, `raise_errors`, and custom `batch_ids` ({issue}`20`)
- Add `xyzpy-grow` CLI entry point for driving {meth}`~xyzpy.Crop.grow_subprocess` from the command line
- Add {class}`~xyzpy.RayExecutor` and {class}`~xyzpy.RayGPUExecutor` for Ray-based parallel execution, also usable via `xyzpy-grow --ray`
- Add {func}`~xyzpy.plot.infiniplot.infiniplot` — a new unified plotting interface accessible via `ds.xyz.plot()` that auto-detects plot type (line, scatter, heatmap) from data dimensions - see {doc}`plotting` for details
- Add scatter plot support (`data_var` vs `data_var`) to `xyz.plot`
- Add {func}`~xyzpy.cmoke` (OKLCH-based) and {func}`~xyzpy.cimluv` (HSLuv-based) perceptually uniform single-hue colormap generators
- Add {class}`~xyzpy.MemoryMonitor` context manager for peak memory tracking, plus {func}`~xyzpy.get_peak_memory_usage`, {func}`~xyzpy.report_memory`, and {func}`~xyzpy.report_memory_gpu` utilities
- Add {func}`~xyzpy.visualize_tensor` for visualizing arbitrarily high dimensional tensors via 2D projections
- Add {func}`~xyzpy.format_number_with_error` for nicely formatting numbers with known errors

**Enhancements:**

- Add {func}`~xyzpy.parse_into_cases`, {func}`~xyzpy.find_missing_cases`, and {func}`~xyzpy.is_case_missing` for case-aware dataset filtering, with vectorized internals for large speedups on big parameter spaces
- Add {meth}`~xyzpy.Crop.load_batch`, {meth}`~xyzpy.Crop.load_result`, {meth}`~xyzpy.Crop.save_result` methods for direct batch data access
- Add {meth}`~xyzpy.Crop.delete_all` to cleanly remove a crop directory and reset object state
- Add `missing_only` option to {meth}`~xyzpy.Harvester.harvest_combos`
- Allow functions to return plain `dict`, including for mixed cases+combos
- Add `background_color`, `label`, `xticks`, `yticks`, `xticklabels`, `yticklabels` options to `xyz.plot`
- Allow `hlines`/`vlines` to be strings referencing `data_vars` so spans can vary by row and column
- Legend and colorbar improvements in `xyz.plot`
- Export {func}`~xyzpy.neutral_style` and {func}`~xyzpy.get_neutral_style` for matplotlib styling
- Make various SLURM header options optional in {meth}`~xyzpy.Crop.grow_cluster`
- Warn instead of erroring on non-picklable constants
- Catch confusing bug when combos involve duplicated values
- Move build to `pyproject.toml` with hatchling + hatch-vcs

**Bug fixes:**

- Fix saving of complex datasets with `engine='netcdf'` ({issue}`15`)
- Fix {meth}`~xyzpy.Crop.reap` with incomplete crops when the final batch has fewer items than `batchsize`
- Fix for recent `joblib` (`cachedir` → `location` kwarg)
- Fix Ellipsis handling bug in crop preparation


(whats-new-1-2-1)=
## v1.2.1 (12th August 2021)

**Bug fixes:**

- fix a few bugs related to crops and batchsizes


(whats-new-1-2-0)=
## v1.2.0 (12th August 2021)

**Enhancements**

- unified interface for mixing both `cases` and `combos`
- add random shuffling of growing order to load balance on batch systems etc
- add {meth}`~xyzpy.Harvester.expand_dims` and {meth}`~xyzpy.Harvester.drop_sel`


(whats-new-1-1-0)=
## v1.1.0 (25th July 2021)

**Enhancements**

- Defer `Crop.reap` clean up until *after* dataset sync (useful if you forget to set `overwrite=True`)
- Capture and print `Crop.grow_cluster` output
- add `visualize_matrix` tool
- add `cimple` colormap generator
- allow `@xyz.label` decorator to specify `harvester=`
- spruce docs

**Bug fixes:**

- Fix sowing, reaping and merging multiple sets of cases ({issue}`13`)
- Fix incomplete crop reaping when there is a non-zero batch remainder size
- Fix for futures that raise attribute errors themselves


(whats-new-1-0-0)=
## v1.0.0 (24th October 2020)

**Breaking changes**

- Remove all the data processing functionality which can now pretty much all be found in `xarray`. This also removes the `numba`, `scipy` and `cytoolz` dependencies completely.

**Enhancements**

- Generalize (and deprecate) {meth}`xyzpy.Crop.qsub_grow` to {meth}`xyzpy.Crop.grow_cluster` ({pull}`10`)
- Add SLURM support to {meth}`xyzpy.Crop.grow_cluster` ({pull}`10`)
- Add PBS support to {meth}`xyzpy.Crop.grow_cluster`
- Fix PBS crop submission for job arrays of size 1
- Add {func}`xyzpy.save_merge_ds` for manually aggregating datasets to disk
- Add `allow_incomplete=True` option to {meth}`xyzpy.Crop.reap` for gathering data even if the crop is not fully grown ({issue}`7`)
- Make new {class}`~xyzpy.Crop` instances by default automatically load information from disk if they have been already prepared/sown ({issue}`7`)
- Automatically load Crops in the current (or specified) directory with {func}`xyzpy.load_crops`.
- Add `'joblib'` and `'zarr'` as possible engines for saving and loading datasets
- Add utility {func}`xyzpy.getsizeof` to quite accurately get a python objects size
- Keep a running track of covariance using {class}`~xyzpy.utils.RunningCovariance`.


(whats-new-0-3-1)=
## v0.3.1 (25th January 2019)

**Bug fixes:**

- Make sure license is included in sdist/wheel distributions ({pull}`6`)


(whats-new-0-3-0)=
## v0.3.0 (21st January 2019)

**Breaking changes**

- Changed plot option `markersize` -> `marker_size` to match other keywords.

**Enhancements**

- New {class}`~xyzpy.Sampler` object - sparsely sample `combos` into a `pandas.DataFrame`
- Decorate functions directly into `Runner` instances using {func}`~xyzpy.label`


(whats-new-0-2-5)=
## v0.2.5 (3rd December 2018)

**Breaking changes**

- ({issue}`5`) `combo_runner` key argument `pool` renamed to `executor`

**Enhancements**

- ({issue}`5`) Support `multiprocessing.pool` in `combo_runner`
- Document timing and estimation utilities
- Use `loky` as the default parallel executor
- plotting: add `xjitter` and `yjitter`

**Bug fixes:**

- make sure `Crop._batch_remainder` synced with disk.
- update pytest marking parametrizations to xfail for recent pytest
- compatibility updates for dask and numba
- fix farming example which wasn't appearing


(whats-new-0-2-4)=
## v0.2.4 (1st November 2018)

**Bug fixes:**

- Various compatibility fixes for plotting functionality


(whats-new-0-2-3)=
## v0.2.3 (4th October 2018)

**Enhancements:**

- add {class}`~xyzpy.Timer`
- add {func}`~xyzpy.benchmark`
- add {class}`~xyzpy.Benchmarker`
- add {class}`~xyzpy.RunningStatistics`
- add {func}`~xyzpy.estimate_from_repeats`

**Bug fixes:**

- various fixes to batch growing and {class}`~xyzpy.Crop`
- various fixes to plotting


(whats-new-0-2-2)=
## v0.2.2 (7th June 2018)

**Enhancements:**

- allow `case_runner` to return `Dataset`

**Bug fixes:**

- ({issue}`1`) make `numba` an optional dependency


(whats-new-0-2-1)=
## v0.2.1 (27th May 2018)

**Bug fixes:**

- docs updates
- distribute crop batches more evenly
