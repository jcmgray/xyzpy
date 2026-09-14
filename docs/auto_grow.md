# Growing crops automatically

`xyzpy-auto-grow` watches one directory for `.xyz-*` [`Crop`](#cropping.Crop)
directories. It grows missing batches from all Crops with one worker pool. It
takes one batch from each Crop in turn. This lets a new Crop start before an
older Crop finishes. See
[Batched / Distributed generation](computing_results.ipynb) for Crops
themselves, and the other ways to grow them.

## Start the watcher

Run the command in the directory that contains the crops:

```bash
xyzpy-auto-grow --num-workers 8 --num-threads 4
```

Each worker starts a new process for one batch. `--num-workers` sets the number
of batches that can run at once. `--num-threads` sets the BLAS, OpenMP,
NumExpr, and Numba thread limits in each process. These controls match
[`Crop.grow_subprocess`](#Crop.grow_subprocess).

GPU and CPU pools can also limit how many batches run at once:

```bash
xyzpy-auto-grow --num-workers 8 --gpus 0,0,1,1
xyzpy-auto-grow --num-workers 8 --affinities 0,1,2,3
```

Each GPU ID or CPU affinity is one worker slot. Repeat a GPU ID to let more
than one worker use that device. Batch output goes to
`.xyz-NAME/logs/batch-ID.log` by default.

## Sow from a notebook

Use [`sow`](#farming.sow) to write batches without growing or reaping them:

```python
import xyzpy as xyz


def calculate(x, scale=2):
    return scale * x


crop_a = xyz.sow(
    calculate,
    var_names="value",
    data_name="results.h5",
    combos={"x": range(10)},
    batchsize=2,
)
crop_b = xyz.sow(
    calculate,
    var_names="value",
    data_name="results.h5",
    combos={"x": range(10, 20)},
    batchsize=2,
)
```

One watcher in the notebook directory grows both Crops. Use
[`Crop.reap`](#Crop.reap) later to add their results to `results.h5`:

```python
for crop in (crop_a, crop_b):
    if crop is not None:
        crop.reap(wait=True)
```

:::{note}
The watcher grows batches. It does not call
[`Crop.reap`](#Crop.reap) or change a dataset.
:::

Use [`Harvester.sow`](#Harvester.sow) with `combos=...` or `cases=...` when
you already have a [`Harvester`](#farming.Harvester). It parses combos and
cases like [`cultivate`](#farming.cultivate). A value of `...` uses the current
values of that coordinate. With `missing_only=True`, it skips cases that are
already in the dataset. It does not check other pending Crops.

A new request returns `None` when all its cases are already in the dataset.
A matching Crop is returned even when it has no missing work.
`missing_only=False` only affects a new Crop. Give a new `name` to submit the
same request again.

Use [`Crop.reap`](#Crop.reap) with `overwrite=True` to replace conflicting
values in the dataset. A failed reap keeps the Crop, so you can retry it.

## Crop names and reuse

The default crop name contains the function name and a stable request key. The
key includes the complete coordinate request and the absolute dataset path. It
is made before missing cases are removed. Input order does not affect it.
Constants used as output coordinates do affect it. Moving the dataset to a new
absolute path changes the key.

The key does not include function source, non-coordinate constants, resources,
attributes, batch settings, or shuffle settings. When a matching crop exists,
the current function replaces its saved function. Supplied constants update
batches without results when their values change. Unchanged constants leave
batch files untouched and held failures remain held. Omitted constants keep
their saved values. Completed batch files and results stay unchanged, as do
batch order and saved Harvester settings. Running batches that already loaded
their inputs can use the old values.

Shared dataset attributes reflect the latest constants, even when some results
were computed with earlier values. Existing result files keep completed
batches out of the grower queue.

An explicit name cannot refer to a different coordinate request. Old crops
without a request key are not reused. Incomplete crops are not overwritten.
Remove an incomplete crop directory or use a new name.

New Crops are prepared in a hidden temporary directory. A failed preparation
leaves the final name free for another attempt. The watcher only sees the Crop
after all batch files and metadata are ready. At this point,
[`Crop.is_prepared`](#Crop.is_prepared) returns `True`.

## Dataset state

For a disk-backed [`Harvester`](#farming.Harvester), the file on disk is the
source of truth. Accessing [`Harvester.full_ds`](#Harvester.full_ds) reloads it
after its file or Zarr store changes. A reload discards unsaved in-memory
edits. Normal harvest operations save their changes. Calling
[`Harvester.add_ds`](#Harvester.add_ds) with `sync=False` makes a temporary
in-memory change.

A reused in-memory Crop adds results to the Harvester that called
[`Harvester.sow`](#Harvester.sow). It still uses the output metadata saved
with the Crop.

Use [`Crop.reap`](#Crop.reap) with `allow_incomplete=True` to add available
results and keep the Crop for its remaining batches.

## Change settings while running

The watcher reads `.xyzpy-auto-grow.toml` from the watched directory. The file
is optional. Its values override command-line values. You can create, change,
or remove it while the watcher runs.

```toml
num_workers = 8
num_threads = 4
gpus = [0, 0, 1, 1]
log = true
```

The file accepts these keys: `num_workers`, `num_threads`, `gpus`,
`affinities`, `raise_errors`, `log`, `min_wait`, `max_wait`,
`verbosity`, `verbosity_grow`, `scan_interval`, `refresh_interval`, and
`desc`.

An empty GPU or affinity list disables that pool. Set `num_workers = 0` to
pause new batches.

:::{note}
New settings apply only to new batches. Running batches keep their settings.
When you enable a GPU or affinity pool, the watcher first waits for active
batches without that assignment to finish.
:::

An invalid file is reported. The last valid settings remain active.

## Progress, failures, and stopping

In a terminal, the watcher shows the number of complete, queued, running, and
failed batches for each crop. It also shows active resource assignments and
recent failures. Redirected output writes one line for each event.

A failed batch is held for the rest of the watcher session. Other batches
continue. The next scan clears the held failure if you call
[`Crop.delete_all`](#Crop.delete_all) or replace its batch file. Restarting the
watcher also retries missing batches.

With `raise_errors = true`, the first failure stops new batches. The watcher
waits for active batches, then exits with an error.

The first Ctrl-C stops new batches and waits for active batches. A second
Ctrl-C terminates them. Only one watcher can manage a directory at a time.

:::{warning}
Stop the watcher and wait for active batches before
[`Crop.grow`](#Crop.grow) or another grower changes the same Crops.
:::

Use `--once` to grow all work that can start now, then exit. If launches are
paused or no configured resource slot is available, it exits without removing
queued batches.
