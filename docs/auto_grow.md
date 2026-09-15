# Growing crops automatically

Use the CLI `xyzpy-auto-grow` to watch a directory for *any* `.xyz-*`
[`Crop`](#cropping.Crop) directories and automatically grow missing batches.
The batches are taken round-robin fashion from the crops, so that no one crop
blocks all others. See
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
# in both cases here num_workers is effectively 4
```

Each GPU ID or CPU affinity is one worker slot. Repeat a GPU ID to let more
than one worker use that device. Batch output goes to
`.xyz-NAME/logs/batch-ID.log` by default.


## Sow from a notebook

A useful companion to the watcher is [`xyz.sow`](#farming.sow) or
[`Harvester.sow`](#Harvester.sow). These take a function, desired set of
`combos` and/or `cases` and a dataset name, parse out the missing cases only
(by default), and create a uniquely named `Crop` in the current directory.
One can thus interactively `sow` and watch progress in a notebook unblocked,
while the `xyzpy-auto-grow` watcher generates results in a terminal for
example.

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
The watcher just grows batches. It does not call [`Crop.reap`](#Crop.reap) or
update any datasets.
:::

Use [`Harvester.sow`](#Harvester.sow) with `combos=...` or `cases=...` when
you already have a [`Harvester`](#farming.Harvester). It parses combos and
cases like [`cultivate`](#farming.cultivate). A value of `...` uses the current
values of that coordinate. With the default `missing_only=True`, it skips cases
that are already in the dataset (but it does not check other pending Crops
yet, and only takes affect on first sow).

If you call `sow` with equivalent function, combos, cases, and dataset name, it
returns the same `Crop` as before, or `None` if all cases are already present
in the dataset. You can use an explicit `name=` to create a separate Crop with
identical work if necessary.

Use [`Crop.reap`](#Crop.reap) with `overwrite=True` to replace conflicting
values in the dataset. A failed reap keeps the Crop, so you can retry it.


## Crop names and reuse

The default crop name contains the function name and a stable 'request key'
identifying the set of results being generated. The key includes the complete
coordinate request and the absolute dataset path. It is made before missing
cases are removed. Input order does not affect it. Constants used as
*output coordinates* do affect it. Moving the dataset to a new absolute path
also changes the key.

Everthing else - function source, non-coordinate constants, resources,
attributes, batch settings, and shuffle settings - do not affect the key and
should be modified with care when reusing the same Crop. If you do resow, only
the function will updated on disk, and any new `constants` will also be written
into *remaining* batches. **Completed results are untouched**. Any result files
also keep completed batches out of the `xyzpy-auto-grow` queue.

```{hint}
Re-sowing is useful if you make some optimization or bug fix to the function,
and simply want missing results to be generated with this new implementation.
```
```{note}
*Other notes:*

- An explicit `name` cannot refer to a *different* coordinate request.
- Old-style crops without a 'request key' are not reused.
- Incomplete crops are never overwritten.
- Remove an incomplete crop directory or use a new `name`.
- New Crops are prepared in a hidden temporary directory so a failed
  preparation leaves the final name free for another attempt. The watcher only
  sees the Crop after all batch files and metadata are ready. At which point,
  [`Crop.is_prepared`](#Crop.is_prepared) returns `True`.
```


## Dataset state

Note for a disk-backed [`Harvester`](#farming.Harvester), the file on disk is
the source of truth. Accessing [`Harvester.full_ds`](#Harvester.full_ds)
reloads it after its file or Zarr store changes. A reload discards unsaved
in-memory edits. Normal harvest operations save their changes. Calling
[`Harvester.add_ds`](#Harvester.add_ds) with `sync=False` makes a temporary
in-memory change.

A reused in-memory Crop adds results to the Harvester that called
[`Harvester.sow`](#Harvester.sow). I.e. it always uses the initial output
metadata saved with the Crop.

Use [`Crop.reap`](#Crop.reap) with `allow_incomplete=True` to add available
results and keep the Crop for its remaining batches.


## Change settings while running

The watcher reads `.xyzpy-auto-grow.toml` from the watched directory. The file
is optional. Its values override command-line values. You can create, change,
or remove it while the watcher runs.

```toml
num_workers = 8
num_threads = 8
gpus = [0, 0, 1, 1, 2, 2, 3, 3]
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

An invalid file is reported and last *valid settings* remain active.


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
