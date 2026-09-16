"""Start and monitor crop batch processes."""

import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from subprocess import DEVNULL, Popen

_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
)


def _parse_resource_ids(raw):
    """Parse resource IDs as integers."""
    if raw is None:
        return None
    if isinstance(raw, int):
        return [raw]
    if isinstance(raw, str):
        if not raw.strip():
            return []
        raw = raw.split(",")
    return list(map(int, raw))


def _acquire_affinity(resource_id, args, env):
    """Add the command that selects one CPU core."""
    args[0:0] = ["taskset", "-c", str(resource_id)]


def _acquire_gpu(resource_id, args, env):
    """Select one GPU for the process."""
    env["CUDA_VISIBLE_DEVICES"] = str(resource_id)


@dataclass(frozen=True)
class _BatchTask:
    """One crop batch to run."""

    crop_name: str
    parent_dir: Path
    batch_id: int

    @property
    def key(self):
        return (str(self.crop_dir.resolve()), self.batch_id)

    @property
    def crop_dir(self):
        return self.parent_dir / f".xyz-{self.crop_name}"

    @property
    def batch_file(self):
        return self.crop_dir / "batches" / f"xyz-batch-{self.batch_id}.jbdmp"

    @property
    def result_file(self):
        return self.crop_dir / "results" / f"xyz-result-{self.batch_id}.jbdmp"

    @property
    def log_file(self):
        return self.crop_dir / "logs" / f"batch-{self.batch_id}.log"


@dataclass
class _BatchCompletion:
    """The result of one batch process."""

    task: _BatchTask
    success: bool
    returncode: int
    message: str
    log_path: Path | None


@dataclass
class _ActiveBatch:
    task: _BatchTask
    process: Popen
    output_file: object
    log_path: Path | None
    started: float
    gpu: int | None
    affinity: int | None


class _SubprocessRunner:
    """Start and monitor batch processes with resource limits."""

    def __init__(
        self,
        *,
        num_workers=1,
        num_threads=1,
        gpus=None,
        affinities=None,
        log=False,
        verbosity_grow=0,
        append_logs=False,
        raise_errors=True,
    ):
        self.active = {}
        self.configure(
            num_workers=num_workers,
            num_threads=num_threads,
            gpus=gpus,
            affinities=affinities,
            log=log,
            verbosity_grow=verbosity_grow,
        )
        self.append_logs = append_logs
        self.raise_errors = raise_errors

    def configure(
        self,
        *,
        num_workers,
        num_threads,
        gpus,
        affinities,
        log,
        verbosity_grow,
    ):
        """Set limits for new processes."""
        self.num_workers = num_workers
        self.num_threads = num_threads
        self.gpus = _parse_resource_ids(gpus)
        self.affinities = _parse_resource_ids(affinities)
        if self.affinities and shutil.which("taskset") is None:
            raise ValueError("CPU affinity needs the Linux `taskset` command.")
        self.log = log
        self.verbosity_grow = verbosity_grow

    def _free_slots(self, name):
        configured = getattr(self, name)
        if configured is None:
            return None

        assignment = {"gpus": "gpu", "affinities": "affinity"}[name]
        free = list(configured)
        for active in self.active.values():
            resource_id = getattr(active, assignment)
            if resource_id is None:
                # wait for work started before this pool was enabled
                return []
            try:
                free.remove(resource_id)
            except ValueError:
                pass
        return free

    def can_submit(self):
        if len(self.active) >= self.num_workers:
            return False
        for name in ("gpus", "affinities"):
            free = self._free_slots(name)
            if free is not None and not free:
                return False
        return True

    def submit(self, task):
        """Start one batch with the current settings."""
        if task.key in self.active:
            raise ValueError(f"Batch {task.key} is already running.")
        if not self.can_submit():
            raise RuntimeError("No worker slot is available.")

        gpu_slots = self._free_slots("gpus")
        affinity_slots = self._free_slots("affinities")
        gpu = None if gpu_slots is None else gpu_slots[-1]
        affinity = None if affinity_slots is None else affinity_slots[-1]

        args = [
            sys.executable,
            "-m",
            "xyzpy_grow",
            task.crop_name,
            "--parent-dir",
            str(task.parent_dir),
            "--batch-ids",
            str(task.batch_id),
            "--num-threads",
            str(self.num_threads),
            # the child grows its single batch here, in itself
            "--subprocess",
            "false",
            "--verbosity",
            "0",
            "--verbosity-grow",
            str(self.verbosity_grow),
        ]
        if self.raise_errors:
            args.append("--raise-errors")
        env = os.environ.copy()
        for name in _THREAD_ENV_VARS:
            env[name] = str(self.num_threads)
        if gpu is not None:
            _acquire_gpu(gpu, args, env)
        if affinity is not None:
            _acquire_affinity(affinity, args, env)

        if self.log:
            task.log_file.parent.mkdir(exist_ok=True)
            mode = "a" if self.append_logs else "w"
            output_file = open(task.log_file, mode, encoding="utf-8")  # noqa: SIM115
            if self.append_logs:
                stamp = time.strftime("%Y-%m-%d %H:%M:%S")
                output_file.write(
                    f"\n=== attempt {stamp}, batch {task.batch_id} ===\n"
                )
                output_file.flush()
            stdout = output_file
            stderr = output_file
            log_path = task.log_file
        else:
            output_file = tempfile.TemporaryFile(mode="w+t")  # noqa: SIM115
            stdout = DEVNULL
            stderr = output_file
            log_path = None

        try:
            process = Popen(
                args,
                stdout=stdout,
                stderr=stderr,
                text=True,
                env=env,
            )
        except BaseException:
            output_file.close()
            raise

        self.active[task.key] = _ActiveBatch(
            task=task,
            process=process,
            output_file=output_file,
            log_path=log_path,
            started=time.monotonic(),
            gpu=gpu,
            affinity=affinity,
        )

    def poll(self):
        """Return the batches that have finished."""
        completed = []
        for key, active in tuple(self.active.items()):
            returncode = active.process.poll()
            if returncode is None:
                continue

            del self.active[key]
            if active.log_path is None:
                active.output_file.seek(0)
                stderr = active.output_file.read().strip()
            else:
                stderr = ""
            active.output_file.close()

            result_exists = active.task.result_file.is_file()
            success = returncode == 0 and result_exists
            if returncode != 0:
                message = f"process exited with status {returncode}"
            elif not result_exists:
                message = "process exited without writing a result"
            else:
                message = "completed"
            if stderr:
                message = f"{message}: {stderr}"

            completed.append(
                _BatchCompletion(
                    task=active.task,
                    success=success,
                    returncode=returncode,
                    message=message,
                    log_path=active.log_path,
                )
            )
        return completed

    def terminate(self):
        """Stop all active processes and close their output files."""
        for active in self.active.values():
            active.process.terminate()
        for active in self.active.values():
            try:
                active.process.wait(timeout=5)
            except Exception:  # noqa: BLE001
                active.process.kill()
                active.process.wait()
            active.output_file.close()
        self.active.clear()
