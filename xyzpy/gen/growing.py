"""Start and monitor crop batch processes."""

import os
import re
import shutil
import signal
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


_MEMORY_RE = re.compile(r"([0-9]*\.?[0-9]+)\s*([kmgt]?)(?:i?b)?")


def _parse_memory(raw):
    """Parse a memory size such as ``"100G"`` or ``"512mb"`` as bytes.

    Units are powers of 1024. A plain number is bytes.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        raise ValueError("max_memory must be a size such as '100G'.")
    if isinstance(raw, int):
        nbytes = raw
    else:
        match = _MEMORY_RE.fullmatch(str(raw).strip().lower())
        if match is None:
            raise ValueError(f"Invalid memory size {raw!r}, use e.g. '100G'.")
        number, unit = match.groups()
        nbytes = int(float(number) * 1024 ** "_kmgt".index(unit or "_"))
    if nbytes <= 0:
        raise ValueError("max_memory must be greater than zero.")
    return nbytes


def _format_memory(nbytes):
    """Format a number of bytes with a unit, such as ``"1.5G"``."""
    unit = ""
    for next_unit in "KMGT":
        if nbytes < 1024:
            break
        nbytes /= 1024
        unit = next_unit
    return f"{nbytes:.1f}{unit}"


def _process_tree(pid):
    """Return a process and all its descendants, read from Linux /proc."""
    pids = [pid]
    for parent in pids:
        for children in Path(f"/proc/{parent}/task").glob("*/children"):
            try:
                pids.extend(map(int, children.read_text().split()))
            except OSError:
                # the thread or process has already exited
                pass
    return pids


def _memory_usage(pids):
    """Return the total resident memory of some processes, in bytes."""
    # pages shared between processes are counted once for each
    page_size = os.sysconf("SC_PAGE_SIZE")
    nbytes = 0
    for pid in pids:
        try:
            with open(f"/proc/{pid}/statm") as f:
                nbytes += int(f.read().split()[1]) * page_size
        except OSError:
            pass
    return nbytes


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
    max_memory: int | None
    memory_used: int | None = None


class _SubprocessRunner:
    """Start and monitor batch processes with resource limits."""

    def __init__(
        self,
        *,
        num_workers=1,
        num_threads=1,
        gpus=None,
        affinities=None,
        max_memory=None,
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
            max_memory=max_memory,
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
        max_memory,
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
        self.max_memory = _parse_memory(max_memory)
        if self.max_memory and not os.path.exists(
            f"/proc/self/task/{os.getpid()}/children"
        ):
            raise ValueError(
                "A memory limit needs Linux, to read process memory from /proc."
            )
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
            max_memory=self.max_memory,
        )

    def poll(self):
        """Return the batches that have finished."""
        completed = []
        for key, active in tuple(self.active.items()):
            if active.max_memory is not None and active.memory_used is None:
                pids = _process_tree(active.process.pid)
                memory_used = _memory_usage(pids)
                if memory_used > active.max_memory:
                    active.memory_used = memory_used
                    for pid in pids:
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass

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
            if active.memory_used is not None and not success:
                message = (
                    "process was killed for using "
                    f"{_format_memory(active.memory_used)}, over the memory "
                    f"limit of {_format_memory(active.max_memory)}"
                )
            elif returncode != 0:
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
