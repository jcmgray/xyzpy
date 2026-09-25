"""Grow batches from all crops in one directory."""

import argparse
import os
import signal
import sys
import time
import tomllib
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from .cropping import BTCH_RE, FNCT_NM, INFO_NM, RSLT_RE
from .growing import (
    _BatchTask,
    _parse_memory,
    _parse_resource_ids,
    _SubprocessRunner,
)


def _file_stamp(path):
    """Return a file stamp, or None when the path does not exist."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return (stat.st_dev, stat.st_ino, stat.st_mtime_ns, stat.st_size)


CONFIG_KEYS = {
    "num_workers",
    "num_threads",
    "gpus",
    "affinities",
    "max_memory",
    "raise_errors",
    "log",
    "min_wait",
    "max_wait",
    "verbosity",
    "verbosity_grow",
    "scan_interval",
    "refresh_interval",
    "desc",
}


def _failure_stamp(task):
    """Return the input stamp that must change before a task is retried."""
    return (
        _file_stamp(task.batch_file),
        _file_stamp(task.crop_dir / FNCT_NM),
    )


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes"):
        return True
    if value in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value {value!r}.")


def _validate_settings(settings):
    unknown = set(settings) - CONFIG_KEYS
    if unknown:
        names = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown settings: {names}.")

    validated = dict(settings)
    for name, minimum in (("num_workers", 0), ("num_threads", 1)):
        value = validated[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer.")
        if value < minimum:
            raise ValueError(f"{name} must be at least {minimum}.")

    for name in ("verbosity", "verbosity_grow"):
        value = validated[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be zero or greater.")

    for name in ("min_wait", "max_wait", "scan_interval", "refresh_interval"):
        value = validated[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number.")
        if value <= 0:
            raise ValueError(f"{name} must be greater than zero.")
        validated[name] = float(value)
    if validated["min_wait"] > validated["max_wait"]:
        raise ValueError("min_wait cannot be greater than max_wait.")

    for name in ("raise_errors", "log"):
        if not isinstance(validated[name], bool):
            raise TypeError(f"{name} must be true or false.")

    for name in ("gpus", "affinities"):
        try:
            value = _parse_resource_ids(validated[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain integer IDs.") from exc
        validated[name] = value or None

    # check only, keep the value as given so the display shows e.g. '100G'
    _parse_memory(validated["max_memory"])

    if not isinstance(validated["desc"], str):
        raise TypeError("desc must be a string.")
    return validated


class ConfigFile:
    """Reload a TOML settings file when it changes."""

    _UNREAD = object()

    def __init__(self, path, base_settings):
        self.path = Path(path)
        self.base_settings = dict(base_settings)
        self.stamp = self._UNREAD

    def _stamp(self):
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return None
        return (stat.st_mtime_ns, stat.st_size)

    def read(self):
        """Return ``(changed, settings, error)`` after checking the file."""
        stamp = self._stamp()
        if stamp == self.stamp:
            return False, None, None
        self.stamp = stamp

        try:
            if stamp is None:
                overrides = {}
            else:
                with self.path.open("rb") as file:
                    overrides = tomllib.load(file)
                if not isinstance(overrides, dict):
                    raise ValueError(
                        "The settings file must contain one TOML table."
                    )
            unknown = set(overrides) - CONFIG_KEYS
            if unknown:
                names = ", ".join(sorted(unknown))
                raise ValueError(f"Unknown settings: {names}.")
            settings = dict(self.base_settings)
            settings.update(overrides)
            return True, _validate_settings(settings), None
        except (OSError, tomllib.TOMLDecodeError, ValueError) as exc:
            return True, None, str(exc)


class DirectoryLock:
    """Allow only one watcher in a directory."""

    def __init__(self, directory):
        self.path = Path(directory) / ".xyzpy-auto-grow.lock"
        self.file = None

    def __enter__(self):
        self.file = self.path.open("a+")
        try:
            if os.name == "nt":
                import msvcrt

                self.file.seek(0)
                if self.file.read(1) == "":
                    self.file.write(" ")
                    self.file.flush()
                self.file.seek(0)
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            self.file.close()
            self.file = None
            raise RuntimeError(
                f"xyzpy-auto-grow is already watching {self.path.parent}."
            ) from exc

        self.file.seek(0)
        self.file.truncate()
        self.file.write(f"{os.getpid()}\n")
        self.file.flush()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.file is None:
            return
        if os.name == "nt":
            import msvcrt

            self.file.seek(0)
            msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
        self.file.close()


@dataclass
class CropState:
    name: str
    path: Path
    batch_ids: tuple[int, ...]
    result_ids: set[int]
    pending_ids: deque[int]


class AutoGrower:
    """Find crop batches and start them in round-robin order."""

    def __init__(self, parent_dir, settings, config_path, once=False):
        self.parent_dir = Path(parent_dir).expanduser().resolve()
        self.settings = _validate_settings(settings)
        self.config = ConfigFile(config_path, self.settings)
        self.once = once
        self.runner = _SubprocessRunner(
            num_workers=self.settings["num_workers"],
            num_threads=self.settings["num_threads"],
            gpus=self.settings["gpus"],
            affinities=self.settings["affinities"],
            max_memory=self.settings["max_memory"],
            log=self.settings["log"],
            verbosity_grow=self.settings["verbosity_grow"],
            append_logs=True,
            raise_errors=True,
        )
        self.crops = []
        self.cursor = 0
        self.failed = {}
        self.recent_failures = []
        self.config_error = None
        self.stopping = False
        self.stop_error = None
        self._last_event = "starting"
        self._last_scan = 0.0
        self._last_refresh = 0.0

    def _event(self, message):
        self._last_event = message
        if self.settings["verbosity"] and not sys.stdout.isatty():
            print(f"xyzpy-auto-grow: {message}", flush=True)

    def _apply_config(self, initial=False):
        changed, settings, error = self.config.read()
        if not changed:
            return
        if error is not None:
            if initial:
                raise ValueError(f"Invalid settings: {error}")
            self.config_error = error
            self._event(f"settings error, keeping previous values: {error}")
            return

        self.config_error = None
        self.settings = settings
        self.runner.configure(
            num_workers=settings["num_workers"],
            num_threads=settings["num_threads"],
            gpus=settings["gpus"],
            affinities=settings["affinities"],
            max_memory=settings["max_memory"],
            log=settings["log"],
            verbosity_grow=settings["verbosity_grow"],
        )
        if not initial:
            self._event("settings reloaded")

    def scan(self):
        """Refresh crops, queues, and held failures from disk."""
        crops = []
        seen = set()
        batch_stamps = {}
        active = self.runner.active
        for path in sorted(self.parent_dir.glob(".xyz-*")):
            if not path.is_dir():
                continue
            if not (path / INFO_NM).is_file():
                continue
            if not (path / FNCT_NM).is_file():
                continue

            name = path.name[5:]
            path = path.resolve()
            fn_stamp = _file_stamp(path / FNCT_NM)
            if path in seen:
                continue
            # treat links to the same directory as one crop
            seen.add(path)
            batch_dir = path / "batches"
            result_dir = path / "results"
            batch_ids = []
            if batch_dir.is_dir():
                try:
                    batch_files = tuple(batch_dir.iterdir())
                except FileNotFoundError:
                    continue
                for file in batch_files:
                    match = BTCH_RE.fullmatch(file.name)
                    if match and file.is_file():
                        batch_id = int(match.group(1))
                        stamp = _file_stamp(file)
                        if stamp is not None:
                            batch_ids.append(batch_id)
                            batch_stamps[(str(path), batch_id)] = (
                                stamp,
                                fn_stamp,
                            )
            result_ids = set()
            if result_dir.is_dir():
                try:
                    result_files = tuple(result_dir.iterdir())
                except FileNotFoundError:
                    continue
                for file in result_files:
                    match = RSLT_RE.fullmatch(file.name)
                    if match and file.is_file():
                        result_ids.add(int(match.group(1)))

            crops.append(
                CropState(
                    name=name,
                    path=path,
                    batch_ids=tuple(sorted(batch_ids)),
                    result_ids=result_ids,
                    pending_ids=deque(),
                )
            )

        # retry failed batches whose input file or function changed
        self.failed = {
            key: stamp
            for key, stamp in self.failed.items()
            if key in batch_stamps and batch_stamps[key] == stamp
        }
        for crop in crops:
            prefix = str(crop.path)
            crop.pending_ids.extend(
                batch_id
                for batch_id in crop.batch_ids
                if batch_id not in crop.result_ids
                and (prefix, batch_id) not in active
                and (prefix, batch_id) not in self.failed
            )
        self.crops = crops
        self._apply_config()

    def has_pending(self):
        """Return True if any crop has a queued batch."""
        return any(crop.pending_ids for crop in self.crops)

    def next_task(self):
        """Remove and return the next batch in round-robin order."""
        if not self.crops:
            return None
        for offset in range(len(self.crops)):
            index = (self.cursor + offset) % len(self.crops)
            crop = self.crops[index]
            if crop.pending_ids:
                self.cursor = (index + 1) % len(self.crops)
                return _BatchTask(
                    crop_name=crop.name,
                    parent_dir=self.parent_dir,
                    batch_id=crop.pending_ids.popleft(),
                )
        return None

    def _handle_completions(self):
        for completion in self.runner.poll():
            key = completion.task.key
            label = f"{completion.task.crop_name}:{completion.task.batch_id}"
            if completion.success:
                self._event(f"completed {label}")
                continue

            self.failed[key] = _failure_stamp(completion.task)
            message = f"failed {label}: {completion.message}"
            if completion.log_path is not None:
                message += f". Log: {completion.log_path}"
            self.recent_failures.append(message)
            self.recent_failures = self.recent_failures[-5:]
            self._event(message)
            if self.settings["raise_errors"]:
                self.stopping = True
                self.stop_error = RuntimeError(message)

    def _launch_available(self):
        while not self.stopping and self.runner.can_submit():
            task = self.next_task()
            if task is None:
                return
            if not task.batch_file.is_file() or task.result_file.is_file():
                # files changed after the scan, rebuild the queue
                self.scan()
                return
            try:
                self.runner.submit(task)
                self._event(f"started {task.crop_name}:{task.batch_id}")
            except Exception as exc:  # noqa: BLE001
                self.failed[task.key] = _failure_stamp(task)
                message = (
                    f"failed to start {task.crop_name}:{task.batch_id}: {exc}"
                )
                self.recent_failures.append(message)
                self.recent_failures = self.recent_failures[-5:]
                self._event(message)
                if self.settings["raise_errors"]:
                    self.stopping = True
                    self.stop_error = RuntimeError(message)

    def _render(self):
        if not self.settings["verbosity"] or not sys.stdout.isatty():
            return
        gpu_text = (
            "any" if self.settings["gpus"] is None else self.settings["gpus"]
        )
        affinity_text = (
            "any"
            if self.settings["affinities"] is None
            else self.settings["affinities"]
        )
        lines = [
            f"{self.settings['desc']}  {self.parent_dir}",
            (
                f"workers={self.settings['num_workers']} "
                f"threads={self.settings['num_threads']} "
                f"gpus={gpu_text} affinities={affinity_text} "
                f"max_memory={self.settings['max_memory'] or 'any'}"
            ),
            "",
            "crop                     done  queued  running  failed",
        ]
        for crop in self.crops:
            prefix = str(crop.path)
            running = sum(key[0] == prefix for key in self.runner.active)
            failed = sum(key[0] == prefix for key in self.failed)
            queued = len(crop.pending_ids)
            done = len(crop.result_ids)
            lines.append(
                f"{crop.name[:24]:24} {done:5} {queued:7} {running:8} {failed:7}"
            )
        if not self.crops:
            lines.append("waiting for sown crops")

        if self.runner.active:
            lines.extend(("", "active batches"))
            now = time.monotonic()
            for active in self.runner.active.values():
                resources = []
                if active.gpu is not None:
                    resources.append(f"gpu={active.gpu}")
                if active.affinity is not None:
                    resources.append(f"cpu={active.affinity}")
                suffix = f" ({', '.join(resources)})" if resources else ""
                elapsed = now - active.started
                lines.append(
                    f"  {active.task.crop_name}:{active.task.batch_id} "
                    f"{elapsed:.1f}s{suffix}"
                )
        if self.recent_failures:
            lines.extend(("", "recent failures", *self.recent_failures))
        if self.config_error:
            lines.extend(("", f"settings error: {self.config_error}"))
        lines.extend(("", self._last_event, "Ctrl-C: stop and wait"))
        print("\033[2J\033[H" + "\n".join(lines), end="", flush=True)

    def run(self):
        """Watch until stopped or a one-shot run cannot start more work."""
        old_sigterm = signal.getsignal(signal.SIGTERM)

        def handle_sigterm(signum, frame):
            self.stopping = True
            self._event("SIGTERM received, waiting for active batches")

        try:
            self._apply_config(initial=True)
            self.scan()
            self._event(f"watching {self.parent_dir}")
            signal.signal(signal.SIGTERM, handle_sigterm)
            dt = self.settings["min_wait"]
            while True:
                try:
                    now = time.monotonic()
                    self._handle_completions()
                    if now - self._last_scan >= self.settings["scan_interval"]:
                        self.scan()
                        self._last_scan = now
                    self._launch_available()
                    if (
                        now - self._last_refresh
                        >= self.settings["refresh_interval"]
                    ):
                        self._render()
                        self._last_refresh = now

                    if self.stopping and not self.runner.active:
                        break
                    # a paused one-shot run leaves its queue on disk
                    if (
                        self.once
                        and not self.runner.active
                        and (
                            not self.has_pending()
                            or not self.runner.can_submit()
                        )
                    ):
                        break

                    if self.runner.active:
                        time.sleep(dt)
                        dt = min(1.2 * dt, self.settings["max_wait"])
                    else:
                        time.sleep(min(self.settings["scan_interval"], 0.2))
                        dt = self.settings["min_wait"]
                except KeyboardInterrupt:
                    if self.stopping:
                        self._event("terminating active batches")
                        self.runner.terminate()
                        return 130
                    self.stopping = True
                    self._event(
                        "stopping new batches and waiting for active batches"
                    )
        finally:
            self.runner.terminate()
            signal.signal(signal.SIGTERM, old_sigterm)
            self._render()
        if self.stop_error is not None:
            raise self.stop_error
        return 0


def build_parser():
    parser = argparse.ArgumentParser(
        description="Grow all sown xyzpy crops in one directory."
    )
    parser.add_argument("--parent-dir", default=".")
    parser.add_argument("--config", default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--affinities", default=None)
    parser.add_argument("--max-memory", default=None)
    parser.add_argument(
        "--raise-errors",
        nargs="?",
        const=True,
        default=False,
        type=_parse_bool,
    )
    parser.add_argument(
        "--log", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--min-wait", type=float, default=0.01)
    parser.add_argument("--max-wait", type=float, default=0.2)
    parser.add_argument("--scan-interval", type=float, default=2.0)
    parser.add_argument("--refresh-interval", type=float, default=1.0)
    parser.add_argument("--verbosity", type=int, default=1)
    parser.add_argument("--verbosity-grow", type=int, default=0)
    parser.add_argument("--desc", default="xyzpy auto grow")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Grow work that can start now, then exit.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    parent_dir = Path(args.parent_dir).expanduser().resolve()
    if not parent_dir.is_dir():
        raise ValueError(f"Directory does not exist: {parent_dir}")
    config_path = (
        parent_dir / ".xyzpy-auto-grow.toml"
        if args.config is None
        else Path(args.config).expanduser().resolve()
    )
    settings = {name: getattr(args, name) for name in CONFIG_KEYS}
    with DirectoryLock(parent_dir):
        return AutoGrower(
            parent_dir=parent_dir,
            settings=settings,
            config_path=config_path,
            once=args.once,
        ).run()


if __name__ == "__main__":
    sys.exit(main())
