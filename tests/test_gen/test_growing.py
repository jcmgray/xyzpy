import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import xyzpy as xyz
from xyzpy.gen import growing
from xyzpy.gen.growing import _parse_memory, _SubprocessRunner

from ..test_auto_grow import add_one


def allocate(nbytes):
    # use nonzero bytes, so the memory is really used
    return len(b"x" * nbytes)


def can_watch_memory():
    return os.path.exists(f"/proc/self/task/{os.getpid()}/children")


class TestParseMemory:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            (None, None),
            (1000, 1000),
            ("1000", 1000),
            ("4k", 4 * 1024),
            ("512M", 512 * 1024**2),
            ("512mb", 512 * 1024**2),
            ("100G", 100 * 1024**3),
            ("100 GiB", 100 * 1024**3),
            ("1.5t", int(1.5 * 1024**4)),
        ],
    )
    def test_valid(self, raw, expected):
        assert _parse_memory(raw) == expected

    @pytest.mark.parametrize("raw", ["lots", "100X", "G", "0", -1, True])
    def test_invalid(self, raw):
        with pytest.raises(ValueError):
            _parse_memory(raw)


class TestSubprocessRunnerResources:
    def test_enabling_pool_waits_for_unassigned_batches(self):
        runner = _SubprocessRunner(num_workers=2)
        runner.active["running"] = SimpleNamespace(gpu=None, affinity=None)

        runner.configure(
            num_workers=2,
            num_threads=4,
            gpus=[0, 1],
            affinities=None,
            max_memory=None,
            log=True,
            verbosity_grow=1,
        )
        assert not runner.can_submit()

        runner.active.clear()
        assert runner.can_submit()
        assert runner.num_threads == 4
        assert runner.log

    def test_reconfigure_preserves_repeated_resource_ids(self):
        runner = _SubprocessRunner(num_workers=3, gpus=[0, 0])
        runner.active["running"] = SimpleNamespace(gpu=0, affinity=None)

        assert runner.can_submit()
        assert runner._free_slots("gpus") == [0]

        runner.configure(
            num_workers=3,
            num_threads=1,
            gpus=[1],
            affinities=None,
            max_memory=None,
            log=False,
            verbosity_grow=0,
        )
        assert runner.can_submit()
        assert runner._free_slots("gpus") == [1]


class TestSubprocessRunner:
    @pytest.mark.parametrize("raise_errors", [False, True])
    def test_raise_errors_is_passed_to_child(
        self, tmp_path, monkeypatch, raise_errors
    ):
        process = Mock()
        spawn = Mock(return_value=process)
        monkeypatch.setattr(growing, "Popen", spawn)
        runner = growing._SubprocessRunner(raise_errors=raise_errors)
        runner.submit(growing._BatchTask("test", tmp_path, 1))
        assert ("--raise-errors" in spawn.call_args.args[0]) == raise_errors
        runner.terminate()

    def test_spawn_failure_stops_active_child_and_closes_output(
        self, tmp_path, monkeypatch
    ):
        crop = xyz.Crop(fn=add_one, parent_dir=tmp_path)
        crop.sow_cases("x", [1, 2], verbosity=0)
        process = Mock()
        process.poll.return_value = None
        spawn = Mock(side_effect=[process, FileNotFoundError("taskset")])
        monkeypatch.setattr(growing, "Popen", spawn)
        with pytest.raises(FileNotFoundError):
            crop.grow_subprocess(num_workers=2, verbosity=0)
        process.terminate.assert_called_once()
        process.wait.assert_called_once()
        for call in spawn.call_args_list:
            assert call.kwargs["stderr"].closed

    def test_duplicate_batch_ids_launch_once(self, tmp_path, monkeypatch):
        crop = xyz.Crop(fn=add_one, parent_dir=tmp_path)
        crop.sow_cases("x", [1], verbosity=0)
        spawn = Mock(wraps=growing.Popen)
        monkeypatch.setattr(growing, "Popen", spawn)
        crop.grow_subprocess(batch_ids=[1, 1], num_workers=2, verbosity=0)
        assert spawn.call_count == 1
        assert crop.load_result(1) == (2,)

    def test_rejects_duplicate_active_batch(self, tmp_path, monkeypatch):
        spawn = Mock()
        monkeypatch.setattr(growing, "Popen", spawn)
        runner = growing._SubprocessRunner(num_workers=2)
        task = growing._BatchTask("test", tmp_path, 1)
        runner.submit(task)
        try:
            with pytest.raises(ValueError, match="already running"):
                runner.submit(task)
            assert spawn.call_count == 1
        finally:
            runner.terminate()


class TestSubprocessRunnerMemory:
    def test_command_is_unchanged(self, tmp_path, monkeypatch):
        monkeypatch.setattr(growing.shutil, "which", lambda cmd: cmd)
        monkeypatch.setattr(growing.os.path, "exists", lambda path: True)
        spawn = Mock()
        monkeypatch.setattr(growing, "Popen", spawn)
        runner = _SubprocessRunner(max_memory="1G", affinities=[3])
        runner.submit(growing._BatchTask("test", tmp_path, 1))
        args = spawn.call_args.args[0]
        runner.terminate()
        assert args[:4] == ["taskset", "-c", "3", sys.executable]

    def test_missing_proc_is_rejected(self, monkeypatch):
        monkeypatch.setattr(growing.os.path, "exists", lambda path: False)
        with pytest.raises(ValueError, match="Linux"):
            _SubprocessRunner(max_memory="1G")

    def test_kill_reports_memory_limit(self, tmp_path, monkeypatch):
        monkeypatch.setattr(growing.os.path, "exists", lambda path: True)
        monkeypatch.setattr(growing, "_process_tree", lambda pid: [pid])
        monkeypatch.setattr(growing, "_memory_usage", lambda pids: 3 * 1024**3)
        kill = Mock()
        monkeypatch.setattr(growing.os, "kill", kill)
        process = Mock(pid=123)
        process.poll.return_value = -9
        monkeypatch.setattr(growing, "Popen", Mock(return_value=process))
        runner = _SubprocessRunner(max_memory="1G")
        runner.submit(growing._BatchTask("test", tmp_path, 1))
        (completion,) = runner.poll()
        assert kill.call_args.args[0] == 123
        assert not completion.success
        assert "3.0G, over the memory limit of 1.0G" in completion.message

    def test_under_limit_is_not_killed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(growing.os.path, "exists", lambda path: True)
        monkeypatch.setattr(growing, "_process_tree", lambda pid: [pid])
        monkeypatch.setattr(growing, "_memory_usage", lambda pids: 1024)
        kill = Mock()
        monkeypatch.setattr(growing.os, "kill", kill)
        process = Mock(pid=123)
        process.poll.return_value = None
        monkeypatch.setattr(growing, "Popen", Mock(return_value=process))
        runner = _SubprocessRunner(max_memory="1G")
        runner.submit(growing._BatchTask("test", tmp_path, 1))
        completions = runner.poll()
        runner.terminate()
        assert completions == []
        kill.assert_not_called()

    @pytest.mark.skipif(not can_watch_memory(), reason="needs Linux /proc")
    def test_process_tree_and_usage(self):
        code = (
            "import subprocess, sys, time; "
            "subprocess.Popen([sys.executable, '-c', "
            "'import time; time.sleep(30)']); "
            "time.sleep(30)"
        )
        process = subprocess.Popen([sys.executable, "-c", code])
        try:
            for _ in range(100):
                pids = growing._process_tree(process.pid)
                if len(pids) == 2:
                    break
                time.sleep(0.05)
            assert pids[0] == process.pid
            assert len(pids) == 2
            assert growing._memory_usage(pids) > 0
        finally:
            for pid in reversed(growing._process_tree(process.pid)):
                os.kill(pid, signal.SIGKILL)
            process.wait()

    @pytest.mark.skipif(not can_watch_memory(), reason="needs Linux /proc")
    def test_batch_over_limit_is_killed(self, tmp_path):
        crop = xyz.Crop(fn=allocate, parent_dir=tmp_path)
        crop.sow_cases("nbytes", [1024, 2 * 1024**3], verbosity=0)
        crop.grow(max_memory="500M", verbosity=0)
        assert crop.load_result(1) == (1024,)
        assert not crop.is_ready_to_reap()
        assert crop.missing_results() == (2,)
