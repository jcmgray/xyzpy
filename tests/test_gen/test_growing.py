from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import xyzpy as xyz
from xyzpy.gen import growing
from xyzpy.gen.growing import _SubprocessRunner

from ..test_auto_grow import add_one


class TestSubprocessRunnerResources:
    def test_enabling_pool_waits_for_unassigned_batches(self):
        runner = _SubprocessRunner(num_workers=2)
        runner.active["running"] = SimpleNamespace(gpu=None, affinity=None)

        runner.configure(
            num_workers=2,
            num_threads=4,
            gpus=[0, 1],
            affinities=None,
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
