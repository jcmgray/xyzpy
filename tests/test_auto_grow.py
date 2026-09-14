from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

import pytest

import xyzpy as xyz
from xyzpy.gen import cropping
from xyzpy.gen.auto_growing import AutoGrower, ConfigFile, DirectoryLock
from xyzpy.gen.cropping import Crop
from xyzpy.gen.growing import _BatchTask


def add_one(x):
    return x + 1


def fail_if_negative(x):
    if x < 0:
        raise ValueError("negative value")
    return x + 1


def make_settings(**overrides):
    settings = {
        "num_workers": 1,
        "num_threads": 1,
        "gpus": None,
        "affinities": None,
        "raise_errors": False,
        "log": True,
        "min_wait": 0.001,
        "max_wait": 0.01,
        "verbosity": 0,
        "verbosity_grow": 0,
        "scan_interval": 0.01,
        "refresh_interval": 0.01,
        "desc": "test grower",
    }
    settings.update(overrides)
    return settings


class TestConfigFile:
    def test_removing_file_restores_cli_settings(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            config = ConfigFile(path, make_settings(num_workers=2))

            changed, settings, error = config.read()
            assert changed
            assert error is None
            assert settings["num_workers"] == 2

            path.write_text("num_workers = 4\ngpus = [0, 0, 1]\n")
            changed, settings, error = config.read()
            assert changed
            assert error is None
            assert settings["num_workers"] == 4
            assert settings["gpus"] == [0, 0, 1]

            path.unlink()
            changed, settings, error = config.read()
            assert changed
            assert error is None
            assert settings["num_workers"] == 2
            assert settings["gpus"] is None

    def test_invalid_edit_is_reported(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "config.toml"
            config = ConfigFile(path, make_settings())
            config.read()

            path.write_text("num_workers = -1\n")
            changed, settings, error = config.read()
            assert changed
            assert settings is None
            assert "num_workers" in error


class TestDirectoryLock:
    def test_rejects_second_watcher(self):
        with TemporaryDirectory() as directory:
            with (
                DirectoryLock(directory),
                pytest.raises(RuntimeError, match="already"),
                DirectoryLock(directory),
            ):
                pass

            with DirectoryLock(directory):
                pass


class TestAutoGrower:
    def test_round_robin_across_crops(self):
        with TemporaryDirectory() as directory:
            parent = Path(directory)
            for name in ("alpha", "beta"):
                crop = parent / f".xyz-{name}"
                (crop / "batches").mkdir(parents=True)
                (crop / "results").mkdir()
                (crop / "xyz-settings.jbdmp").touch()
                (crop / "xyz-function.clpkl").touch()
                for batch_id in (1, 2):
                    (crop / "batches" / f"xyz-batch-{batch_id}.jbdmp").touch()

            grower = AutoGrower(
                parent,
                make_settings(),
                parent / "config.toml",
            )
            grower.scan()
            selected = []
            for _ in range(4):
                task = grower.next_task()
                selected.append((task.crop_name, task.batch_id))

            assert selected == [
                ("alpha", 1),
                ("beta", 1),
                ("alpha", 2),
                ("beta", 2),
            ]

    def test_grows_multiple_crops_from_one_queue(self):
        with TemporaryDirectory() as directory:
            alpha = Crop(
                fn=add_one,
                name="alpha",
                parent_dir=directory,
                batchsize=1,
            )
            beta = Crop(
                fn=add_one,
                name="beta",
                parent_dir=directory,
                batchsize=1,
            )
            alpha.sow_cases("x", [1, 2], verbosity=0)
            beta.sow_cases("x", [10, 20], verbosity=0)

            grower = AutoGrower(
                directory,
                make_settings(num_workers=2),
                Path(directory) / "config.toml",
                once=True,
            )
            assert grower.run() == 0
            assert alpha.is_ready_to_reap()
            assert beta.is_ready_to_reap()
            assert alpha.reap(verbosity=0) == (2, 3)
            assert beta.reap(verbosity=0) == (11, 21)

    def test_holds_failure_and_continues(self):
        with TemporaryDirectory() as directory:
            crop = Crop(
                fn=fail_if_negative,
                name="mixed",
                parent_dir=directory,
                batchsize=1,
            )
            crop.sow_cases("x", [-1, 1], verbosity=0)

            grower = AutoGrower(
                directory,
                make_settings(),
                Path(directory) / "config.toml",
                once=True,
            )
            assert grower.run() == 0
            assert len(grower.failed) == 1
            assert crop.missing_results() == (1,)
            assert crop.load_result(2) == (2,)

            crop.delete_all()
            grower.scan()
            assert not grower.failed


class TestAutoGrowerScheduling:
    def test_symlink_aliases_share_one_queue(self, tmp_path):
        store = tmp_path / "store"
        watched = tmp_path / "watched"
        watched.mkdir()
        crop = Crop(fn=add_one, name="real", parent_dir=store)
        crop.sow_cases("x", [1], verbosity=0)
        try:
            for name in ("alpha", "beta"):
                (watched / f".xyz-{name}").symlink_to(
                    crop.location, target_is_directory=True
                )
        except OSError:
            pytest.skip("directory symlinks unavailable")
        watcher = AutoGrower(
            watched, make_settings(), watched / "config.toml", once=True
        )
        watcher.scan()
        assert len(watcher.crops) == 1
        task = watcher.next_task()
        assert task.batch_file.is_file()
        assert task.key[0] == str(Path(crop.location).resolve())
        assert watcher.run() == 0
        assert crop.load_result(1) == (2,)

    def test_paused_once_preserves_pending_work(self, tmp_path):
        crop = Crop(fn=add_one, parent_dir=tmp_path)
        crop.sow_cases("x", [1, 2], verbosity=0)
        watcher = AutoGrower(
            tmp_path,
            make_settings(num_workers=0),
            tmp_path / "config.toml",
            once=True,
        )
        assert watcher.run() == 0
        assert crop.missing_results() == (1, 2)

    @pytest.mark.parametrize("directory", ["batches", "results"])
    def test_scan_skips_crop_if_directory_disappears(
        self, tmp_path, monkeypatch, directory
    ):
        crop = Crop(fn=add_one, parent_dir=tmp_path)
        crop.sow_cases("x", [1], verbosity=0)
        original_iterdir = Path.iterdir

        def disappearing(path):
            if path.name == directory:
                raise FileNotFoundError(path)
            return original_iterdir(path)

        monkeypatch.setattr(Path, "iterdir", disappearing)
        watcher = AutoGrower(
            tmp_path, make_settings(), tmp_path / "config.toml"
        )
        watcher.scan()
        assert not watcher.crops

    def test_run_terminates_children_on_error(self, tmp_path, monkeypatch):
        watcher = AutoGrower(
            tmp_path, make_settings(), tmp_path / "config.toml"
        )
        watcher.runner.active["running"] = object()
        terminate = Mock()
        monkeypatch.setattr(watcher.runner, "terminate", terminate)
        monkeypatch.setattr(watcher.runner, "poll", Mock(return_value=[]))
        monkeypatch.setattr(
            watcher,
            "_launch_available",
            Mock(side_effect=RuntimeError("watcher failed")),
        )
        with pytest.raises(RuntimeError, match="watcher failed"):
            watcher.run()
        terminate.assert_called_once_with()

    def test_stale_task_stops_launch_loop(self, tmp_path, monkeypatch):
        watcher = AutoGrower(
            tmp_path, make_settings(), tmp_path / "config.toml"
        )
        task = _BatchTask("absent", tmp_path, 1)
        selected = Mock(side_effect=[task, AssertionError("busy loop")])
        monkeypatch.setattr(watcher, "next_task", selected)
        watcher._launch_available()
        assert selected.call_count == 1

    def test_watcher_ignores_crop_until_sow_finishes(
        self, tmp_path, monkeypatch
    ):
        watcher = AutoGrower(
            tmp_path, make_settings(), tmp_path / "config.toml"
        )
        original = cropping.Sower.__call__

        def checked_sow(sower, **kwargs):
            result = original(sower, **kwargs)
            watcher.scan()
            assert not watcher.crops
            return result

        monkeypatch.setattr(cropping.Sower, "__call__", checked_sow)
        crop = xyz.sow(
            add_one,
            var_names="value",
            combos={"x": [1, 2]},
            parent_dir=tmp_path,
            verbosity=0,
        )
        watcher.scan()
        assert watcher.crops[0].batch_ids == (1, 2)
        assert watcher.crops[0].name == crop.name
