import os
import sys

import pytest

import xyzpy as xyz
import xyzpy_grow

from .test_auto_grow import add_one


def run_cli(options, tmp_path, monkeypatch):
    """Run the ``xyzpy-grow`` CLI on a fresh crop, returning the options it
    passes on to :meth:`Crop.grow`.
    """
    crop = xyz.Crop(fn=add_one, parent_dir=tmp_path)
    crop.sow_cases("x", [1], verbosity=0)

    grow_kwargs = {}
    monkeypatch.setattr(
        xyz.Crop, "grow", lambda self, **kwargs: grow_kwargs.update(kwargs)
    )
    # main sets thread env vars and extends sys.path, keep both local
    monkeypatch.setattr(xyzpy_grow.os, "environ", dict(os.environ))
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(
        sys,
        "argv",
        ["xyzpy-grow", crop.name, "--parent-dir", str(tmp_path), *options],
    )
    xyzpy_grow.main()
    return grow_kwargs


class TestSubprocessAuto:
    def test_default_grows_in_process(self, tmp_path, monkeypatch):
        grow_kwargs = run_cli([], tmp_path, monkeypatch)
        assert grow_kwargs["subprocess"] is False
        assert "gpus" not in grow_kwargs

    @pytest.mark.parametrize(
        "options",
        [["--gpus", "0"], ["--affinities", "0"], ["--log"]],
    )
    def test_child_only_options_imply_subprocess(
        self, tmp_path, monkeypatch, options
    ):
        grow_kwargs = run_cli(options, tmp_path, monkeypatch)
        assert grow_kwargs["subprocess"] is True
        assert grow_kwargs["num_threads"] == 1

    def test_num_threads_alone_grows_in_process(self, tmp_path, monkeypatch):
        # else each child process would spawn another child process
        grow_kwargs = run_cli(["--num-threads", "4"], tmp_path, monkeypatch)
        assert grow_kwargs["subprocess"] is False

    def test_explicitly_enabled(self, tmp_path, monkeypatch):
        grow_kwargs = run_cli(["--subprocess"], tmp_path, monkeypatch)
        assert grow_kwargs["subprocess"] is True
        assert grow_kwargs["gpus"] is None

    def test_explicitly_disabled(self, tmp_path, monkeypatch):
        grow_kwargs = run_cli(
            ["--subprocess", "false", "--num-threads", "4"],
            tmp_path,
            monkeypatch,
        )
        assert grow_kwargs["subprocess"] is False

    def test_explicitly_disabled_rejects_child_only_options(
        self, tmp_path, monkeypatch
    ):
        with pytest.raises(SystemExit):
            run_cli(
                ["--subprocess", "false", "--gpus", "0"],
                tmp_path,
                monkeypatch,
            )

    def test_ray_rejects_implied_subprocess(self, tmp_path, monkeypatch):
        with pytest.raises(SystemExit):
            run_cli(["--ray", "--gpus", "0"], tmp_path, monkeypatch)
