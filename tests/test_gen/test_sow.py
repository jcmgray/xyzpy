from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import xarray as xr

import xyzpy as xyz
from xyzpy.gen import cropping
from xyzpy.gen.auto_growing import AutoGrower
from xyzpy.utils import XYZError

from ..test_auto_grow import add_one, make_settings


def calculate(x, y=0, offset=0):
    return x + y + offset


class TestSow:
    @pytest.mark.parametrize("runner_constants", [False, True])
    @pytest.mark.parametrize(
        "value",
        [
            10,
            np.nan,
            np.array([1.0, np.nan]),
            np.array(["a", "b"]),
            {"nested": [np.array([1, 2]), (None, np.nan)]},
        ],
    )
    def test_unchanged_constants_leave_batch_files_untouched(
        self, tmp_path, runner_constants, value
    ):
        from copy import deepcopy

        options = {
            "var_names": "value",
            "combos": {"x": [1, 2]},
            "parent_dir": tmp_path,
            "verbosity": 0,
        }
        if runner_constants:
            options["runner_opts"] = {"constants": {"offset": value}}
        else:
            options["constants"] = {"offset": value}
        crop = xyz.sow(calculate, **options)
        paths = [
            Path(crop.location) / cropping.INFO_NM,
            *sorted((Path(crop.location) / "batches").iterdir()),
        ]
        before = [
            (p.stat().st_ino, p.stat().st_mtime_ns, p.read_bytes())
            for p in paths
        ]
        xyz.sow(calculate, **deepcopy(options))
        assert [
            (p.stat().st_ino, p.stat().st_mtime_ns, p.read_bytes())
            for p in paths
        ] == before

    def test_unchanged_constants_keep_failure_held(self, tmp_path):
        def fail_until_fixed(x, offset):
            if offset < 0:
                raise ValueError("negative offset")
            return x + offset

        options = {
            "var_names": "value",
            "combos": {"x": [1]},
            "parent_dir": tmp_path,
            "verbosity": 0,
        }
        crop = xyz.sow(fail_until_fixed, constants={"offset": -1}, **options)
        watcher = AutoGrower(
            tmp_path, make_settings(), tmp_path / "config.toml", once=True
        )
        assert watcher.run() == 0
        assert len(watcher.failed) == 1

        xyz.sow(fail_until_fixed, constants={"offset": -1}, **options)
        watcher.scan()
        assert len(watcher.failed) == 1
        assert not watcher.has_pending()

        xyz.sow(fail_until_fixed, constants={"offset": 10}, **options)
        watcher.scan()
        assert not watcher.failed
        assert watcher.has_pending()
        assert watcher.run() == 0
        assert crop.load_result(1) == (11,)

    @pytest.mark.parametrize("runner_constants", [False, True])
    def test_resow_updates_batches_without_regrowing_results(
        self, tmp_path, runner_constants
    ):
        options = {
            "var_names": "value",
            "data_name": tmp_path / "data.h5",
            "parent_dir": tmp_path,
            "combos": {"x": [1, 2, 3, 4]},
            "batchsize": 2,
            "shuffle": False,
            "verbosity": 0,
        }

        def constant_options(value):
            if runner_constants:
                return {"runner_opts": {"constants": {"offset": value}}}
            return {"constants": {"offset": value}}

        crop = xyz.sow(calculate, **options, **constant_options(10))
        crop.grow(1, verbosity=0)
        result_path = (
            Path(crop.location) / "results" / cropping.RSLT_NM.format(1)
        )
        original_result = result_path.read_bytes()
        crop = xyz.sow(calculate, **options, **constant_options(100))
        assert crop.load_batch(1) == [
            {"x": 1, "offset": 10},
            {"x": 2, "offset": 10},
        ]
        assert crop.load_batch(2) == [
            {"x": 3, "offset": 100},
            {"x": 4, "offset": 100},
        ]

        watcher = AutoGrower(
            tmp_path, make_settings(), tmp_path / "config.toml", once=True
        )
        watcher.scan()
        assert list(watcher.crops[0].pending_ids) == [2]
        assert watcher.run() == 0
        assert result_path.read_bytes() == original_result
        assert crop.load_result(1) == (11, 12)
        assert crop.load_result(2) == (103, 104)
        ds = crop.reap(verbosity=0)
        assert ds["value"].values.tolist() == [11, 12, 103, 104]
        assert ds.attrs["offset"] == 100

    def test_reuses_equivalent_requests_and_updates_function_and_constants(
        self, tmp_path
    ):
        crop = xyz.sow(
            calculate,
            var_names="value",
            data_name=tmp_path / "data.h5",
            parent_dir=tmp_path,
            combos={"x": np.array([1, 2]), "y": [3, 4]},
            constants={"offset": 10},
            batchsize=2,
            shuffle=False,
            verbosity=0,
        )
        batches = [crop.load_batch(i) for i in (1, 2)]

        def replacement(x, y=0, offset=0):
            return -100

        replacement.__name__ = calculate.__name__
        same = xyz.sow(
            replacement,
            var_names="different",
            data_name=tmp_path / "data",
            parent_dir=tmp_path,
            combos={"y": [4, 3]},
            cases=[{"x": 2.0}, {"x": 1.0}],
            constants={"offset": 100},
            batchsize=1,
            verbosity=0,
        )
        assert same.location == crop.location
        assert same.batchsize == 2
        assert same.fn(1, 3, 10) == -100
        assert same.runner.var_names == ("value",)
        for batch_id, batch in enumerate(batches, 1):
            assert same.load_batch(batch_id) == [
                {**case, "offset": 100} for case in batch
            ]

    def test_changed_coordinates_and_targets_get_distinct_names(
        self, tmp_path
    ):
        crops = [
            xyz.sow(
                calculate,
                var_names="value",
                combos={"x": values},
                data_name=tmp_path / target,
                parent_dir=tmp_path,
                verbosity=0,
            )
            for values, target in [([1], "a.h5"), ([2], "a.h5"), ([1], "b.h5")]
        ]
        assert len({crop.name for crop in crops}) == 3
        assert all(crop.name.startswith("calculate-") for crop in crops)

    def test_explicit_name_reuses_request_and_rejects_other_request(
        self, tmp_path
    ):
        options = {
            "var_names": "value",
            "name": "job",
            "parent_dir": tmp_path,
            "verbosity": 0,
        }
        crop = xyz.sow(calculate, combos={"x": [1]}, **options)
        assert (
            xyz.sow(calculate, combos={"x": [1]}, **options).name == crop.name
        )
        with pytest.raises(XYZError, match="another request"):
            xyz.sow(calculate, combos={"x": [2]}, **options)

    def test_crops_without_request_key_and_incomplete_crops_are_not_overwritten(
        self, tmp_path
    ):
        crop = xyz.Crop(fn=calculate, name="legacy", parent_dir=tmp_path)
        crop.sow_combos({"x": [1]}, verbosity=0)
        with pytest.raises(XYZError, match="no sow metadata"):
            xyz.sow(
                calculate,
                var_names="v",
                combos={"x": [1]},
                name="legacy",
                parent_dir=tmp_path,
                verbosity=0,
            )
        options = {
            "var_names": "v",
            "combos": {"x": [1]},
            "name": "new",
            "parent_dir": tmp_path,
            "verbosity": 0,
        }
        fresh = xyz.sow(calculate, **options)
        (
            Path(fresh.location) / "batches" / cropping.BTCH_NM.format(1)
        ).unlink()
        with pytest.raises(XYZError, match="incomplete"):
            xyz.sow(calculate, **options)

    def test_filters_missing_cases_and_returns_none_after_reaping(
        self, tmp_path
    ):
        target = str(tmp_path / "data.h5")
        harvester = xyz.Harvester(xyz.Runner(calculate, "value"), target)
        harvester.harvest_combos({"x": [1]}, verbosity=0)
        crop = harvester.sow(
            combos={"x": [1, 2]}, parent_dir=tmp_path, verbosity=0
        )
        assert crop.load_batch(1) == [{"x": 2}]
        crop.grow(verbosity=0)
        crop.reap(verbosity=0)
        assert (
            harvester.sow(
                combos={"x": [1, 2]}, parent_dir=tmp_path, verbosity=0
            )
            is None
        )
        assert not list(tmp_path.glob(".xyz-*"))
        all_cases = harvester.sow(
            combos={"x": [1, 2]},
            missing_only=False,
            parent_dir=tmp_path,
            shuffle=False,
            verbosity=0,
        )
        assert all_cases.num_batches == 2

    def test_matching_request_is_reused_after_partial_reap(self, tmp_path):
        options = {
            "var_names": "value",
            "combos": {"x": [1, 2]},
            "data_name": tmp_path / "data.h5",
            "parent_dir": tmp_path,
            "shuffle": False,
            "verbosity": 0,
        }
        crop = xyz.sow(calculate, **options)
        crop.grow(1, verbosity=0)
        crop.reap(allow_incomplete=True, verbosity=0)
        same = xyz.sow(calculate, **options)
        assert same.location == crop.location
        assert same.missing_results() == (2,)

    def test_reloads_dataset_and_expands_ellipsis(self, tmp_path):
        target = str(tmp_path / "data.h5")
        h = xyz.Harvester(xyz.Runner(calculate, "value"), target)
        h.harvest_combos({"x": [1]}, verbosity=0)
        other = xyz.Harvester(xyz.Runner(calculate, "value"), target)
        other.harvest_combos({"x": [2]}, verbosity=0)
        assert (
            h.sow(combos={"x": [2]}, parent_dir=tmp_path, verbosity=0) is None
        )
        crop = h.sow(
            combos={"x": ...},
            missing_only=False,
            parent_dir=tmp_path,
            num_batches=1,
            shuffle=False,
            verbosity=0,
        )
        assert crop.load_batch(1) == [{"x": 1}, {"x": 2}]

    def test_empty_cases_do_not_create_crop(self, tmp_path):
        assert (
            xyz.sow(
                calculate,
                var_names="v",
                cases=[],
                parent_dir=tmp_path,
                verbosity=0,
            )
            is None
        )
        assert not list(tmp_path.iterdir())

    def test_coordinate_constants_affect_key(self, tmp_path):
        def vector(x, times):
            return x * np.asarray(times)

        options = {
            "var_names": "value",
            "combos": {"x": [1]},
            "parent_dir": tmp_path,
            "verbosity": 0,
            "runner_opts": {"var_dims": {"value": ("times",)}},
        }
        first = xyz.sow(vector, constants={"times": [1, 2]}, **options)
        second = xyz.sow(vector, constants={"times": [1, 3]}, **options)
        assert first.name != second.name
        recovered = xyz.Crop(name=first.name, parent_dir=tmp_path)
        recovered.grow(verbosity=0)
        ds = recovered.reap(verbosity=0)
        assert ds["times"].values.tolist() == [1, 2]
        assert ds["value"].values.tolist() == [[1, 2]]

    def test_in_memory_dataset_and_tuple_cases(self, tmp_path):
        ds = xr.Dataset({"value": ("x", [1.0])}, coords={"x": [1]})
        h = xyz.Harvester(xyz.Runner(calculate, "value"), full_ds=ds)
        crop = h.sow(cases=[(1,), (2,)], parent_dir=tmp_path, verbosity=0)
        assert crop.load_batch(1) == [{"x": 2}]

    def test_notebook_jobs_grow_and_merge_into_one_dataset(self, tmp_path):
        target = str(tmp_path / "data.h5")
        crops = [
            xyz.sow(
                calculate,
                var_names="value",
                combos={"x": values},
                data_name=target,
                parent_dir=tmp_path,
                verbosity=0,
            )
            for values in ([1, 2], [3, 4])
        ]
        grower = AutoGrower(
            tmp_path,
            make_settings(num_workers=2),
            tmp_path / "config.toml",
            once=True,
        )
        assert grower.run() == 0
        for crop in crops:
            recovered = xyz.Crop(name=crop.name, parent_dir=tmp_path)
            recovered.reap(verbosity=0)
        ds = xyz.load_ds(target)
        assert ds["value"].sel(x=4).item() == 4
        assert ds.sizes["x"] == 4
        ds.close()
        assert not list(tmp_path.glob(".xyz-*"))

    def test_new_name_creates_second_crop_for_same_request(self, tmp_path):
        h = xyz.Harvester(xyz.Runner(add_one, "value"))
        h.harvest_combos({"x": [1]}, verbosity=0)
        options = {
            "combos": {"x": [1, 2]},
            "parent_dir": tmp_path,
            "verbosity": 0,
        }
        first = h.sow(**options)
        reused = h.sow(missing_only=False, **options)
        assert reused.location == first.location
        assert reused.num_batches == 1
        rerun = h.sow(name="rerun", missing_only=False, **options)
        assert rerun.num_batches == 2

    def test_conflicting_reap_can_retry_with_overwrite(self, tmp_path):
        h = xyz.Harvester(xyz.Runner(add_one, "value"))
        h.harvest_combos({"x": [1]}, verbosity=0)
        h.fn = lambda x: 99
        crop = h.sow(
            combos={"x": [1]},
            missing_only=False,
            parent_dir=tmp_path,
            verbosity=0,
        )
        crop.grow(verbosity=0)
        with pytest.raises(Exception, match="conflicting values"):
            crop.reap(verbosity=0)
        assert Path(crop.location).exists()
        crop.reap(overwrite=True, verbosity=0)
        assert h.full_ds.value.item() == 99

    @pytest.mark.parametrize("failure", ["batching", "serialization"])
    def test_failed_preparation_keeps_name_available(
        self, tmp_path, monkeypatch, failure
    ):
        options = {
            "combos": {"x": [1, 2, 3]},
            "parent_dir": tmp_path,
            "verbosity": 0,
        }
        with monkeypatch.context() as context:
            if failure == "serialization":
                context.setattr(
                    cropping.Crop,
                    "save_function_to_disk",
                    Mock(side_effect=ValueError("unpicklable")),
                )
                bad = {}
            else:
                bad = {"batchsize": 2, "num_batches": 5}
            with pytest.raises(ValueError):
                xyz.sow(add_one, var_names="value", **options, **bad)
        assert not list(tmp_path.iterdir())
        crop = xyz.sow(add_one, var_names="value", **options)
        assert crop.num_batches == 3

    def test_range_coordinates_and_lambda_name(self, tmp_path):
        crop = xyz.sow(
            lambda x, ts: x * np.asarray(ts),
            var_names="value",
            runner_opts={"var_dims": {"value": "ts"}},
            constants={"ts": range(3)},
            combos={"x": [2]},
            parent_dir=tmp_path,
            verbosity=0,
        )
        assert "<" not in crop.name and ">" not in crop.name
        crop.grow(verbosity=0)
        ds = crop.reap(verbosity=0)
        assert ds.ts.values.tolist() == [0, 1, 2]
        assert ds.value.values.tolist() == [[0, 2, 4]]
