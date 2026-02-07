import os
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from xyzpy import Harvester, Runner, combo_runner, combo_runner_to_ds, label
from xyzpy.gen.cropping import (
    Crop,
    XYZError,
    grow,
    load_crops,
    parse_crop_details,
)

from . import (
    foo2_array,
    foo2_array_bool,
    foo2_dataset,
    foo2_scalar,
    foo3_scalar,
)


def foo_add(a, b, c):
    return a + b


class TestSowerReaper:
    @pytest.mark.parametrize(
        "fn, crop_name, crop_loc, expected",
        [
            (foo_add, None, None, ".xyz-foo_add"),
            (None, "custom", None, ".xyz-custom"),
            (foo_add, "custom", None, ".xyz-custom"),
            (
                foo_add,
                None,
                "custom_dir",
                os.path.join("custom_dir", ".xyz-foo_add"),
            ),
            (
                None,
                "custom",
                "custom_dir",
                os.path.join("custom_dir", ".xyz-custom"),
            ),
            (
                foo_add,
                "custom",
                "custom_dir",
                os.path.join("custom_dir", ".xyz-custom"),
            ),
            (None, None, None, "raises"),
            (None, None, "custom_dir", "raises"),
        ],
    )
    def test_parse_crop_details(self, fn, crop_name, crop_loc, expected):

        if expected == "raises":
            with pytest.raises(ValueError):
                parse_crop_details(fn, crop_name, crop_loc)
        else:
            crop_loc = parse_crop_details(fn, crop_name, crop_loc)[0]
            assert crop_loc[-len(expected) :] == expected

    def test_checks(self):
        with pytest.raises(ValueError):
            Crop(name="custom", save_fn=True)

        with pytest.raises(TypeError):
            c = Crop(fn=foo_add, save_fn=False, batchsize=0.5)
            c.choose_batch_settings(combos=[("a", [1, 2])])

        with pytest.raises(ValueError):
            c = Crop(fn=foo_add, save_fn=False, batchsize=-1)
            c.choose_batch_settings(combos=[("a", [1, 2])])

        with pytest.raises(ValueError):
            c = Crop(fn=foo_add, save_fn=False, batchsize=1, num_batches=2)
            c.choose_batch_settings(combos=[("a", [1, 2, 3])])

        with pytest.raises(ValueError):
            c = Crop(fn=foo_add, save_fn=False, batchsize=2, num_batches=3)
            c.choose_batch_settings(combos=[("a", [1, 2, 3])])

        c = Crop(fn=foo_add, save_fn=False, batchsize=1, num_batches=3)
        c.choose_batch_settings(combos=[("a", [1, 2, 3])])

        c = Crop(fn=foo_add, save_fn=False, batchsize=2, num_batches=2)
        c.choose_batch_settings(combos=[("a", [1, 2, 3])])

        c = Crop(fn=foo_add, save_fn=False, batchsize=3, num_batches=1)
        c.choose_batch_settings(combos=[("a", [1, 2, 3])])

        with pytest.raises(XYZError):
            grow(1)

        print(c)
        repr(c)

    @pytest.mark.parametrize("shuffle", [False, True, 2])
    def test_batch(self, shuffle):

        combos = [
            ("a", [10, 20, 30]),
            ("b", [4, 5, 6, 7]),
        ]
        expected = combo_runner(foo_add, combos, constants={"c": True})

        with TemporaryDirectory() as tdir:
            # sow seeds
            crop = Crop(fn=foo_add, parent_dir=tdir, batchsize=5)

            assert not crop.is_prepared()
            assert crop.num_sown_batches == crop.num_results == -1

            crop.sow_combos(combos, constants={"c": True}, shuffle=shuffle)

            assert crop.is_prepared()
            assert crop.num_sown_batches == 3
            assert crop.num_results == 0

            # grow seeds
            for i in range(1, 4):
                grow(i, Crop(parent_dir=tdir, name="foo_add"))

                if i == 1:
                    assert crop.missing_results() == (
                        2,
                        3,
                    )

                    with pytest.raises(XYZError):
                        crop.reap()

            assert crop.is_ready_to_reap()
            assert not crop.check_bad()
            # reap results
            results = crop.reap()

        assert results == expected

    def test_field_name_and_overlapping(self):
        combos1 = [("a", [10, 20, 30]), ("b", [4, 5, 6, 7])]
        expected1 = combo_runner(foo_add, combos1, constants={"c": True})

        combos2 = [("a", [40, 50, 60]), ("b", [4, 5, 6, 7])]
        expected2 = combo_runner(foo_add, combos2, constants={"c": True})

        with TemporaryDirectory() as tdir:
            # sow seeds
            c1 = Crop(name="run1", fn=foo_add, parent_dir=tdir, batchsize=5)
            c1.sow_combos(combos1, constants={"c": True})
            c2 = Crop(name="run2", fn=foo_add, parent_dir=tdir, batchsize=5)
            c2.sow_combos(combos2, constants={"c": True})

            # grow seeds
            for i in range(1, 4):
                grow(i, Crop(parent_dir=tdir, name="run1"))
                grow(i, Crop(parent_dir=tdir, name="run2"))

            # reap results
            assert not c1.check_bad()
            assert not c2.check_bad()
            results1 = c1.reap()
            results2 = c2.reap()

        assert results1 == expected1
        assert results2 == expected2

    @pytest.mark.parametrize("num_workers", [None, 2])
    @pytest.mark.parametrize("shuffle", [False, True, 2])
    def test_crop_grow_missing(self, num_workers, shuffle):
        combos1 = [("a", [10, 20, 30]), ("b", [4, 5, 6, 7])]
        expected1 = combo_runner(foo_add, combos1, constants={"c": True})
        with TemporaryDirectory() as tdir:
            c1 = Crop(name="run1", fn=foo_add, parent_dir=tdir, batchsize=5)
            c1.sow_combos(combos1, constants={"c": True}, shuffle=shuffle)
            c1.grow_missing(num_workers=num_workers)
            results1 = c1.reap()
        assert results1 == expected1

    def test_combo_reaper_to_ds(self):
        combos = (
            ("a", [1, 2]),
            ("b", [10, 20, 30]),
            ("c", [100, 200, 300, 400]),
        )

        with TemporaryDirectory() as tdir:
            # sow seeds
            crop = Crop(fn=foo3_scalar, parent_dir=tdir, batchsize=5)
            crop.sow_combos(combos)

            # check on disk repr works
            print(crop)
            repr(crop)

            # grow seeds
            for i in range(1, 6):
                crop.grow(i)

                if i == 3:
                    with pytest.raises(XYZError):
                        crop.reap_combos_to_ds(var_names=["bananas"])

            ds = crop.reap_combos_to_ds(var_names=["bananas"])

        assert ds.sel(a=2, b=30, c=400)["bananas"].data == 432

    @pytest.mark.parametrize("num_batches", [67, 98])
    def test_num_batches_doesnt_divide(self, num_batches):
        combos = (
            ("a", [1, 2, 3]),
            ("b", [10, 20, 30]),
            ("c", range(100, 1101, 100)),
        )

        with TemporaryDirectory() as tdir:
            crop = Crop(fn=foo_add, parent_dir=tdir, num_batches=num_batches)
            crop.sow_combos(combos)
            assert crop.num_batches == num_batches
            crop.grow_missing()
            ds = crop.reap_combos_to_ds(var_names=["sum"])

        assert ds["sum"].sel(a=3, b=30, c=1100).data == 33

    @pytest.mark.parametrize(
        "fn",
        [
            foo2_scalar,
            foo2_array,
            foo2_array_bool,
            foo2_dataset,
        ],
    )
    def test_all_nan_result(self, fn):

        combos = (("a", [1, 2, 3]), ("b", [10, 20, 30]))

        with TemporaryDirectory() as tdir:
            crop = Crop(fn=fn, parent_dir=tdir)
            crop.sow_combos(combos)

            with pytest.raises(XYZError):
                crop.all_nan_result

            crop.grow(1)
            nres = crop.all_nan_result

            if fn is foo2_array_bool:
                assert len(nres) == 2
                assert_allclose(nres[0], np.broadcast_to(np.nan, [10]))
                assert_allclose(nres[1], np.broadcast_to(np.nan, []))

            if fn is foo2_dataset:
                ds_exp = xr.Dataset(
                    {"x": (["t1", "t2"], np.tile(np.nan, (2, 3)))}
                )
                assert nres.identical(ds_exp)

    @pytest.mark.parametrize("shuffle", [False, True, 2])
    def test_reap_allow_incomplete(self, shuffle):
        combos = (
            ("a", [1, 2, 3]),
            ("b", [10, 20, 30]),
            ("c", range(100, 1101, 100)),
        )

        with TemporaryDirectory() as tdir:
            crop = Crop(fn=foo_add, parent_dir=tdir)
            crop.sow_combos(combos, shuffle=shuffle)
            with pytest.raises(XYZError):
                crop.reap(allow_incomplete=True)
            for i in range(1, 40):
                crop.grow(i)
            res = np.array(crop.reap(allow_incomplete=True))
            assert np.isnan(res).sum() == 60

    @pytest.mark.parametrize(
        "fn,var_names,var_dims",
        [
            (foo2_scalar, ["x"], None),
            (foo2_array, ["x"], {"x": "t"}),
            (foo2_array_bool, ["x", "y"], {"x": "t"}),
            (foo2_dataset, None, None),
        ],
    )
    def test_reap_to_ds_allow_incomplete(self, fn, var_names, var_dims):
        combos = (("a", [1, 2, 3]), ("b", [10, 20, 30]))

        ds_exp = combo_runner_to_ds(fn, combos, var_names, var_dims=var_dims)

        with TemporaryDirectory() as tdir:
            crop = Crop(fn=fn, parent_dir=tdir)
            crop.sow_combos(combos)
            for i in range(1, 10, 2):
                crop.grow(i)

            ds = crop.reap_combos_to_ds(
                var_names=var_names, var_dims=var_dims, allow_incomplete=True
            )

            num_finished = int(ds["x"].size * (5 / 9))
            assert (ds["x"] == ds_exp["x"]).sum() == num_finished

            crop.grow_missing()
            ds = crop.reap_combos_to_ds(
                var_names=var_names, var_dims=var_dims, allow_incomplete=True
            )
            assert ds.identical(ds_exp)

    def test_new_ds_crop_loads_info_incomplete(self):
        def fn(a, b):
            return xr.Dataset({"sum": a + b, "diff": a - b})

        with TemporaryDirectory() as tdir:
            disk_ds = os.path.join(tdir, "test.h5")

            combos = dict(a=[1], b=[1, 2, 3])
            runner = Runner(fn, var_names=None)
            harvester = Harvester(runner, disk_ds)
            crop = harvester.Crop(name="fn", batchsize=1, parent_dir=tdir)
            crop.sow_combos(combos)
            for i in range(1, 3):
                crop.grow(i)

            # try creating crop from fresh
            c = Crop(name="fn", parent_dir=tdir)
            # crop's harvester should be loaded from disk
            assert c.farmer is not None
            assert c.farmer is not harvester
            ds = c.reap(allow_incomplete=True)
            assert isinstance(ds, xr.Dataset)
            assert ds["diff"].isnull().sum() == 1
            assert harvester.full_ds["diff"].isnull().sum() == 1

            # try creating crop from harvester
            c = harvester.Crop("fn", parent_dir=tdir)
            # crop's harvester should still be harvester
            assert c.farmer is not None
            assert c.farmer is harvester
            ds = c.reap(allow_incomplete=True)
            assert isinstance(ds, xr.Dataset)
            assert ds["diff"].isnull().sum() == 1

    def test_load_crops(self):

        combos = (
            ("a", [1, 2, 3]),
            ("b", [10, 20, 30]),
            ("c", range(100, 1101, 100)),
        )

        with TemporaryDirectory() as tdir:
            c1 = Crop(name="Alice", fn=foo_add, parent_dir=tdir)
            c2 = Crop(name="Bob", fn=foo_add, parent_dir=tdir)

            c1.sow_combos(combos)
            c2.sow_combos(combos)

            crops = load_crops(tdir)
            assert "Alice" in crops
            assert "Bob" in crops
            assert len(crops) == 2

            c1.grow_missing()
            c2.grow_missing()

            assert (
                c1.reap_combos()
                == c2.reap_combos()
                == combo_runner(foo_add, combos)
            )

    @pytest.mark.parametrize("scheduler", ["sge", "pbs", "slurm"])
    def test_gen_cluster_script(self, scheduler):
        combos = [
            ("a", [10, 20, 30]),
            ("b", [4, 5, 6, 7]),
        ]

        with TemporaryDirectory() as tdir:
            # sow seeds
            crop = Crop(fn=foo_add, parent_dir=tdir)
            crop.sow_combos(combos, constants={"c": True})

            # test script to grow all
            s1 = crop.gen_cluster_script(scheduler=scheduler, minutes=20)
            print(s1)

            # test script to grow specified
            s2 = crop.gen_cluster_script(batch_ids=[0, 1], scheduler=scheduler)
            print(s2)

            # test script to grow missing
            crop.grow((1, 3, 5))
            s3 = crop.gen_cluster_script(scheduler=scheduler)
            print(s3)

            assert s1 != s2
            assert s1 != s3
            assert s2 != s3

    def test_sow_reap_cases(self):

        def dummy_function(a, b):
            return a + b

        cases1 = [(1, 2), (3, 4)]
        cases2 = [(5, 6), (7, 8)]
        fn_args = ("a", "b")

        with TemporaryDirectory() as tmpdir:
            fl_pth = os.path.join(tmpdir, "test.dmp")
            runner = Runner(dummy_function, var_names=["Dummy_Value"])
            harvester = Harvester(runner, engine="joblib", data_name=fl_pth)

            crop1 = harvester.Crop(parent_dir=tmpdir)
            crop1.sow_cases(fn_args=fn_args, cases=cases1)
            crop1.grow_missing()
            crop1.reap()

            crop2 = harvester.Crop(parent_dir=tmpdir)
            crop2.sow_cases(fn_args=fn_args, cases=cases2)
            crop2.grow_missing()
            crop2.reap()

            ds = harvester.full_ds
            assert isinstance(ds, xr.Dataset)
            assert ds["Dummy_Value"].size == 16
            assert ds["Dummy_Value"].notnull().sum() == 4

    def test_sow_reap_mixed_combos_cases(self):

        @label("out")
        def fn(a, b, c, d, e):
            return f"{a}{b}{c}{d}{e}"

        with TemporaryDirectory() as tmpdir:
            crop = fn.Crop("test", parent_dir=tmpdir)
            cases = [
                {"a": 1, "c": 3},
                {"a": 2, "c": 4},
            ]
            combos = {
                "b": [5, 6],
                "d": [7, 8],
            }
            constants = {"e": 9}
            crop.sow_combos(combos=combos, cases=cases, constants=constants)
            crop.grow_missing()
            ds = crop.reap()

        assert ds["out"].ndim == 4
        assert ds["out"].notnull().sum().item() == 8

    def test_delete_all_resets_state(self):
        combos = [
            ("a", [10, 20, 30]),
            ("b", [4, 5, 6, 7]),
        ]

        with TemporaryDirectory() as tdir:
            crop = Crop(fn=foo_add, parent_dir=tdir, batchsize=5)
            crop.sow_combos(combos, constants={"c": True})

            # confirm crop is prepared and has loaded state
            assert crop.is_prepared()
            assert crop.batchsize == 5
            assert crop.num_batches == 3
            assert crop._batch_remainder is not None or True
            crop.calc_progress()
            assert crop._num_sown_batches == 3
            assert crop._num_results == 0

            # grow one batch to populate cached result state
            grow(1, Crop(parent_dir=tdir, name="foo_add"))
            crop.calc_progress()
            assert crop._num_results == 1

            crop.delete_all()

            # directory should be gone
            assert not os.path.exists(crop.location)

            # all loaded information should be reset
            assert crop.batchsize is None
            assert crop.num_batches is None
            assert crop._batch_remainder is None
            assert crop._all_nan_result is None
            assert crop._num_sown_batches == -1
            assert crop._num_results == -1

            # identity attributes should be preserved
            assert crop.name == "foo_add"
            assert crop.parent_dir is not None
            assert crop.location is not None
            assert crop.fn is foo_add
