import pytest
import xarray as xr

import xyzpy as xyz
from xyzpy.gen import farming


def twice(x):
    return x * 2


class TestHarvesterSync:
    @pytest.mark.parametrize("engine", ["h5netcdf", "zarr"])
    def test_reloads_external_harvest_and_deletion(self, tmp_path, engine):
        path = tmp_path / "results"
        first = xyz.Harvester(xyz.Runner(twice, "value"), path, engine=engine)
        second = xyz.Harvester(xyz.Runner(twice, "value"), path, engine=engine)
        first.harvest_combos({"x": [1]}, verbosity=0)
        assert second.full_ds.sizes["x"] == 1
        first.harvest_combos({"x": [2]}, verbosity=0)
        assert second.full_ds.sizes["x"] == 2
        first.delete_ds()
        assert second.full_ds is None

    @pytest.mark.parametrize("engine", ["h5netcdf", "zarr"])
    def test_unchanged_file_uses_cache(self, tmp_path, monkeypatch, engine):
        h = xyz.Harvester(
            xyz.Runner(twice, "value"), tmp_path / "data", engine=engine
        )
        h.harvest_combos({"x": [1]}, verbosity=0)

        def unexpected_load(*args, **kwargs):
            pytest.fail("unchanged dataset was reloaded")

        monkeypatch.setattr(farming, "load_ds", unexpected_load)
        assert h.full_ds is h.full_ds
        assert h.full_ds["value"].item() == 2

    def test_delete_ds_expands_home(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        h = xyz.Harvester(xyz.Runner(twice, "value"), "~/data")
        h.harvest_combos({"x": [1]}, verbosity=0)
        path = tmp_path / "data.h5"
        assert path.is_file()
        h.delete_ds()
        assert not path.exists()
        assert h.full_ds is None

    def test_reused_memory_crop_updates_function_and_caller(self, tmp_path):
        h = xyz.Harvester(xyz.Runner(twice, "value"))
        options = {
            "combos": {"x": [1]},
            "parent_dir": tmp_path,
            "verbosity": 0,
        }
        h.sow(**options)

        def changed(x):
            return -100

        changed.__name__ = twice.__name__
        h.runner = xyz.Runner(changed, "different")
        crop = h.sow(**options)
        crop.grow(verbosity=0)
        crop.reap(verbosity=0)
        assert h.full_ds["value"].item() == -100
        assert "different" not in h.full_ds
        assert h.runner.fn is changed

    def test_reused_disk_crop_updates_callers_view(self, tmp_path):
        h = xyz.Harvester(xyz.Runner(twice, "value"), tmp_path / "data.h5")
        h.harvest_combos({"x": [1]}, verbosity=0)
        options = {
            "combos": {"x": [1, 2]},
            "parent_dir": tmp_path,
            "verbosity": 0,
        }
        h.sow(**options)
        crop = h.sow(**options)
        crop.grow(verbosity=0)
        crop.reap(verbosity=0)
        assert h.full_ds.sizes["x"] == 2

    def test_unsynced_merge_is_lost_after_disk_changes(self, tmp_path):
        h = xyz.Harvester(xyz.Runner(twice, "value"), tmp_path / "data.h5")
        h.harvest_combos({"x": [1]}, verbosity=0)
        h.add_ds(
            xr.Dataset({"value": ("x", [4])}, coords={"x": [2]}), sync=False
        )
        assert h.full_ds.sizes["x"] == 2
        other = xyz.Harvester(xyz.Runner(twice, "value"), h.data_name)
        other.harvest_combos({"x": [3]}, verbosity=0)
        assert h.full_ds.x.values.tolist() == [1, 3]
