import os
from tempfile import TemporaryDirectory

import pytest
import numpy as np
import xarray as xr
from numpy.testing import assert_allclose

from xyzpy import combo_runner, combo_runner_to_ds, Runner, Harvester
from xyzpy.gen.batch import (
    XYZError,
    Crop,
    parse_crop_details,
    grow,
    load_crops,
)

from . import (
    foo3_scalar,
    foo2_scalar,
    foo2_array,
    foo2_array_bool,
    foo2_dataset,
)


def foo_add(a, b, c):
    return a + b


class TestSowerReaper:
    @pytest.mark.parametrize(
        "fn, crop_name, crop_loc, expected",
        [
            (foo_add, None, None, '.xyz-foo_add'),
            (None, 'custom', None, '.xyz-custom'),
            (foo_add, 'custom', None, '.xyz-custom'),
            (foo_add, None, 'custom_dir', os.path.join('custom_dir',
                                                       '.xyz-foo_add')),
            (None, 'custom', 'custom_dir', os.path.join('custom_dir',
                                                        '.xyz-custom')),
            (foo_add, 'custom', 'custom_dir', os.path.join('custom_dir',
                                                           '.xyz-custom')),
            (None, None, None, 'raises'),
            (None, None, 'custom_dir', 'raises'),

        ])
    def test_parse_crop_details(self, fn, crop_name, crop_loc, expected):

        if expected == 'raises':
            with pytest.raises(ValueError):
                parse_crop_details(fn, crop_name, crop_loc)
        else:
            crop_loc = parse_crop_details(fn, crop_name, crop_loc)[0]
            assert crop_loc[-len(expected):] == expected

    def test_checks(self):
        with pytest.raises(ValueError):
            Crop(name='custom', save_fn=True)

        with pytest.raises(TypeError):
            c = Crop(fn=foo_add, save_fn=False, batchsize=0.5)
            c.choose_batch_settings(combos=[('a', [1, 2])])

        with pytest.raises(ValueError):
            c = Crop(fn=foo_add, save_fn=False, batchsize=-1)
            c.choose_batch_settings(combos=[('a', [1, 2])])

        with pytest.raises(ValueError):
            c = Crop(fn=foo_add, save_fn=False, batchsize=1, num_batches=2)
            c.choose_batch_settings(combos=[('a', [1, 2, 3])])

        with pytest.raises(ValueError):
            c = Crop(fn=foo_add, save_fn=False, batchsize=2, num_batches=3)
            c.choose_batch_settings(combos=[('a', [1, 2, 3])])

        c = Crop(fn=foo_add, save_fn=False, batchsize=1, num_batches=3)
        c.choose_batch_settings(combos=[('a', [1, 2, 3])])

        c = Crop(fn=foo_add, save_fn=False, batchsize=2, num_batches=2)
        c.choose_batch_settings(combos=[('a', [1, 2, 3])])

        c = Crop(fn=foo_add, save_fn=False, batchsize=3, num_batches=1)
        c.choose_batch_settings(combos=[('a', [1, 2, 3])])

        with pytest.raises(XYZError):
            grow(1)

        print(c)
        repr(c)

    def test_batch(self):

        combos = [
            ('a', [10, 20, 30]),
            ('b', [4, 5, 6, 7]),
        ]
        expected = combo_runner(foo_add, combos, constants={'c': True})

        with TemporaryDirectory() as tdir:

            # sow seeds
            crop = Crop(fn=foo_add, parent_dir=tdir, batchsize=5)

            assert not crop.is_prepared()
            assert crop.num_sown_batches == crop.num_results == -1

            crop.sow_combos(combos, constants={'c': True})

            assert crop.is_prepared()
            assert crop.num_sown_batches == 3
            assert crop.num_results == 0

            # grow seeds
            for i in range(1, 4):
                grow(i, Crop(parent_dir=tdir, name='foo_add'))

                if i == 1:
                    assert crop.missing_results() == (2, 3,)

                    with pytest.raises(XYZError):
                        crop.reap()

            assert crop.is_ready_to_reap()
            assert not crop.check_bad()
            # reap results
            results = crop.reap()

        assert results == expected

    def test_field_name_and_overlapping(self):
        combos1 = [('a', [10, 20, 30]),
                   ('b', [4, 5, 6, 7])]
        expected1 = combo_runner(foo_add, combos1, constants={'c': True})

        combos2 = [('a', [40, 50, 60]),
                   ('b', [4, 5, 6, 7])]
        expected2 = combo_runner(foo_add, combos2, constants={'c': True})

        with TemporaryDirectory() as tdir:
            # sow seeds
            c1 = Crop(name='run1', fn=foo_add, parent_dir=tdir, batchsize=5)
            c1.sow_combos(combos1, constants={'c': True})
            c2 = Crop(name='run2', fn=foo_add, parent_dir=tdir, batchsize=5)
            c2.sow_combos(combos2, constants={'c': True})

            # grow seeds
            for i in range(1, 4):
                grow(i, Crop(parent_dir=tdir, name='run1'))
                grow(i, Crop(parent_dir=tdir, name='run2'))

            # reap results
            assert not c1.check_bad()
            assert not c2.check_bad()
            results1 = c1.reap()
            results2 = c2.reap()

        assert results1 == expected1
        assert results2 == expected2

    @pytest.mark.parametrize("num_workers", [None, 2])
    def test_crop_grow_missing(self, num_workers):
        combos1 = [('a', [10, 20, 30]),
                   ('b', [4, 5, 6, 7])]
        expected1 = combo_runner(foo_add, combos1, constants={'c': True})
        with TemporaryDirectory() as tdir:
            c1 = Crop(name='run1', fn=foo_add, parent_dir=tdir, batchsize=5)
            c1.sow_combos(combos1, constants={'c': True})
            c1.grow_missing(num_workers=num_workers)
            results1 = c1.reap()
        assert results1 == expected1

    def test_combo_reaper_to_ds(self):
        combos = (('a', [1, 2]),
                  ('b', [10, 20, 30]),
                  ('c', [100, 200, 300, 400]))

        with TemporaryDirectory() as tdir:

            # sow seeds
            crop = Crop(fn=foo3_scalar, parent_dir=tdir, batchsize=5)
            crop.sow_combos(combos)

            # check on disk repr works and gen qsub_script works
            print(crop)
            repr(crop)
            assert crop.gen_qsub_script() is not None

            # grow seeds
            for i in range(1, 6):
                crop.grow(i)

                if i == 3:
                    with pytest.raises(XYZError):
                        crop.reap_combos_to_ds(var_names=['bananas'])

            ds = crop.reap_combos_to_ds(var_names=['bananas'])

        assert ds.sel(a=2, b=30, c=400)['bananas'].data == 432

    @pytest.mark.parametrize("num_batches", [67, 98])
    def test_num_batches_doesnt_divide(self, num_batches):
        combos = (('a', [1, 2, 3]),
                  ('b', [10, 20, 30]),
                  ('c', range(100, 1101, 100)))

        with TemporaryDirectory() as tdir:
            crop = Crop(fn=foo_add, parent_dir=tdir, num_batches=num_batches)
            crop.sow_combos(combos)
            assert crop.num_batches == num_batches
            crop.grow_missing()
            ds = crop.reap_combos_to_ds(var_names=['sum'])

        assert ds['sum'].sel(a=3, b=30, c=1100).data == 33

    @pytest.mark.parametrize('fn', [
        foo2_scalar,
        foo2_array,
        foo2_array_bool,
        foo2_dataset,
    ])
    def test_all_nan_result(self, fn):

        combos = (('a', [1, 2, 3]),
                  ('b', [10, 20, 30]))

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
                ds_exp = xr.Dataset({'x': (['t1', 't2'],
                                           np.tile(np.nan, (2, 3)))})
                assert nres.identical(ds_exp)

    def test_reap_allow_incomplete(self):
        combos = (('a', [1, 2, 3]),
                  ('b', [10, 20, 30]),
                  ('c', range(100, 1101, 100)))

        with TemporaryDirectory() as tdir:
            crop = Crop(fn=foo_add, parent_dir=tdir)
            crop.sow_combos(combos)
            with pytest.raises(XYZError):
                crop.reap(allow_incomplete=True)
            for i in range(1, 40):
                crop.grow(i)
            res = np.array(crop.reap(allow_incomplete=True))
            assert np.isnan(res).sum() == 60

    @pytest.mark.parametrize(
        "fn,var_names,var_dims", [
            (foo2_scalar, ['x'], None),
            (foo2_array, ['x'], {'x': 't'}),
            (foo2_array_bool, ['x', 'y'], {'x': 't'}),
            (foo2_dataset, None, None),
        ]
    )
    def test_reap_to_ds_allow_incomplete(self, fn, var_names, var_dims):
        combos = (('a', [1, 2, 3]),
                  ('b', [10, 20, 30]))

        ds_exp = combo_runner_to_ds(fn, combos, var_names, var_dims=var_dims)

        with TemporaryDirectory() as tdir:
            crop = Crop(fn=fn, parent_dir=tdir)
            crop.sow_combos(combos)
            for i in range(1, 10, 2):
                crop.grow(i)

            ds = crop.reap_combos_to_ds(var_names=var_names, var_dims=var_dims,
                                        allow_incomplete=True)

            num_finished = int(ds['x'].size * (5 / 9))
            assert (ds['x'] == ds_exp['x']).sum() == num_finished

            crop.grow_missing()
            ds = crop.reap_combos_to_ds(var_names=var_names, var_dims=var_dims,
                                        allow_incomplete=True)
            assert ds.identical(ds_exp)

    def test_new_ds_crop_loads_info_incomplete(self):
        def fn(a, b):
            return xr.Dataset({'sum': a + b, 'diff': a - b})

        with TemporaryDirectory() as tdir:
            disk_ds = os.path.join(tdir, 'test.h5')

            combos = dict(a=[1], b=[1, 2, 3])
            runner = Runner(fn, var_names=None)
            harvester = Harvester(runner, disk_ds)
            crop = harvester.Crop(name='fn', batchsize=1, parent_dir=tdir)
            crop.sow_combos(combos)
            for i in range(1, 3):
                crop.grow(i)

            # try creating crop from fresh
            c = Crop(name='fn', parent_dir=tdir)
            # crop's harvester should be loaded from disk
            assert c.farmer is not None
            assert c.farmer is not harvester
            ds = c.reap(allow_incomplete=True)
            assert isinstance(ds, xr.Dataset)
            assert ds['diff'].isnull().sum() == 1
            assert harvester.full_ds['diff'].isnull().sum() == 1

            # try creating crop from harvester
            c = harvester.Crop('fn', parent_dir=tdir)
            # crop's harvester should still be harvester
            assert c.farmer is not None
            assert c.farmer is harvester
            ds = c.reap(allow_incomplete=True)
            assert isinstance(ds, xr.Dataset)
            assert ds['diff'].isnull().sum() == 1

    def test_load_crops(self):

        combos = (('a', [1, 2, 3]),
                  ('b', [10, 20, 30]),
                  ('c', range(100, 1101, 100)))

        with TemporaryDirectory() as tdir:

            c1 = Crop(name='Alice', fn=foo_add, parent_dir=tdir)
            c2 = Crop(name='Bob', fn=foo_add, parent_dir=tdir)

            c1.sow_combos(combos)
            c2.sow_combos(combos)

            crops = load_crops(tdir)
            assert 'Alice' in crops
            assert 'Bob' in crops
            assert len(crops) == 2

            c1.grow_missing()
            c2.grow_missing()

            assert (c1.reap_combos() ==
                    c2.reap_combos() ==
                    combo_runner(foo_add, combos))
