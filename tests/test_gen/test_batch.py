from tempfile import TemporaryDirectory

import pytest

from xyzpy import combo_runner
from xyzpy.gen.batch import (
    XYZError,
    parse_crop_details,
    combos_sow,
    grow,
    combos_reap,
    combos_reap_to_ds,
)

from . import foo3_scalar


def foo_add(a, b, c):
    return a + b


class TestSowerReaper:
    @pytest.mark.parametrize(
        "fn, crop_name, crop_dir, expected",
        [
            (foo_add, None, None, '.xyz-foo_add'),
            (None, 'custom', None, '.xyz-custom'),
            (foo_add, 'custom', None, '.xyz-custom'),
            (foo_add, None, 'custom_dir', 'custom_dir/.xyz-foo_add'),
            (None, 'custom', 'custom_dir', 'custom_dir/.xyz-custom'),
            (foo_add, 'custom', 'custom_dir', 'custom_dir/.xyz-custom'),
            (None, None, None, 'raises'),
            (None, None, 'custom_dir', 'raises'),

        ])
    def test_parse_field_details(self, fn, crop_name, crop_dir, expected):

        if expected == 'raises':
            with pytest.raises(ValueError):
                parse_crop_details(fn, crop_name, crop_dir)
        else:
            crop_dir = parse_crop_details(fn, crop_name, crop_dir)
            assert crop_dir[-len(expected):] == expected

    def test_checks(self):
        combos = {'a': [1]}

        with pytest.raises(ValueError):
            combos_sow(combos, crop_name='custom', save_fn=True)

        with pytest.raises(TypeError):
            combos_sow(combos, fn=foo_add, save_fn=False, batchsize=0.5)

        with pytest.raises(ValueError):
            combos_sow(combos, fn=foo_add, save_fn=False, batchsize=-1)

        with pytest.raises(XYZError):
            grow(1)

    def test_batch(self):

        combos = [
            ('a', [10, 20, 30]),
            ('b', [4, 5, 6, 7]),
        ]
        expected = combo_runner(foo_add, combos, constants={'c': True})

        with TemporaryDirectory() as tdir:

            # sow seeds
            combos_sow(combos, constants={'c': True},
                       fn=foo_add, crop_dir=tdir, batchsize=5)

            # grow seeds
            for i in range(1, 4):
                grow(i, crop_dir=tdir, crop_name='foo_add')

            # reap results
            results = combos_reap(crop_dir=tdir, crop_name='foo_add')

        assert results == expected

    def test_field_name_and_overlapping(self):
        combos1 = [
            ('a', [10, 20, 30]),
            ('b', [4, 5, 6, 7]),
        ]
        expected1 = combo_runner(foo_add, combos1, constants={'c': True})

        combos2 = [
            ('a', [40, 50, 60]),
            ('b', [4, 5, 6, 7]),
        ]
        expected2 = combo_runner(foo_add, combos2, constants={'c': True})

        with TemporaryDirectory() as tdir:
            # sow seeds
            combos_sow(combos1, constants={'c': True}, crop_name='run1',
                       fn=foo_add, crop_dir=tdir, batchsize=5)
            combos_sow(combos2, constants={'c': True}, crop_name='run2',
                       fn=foo_add, crop_dir=tdir, batchsize=5)

            # grow seeds
            for i in range(1, 4):
                grow(i, crop_dir=tdir, crop_name='run1')
                grow(i, crop_dir=tdir, crop_name='run2')

            # reap results
            results1 = combos_reap(crop_dir=tdir, crop_name='run1')
            results2 = combos_reap(crop_dir=tdir, crop_name='run2')

        assert results1 == expected1
        assert results2 == expected2

    def test_combo_reaper_to_ds(self):
        combos = (('a', [1, 2]),
                  ('b', [10, 20, 30]),
                  ('c', [100, 200, 300, 400]))

        with TemporaryDirectory() as tdir:

            # sow seeds
            combos_sow(combos, fn=foo3_scalar, crop_dir=tdir, batchsize=5)

            # grow seeds
            for i in range(1, 6):
                grow(i, crop_dir=tdir, crop_name='foo3_scalar')

            ds = combos_reap_to_ds(crop_dir=tdir, crop_name='foo3_scalar',
                                   var_names=['bananas'])
        assert ds.sel(a=2, b=30, c=400)['bananas'].data == 432
