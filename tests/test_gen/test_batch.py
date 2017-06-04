from tempfile import TemporaryDirectory
# import pytest

from xyzpy import combo_runner
from xyzpy.gen.batch import sow_combos, grow, reap_combos


def foo_add(a, b, c):
    return a + b


class TestSower:

    def test_batch(self):

        combos = [
            ('a', [10, 20, 30]),
            ('b', [4, 5, 6, 7]),
        ]

        expected = combo_runner(foo_add, combos, constants={'c': True})

        with TemporaryDirectory() as tdir:

            sow_combos(combos, constants={'c': True},
                       fn=foo_add, field_dir=tdir, batchsize=5)

            for i in range(1, 4):
                grow(i, field_dir=tdir, field_name='foo_add')

            results = reap_combos(field_dir=tdir, field_name='foo_add')

        assert results == expected
