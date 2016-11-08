from xyzpy.utils import (
    progbar
)


class TestProgbar:
    def test_normal(self):
        for i in progbar(range(10)):
            pass

    def test_overide_ascii(self):
        for i in progbar(range(10), ascii=False):
            pass
