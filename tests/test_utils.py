import functools
from dask import delayed
from xyzpy.utils import (
    progbar,
    _get_fn_name,
)


class TestGetFnName:
    def test_normal(self):
        def foo(a, b):
            pass

        assert _get_fn_name(foo) == 'foo'

    def test_partial(self):
        def foo(a, b):
            pass

        pfoo = functools.partial(foo, b=2)
        assert _get_fn_name(pfoo) == 'foo'

    def test_delayed(self):
        @delayed
        def dfoo(a, b):
            pass

        assert _get_fn_name(dfoo) == 'dfoo'


class TestProgbar:
    def test_normal(self):
        for i in progbar(range(10)):
            pass

    def test_overide_ascii(self):
        for i in progbar(range(10), ascii=False):
            pass
