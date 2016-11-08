import pytest
from xyzpy.gen.prepare import (
    _parse_var_names,
    _parse_var_dims,
)


class TestParsers:
    @pytest.mark.parametrize(
        " var_names,               expctd_var_names",
        [('x',                     ('x',)),
         (('x',),                  ('x',)),
         (('x', 'y'),              ('x', 'y')),
         (['x', 'y'],              ('x', 'y')),
         ((a for a in ('x', 'y')), ('x', 'y'))])
    def test_parse_var_names(self, var_names, expctd_var_names):
        var_names = _parse_var_names(var_names)
        assert var_names == expctd_var_names

    @pytest.mark.parametrize(
        " var_names,  var_dims,            expctd_var_dims",
        [('x',        None,                {'x': ()}),
         ('x',        [],                  {'x': ()}),
         ('x',        't',                 {'x': ('t',)}),
         ('x',        ['t'],               {'x': ('t',)}),
         ('x',        {'x': 't'},          {'x': ('t',)}),
         ('x',        {'x': ['t']},        {'x': ('t',)}),
         ('x',        [('x', ['t'])],      {'x': ('t',)}),
         (['x', 'y'], None,                {'x': (), 'y': ()}),
         (['x', 'y'], ['t', []],           {'x': ('t',), 'y': ()}),
         (['x', 'y'], {'x': 't'},          {'x': ('t',), 'y': ()}),
         (['x', 'y'], {('x', 'y'): 't'},   {'x': ('t',), 'y': ('t',)}),
         (['x', 'y'], {('x', 'y'): ['t']}, {'x': ('t',), 'y': ('t',)}),
         (['x', 'y'], ['t', 's'],          {'x': ('t',), 'y': ('s',)})])
    def test_parse_var_dims(self, var_names, var_dims, expctd_var_dims):
        var_dims = _parse_var_dims(var_dims, var_names)
        assert var_dims == expctd_var_dims

    @pytest.mark.parametrize(
        " var_names,    var_dims",
        [(['x', 'y'], [['t', 's']]),
         (['x', 'y'], 't'),
         (['x', 'y'], {'z': 't'})])
    def test_var_dims_raises(self, var_names, var_dims):
        with pytest.raises(ValueError):
            _parse_var_dims(var_dims, var_names)
