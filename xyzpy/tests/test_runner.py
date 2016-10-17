from pytest import mark, raises

from ..gen.runner import parse_output_mapping


class TestParseOutputMapping:
    @mark.parametrize(
        " outputs,                 expctd_outputs",
        [('x',                     ('x',)),
         (('x',),                  ('x',)),
         (('x', 'y'),              ('x', 'y')),
         (['x', 'y'],              ('x', 'y')),
         ((a for a in ('x', 'y')), ('x', 'y'))])
    def test_parse_outputs(self, outputs, expctd_outputs):
        output_dims = {'x': 't'}
        outputs, output_dims = parse_output_mapping(outputs, output_dims)
        assert outputs == expctd_outputs

    @mark.parametrize(
        " outputs,    output_dims,         expctd_output_dims",
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
    def test_parse_output_dims(self, outputs, output_dims, expctd_output_dims):
        outputs, output_dims = parse_output_mapping(outputs, output_dims)
        assert output_dims == expctd_output_dims

    @mark.parametrize(
        " outputs,    output_dims",
        [(['x', 'y'], [['t', 's']]),
         (['x', 'y'], 't'),
         (['x', 'y'], {'z': 't'})])
    def test_raises(self, outputs, output_dims):
        with raises(ValueError):
            parse_output_mapping(outputs, output_dims)
