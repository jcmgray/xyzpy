
# FIXTURES ------------------------------------------------------------------ #


def foo3_scalar(a, b, c):
    assert abs(a) >= 0
    assert abs(a) < 10
    assert abs(b) >= 10
    assert abs(b) < 100
    assert abs(c) >= 100
    assert abs(c) < 1000
    return a + b + c


def foo3_float_bool(a, b, c):
    assert abs(a) >= 0
    assert abs(a) < 10
    assert abs(b) >= 10
    assert abs(b) < 100
    assert abs(c) >= 100
    assert abs(c) < 1000
    return a + b + c, a % 2 == 0


def foo2_scalar(a, b):
    assert abs(a) >= 0
    assert abs(a) < 10
    assert abs(b) >= 10
    assert abs(b) < 100
    return a + b


def foo2_array(a, b):
    assert abs(a) >= 0
    assert abs(a) < 10
    assert abs(b) >= 10
    assert abs(b) < 100
    return [b + a + 0.1 * i for i in range(10)]


def foo2_array_bool(a, b):
    assert abs(a) >= 0
    assert abs(a) < 10
    assert abs(b) >= 10
    assert abs(b) < 100
    return [b + a + 0.1 * i for i in range(10)], a % 2 == 0


def foo2_array_array(a, b):
    assert abs(a) >= 0
    assert abs(a) < 10
    assert abs(b) >= 10
    assert abs(b) < 100
    return ([b + i * a for i in range(5)],
            [b - i * a for i in range(5)])


def foo2_zarray1_zarray2(a, b):
    assert abs(a) >= 0
    assert abs(a) < 10
    assert abs(b) >= 10
    assert abs(b) < 100
    return ([b + a + 0.1j * i for i in range(5)],
            [b + a - 0.1j * i for i in range(5)])


def foo_array_input(a, t):
    return tuple(a * x for x in t)


def foo2_dataset(a, b):
    import numpy as np
    import xarray as xr
    x = np.tile(a + b, (2, 3))
    return xr.Dataset({'x': (['t1', 't2'], x)})
