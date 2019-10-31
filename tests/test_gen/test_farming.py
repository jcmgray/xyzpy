import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import numpy as np
import xarray as xr

from xyzpy.manage import load_ds, load_df
from xyzpy.gen.farming import Runner, Harvester, Sampler, label
from xyzpy.gen.batch import grow


# -------------------------------- Fixtures --------------------------------- #

def fn1_x(x):
    return x ** 2


def fn3_fba(a, b, c):
    sm = a + b + c
    ev = ((a + b + c) % 2 == 0)
    ts = a * (b * np.linspace(0, 1.0, 3) + c)
    return sm, int(ev), ts


@pytest.fixture
def fn3_fba_runner():
    r = Runner(fn3_fba, fn_args=('a', 'b'),
               var_names=['sum', 'even', 'array'],
               var_dims={'array': ['time']},
               var_coords={'time': np.linspace(0, 1.0, 3)},
               constants={'c': 100},
               attrs={'fruit': 'apples'})
    return r


@pytest.fixture
def fn3_fba_ds():
    return xr.Dataset(
        coords={'a': [1, 2], 'b': [3, 4], 'time': np.linspace(0, 1.0, 3)},
        data_vars={'sum': (('a', 'b'), [[104, 105],
                                        [105, 106]]),
                   'even': (('a', 'b'), [[True, False],
                                         [False, True]]),
                   'array': (('a', 'b', 'time'),
                             [[[100, 101.5, 103], [100, 102, 104]],
                              [[200, 203, 206], [200, 204, 208]]])},
        attrs={'c': 100, 'fruit': 'apples'})


# ------------------------------ Runner Tests ------------------------------- #

class TestRunner:

    def test_basic(self):
        r = Runner(fn1_x, "x^2")
        print(str(r))
        print(repr(r))
        assert r(2) == 4

    @pytest.mark.parametrize("prop, vals", [('var_names', ['S', 'E', 'A']),
                                            ('fn_args', ['A', 'B', 'C']),
                                            ('var_dims', {'array': ['t']}),
                                            ('var_coords', {'time': [1, 2]}),
                                            ('constants', [('c', 200)]),
                                            ('resources', [('c', 300)])])
    def test_properties(self, prop, vals):
        r = Runner(fn3_fba, fn_args=('a', 'b', 'c'),
                   var_names=['sum', 'even', 'array'],
                   var_dims={'array': ['time']},
                   var_coords={'time': np.linspace(0, 1.0, 3)},
                   constants={'c': 100},
                   attrs={'fruit': 'apples'})
        getattr(r, prop)
        setattr(r, prop, vals)
        delattr(r, prop)
        assert getattr(r, prop) is None

    @pytest.mark.parametrize("var_names", ['sq', ('sq',), ['sq']])
    @pytest.mark.parametrize("var_dims", [[], (), {}, {'sq': ()}])
    @pytest.mark.parametrize("combos", [('x', (1, 2, 3)),
                                        (('x', (1, 2, 3)),),
                                        ('x', [1, 2, 3]),
                                        [('x', [1, 2, 3])],
                                        {'x': (1, 2, 3)},
                                        {'x': [1, 2, 3]}])
    def test_runner_combos_parse(self, var_names, var_dims, combos):
        r = Runner(fn1_x, var_names=var_names, var_dims=var_dims)
        r.run_combos(combos)
        expected_ds = xr.Dataset(coords={'x': [1, 2, 3]},
                                 data_vars={'sq': ('x', [1, 4, 9])})
        assert r.last_ds.identical(expected_ds)

    def test_runner_combos(self, fn3_fba_runner, fn3_fba_ds):
        r = fn3_fba_runner
        r.run_combos((('a', (1, 2)), ('b', (3, 4))))
        assert r.last_ds.identical(fn3_fba_ds)

    def test_sow_reap_seperate(self, fn3_fba_runner, fn3_fba_ds):
        with tempfile.TemporaryDirectory() as tmpdir:
            r = fn3_fba_runner
            crop = r.Crop(parent_dir=tmpdir, num_batches=2)
            crop.sow_combos((('a', (1, 2)),
                             ('b', (3, 4))))
            for i in [1, 2]:
                grow(i, crop)
            crop.reap()
            assert r.last_ds.identical(fn3_fba_ds)

    def test_sow_and_reap(self, fn3_fba_runner, fn3_fba_ds):
        with tempfile.TemporaryDirectory() as tmpdir:
            r = fn3_fba_runner
            crop = r.Crop(parent_dir=tmpdir, num_batches=2)

            def concurrent_grow():
                # wait for cases to be sown
                time.sleep(0.5)
                for i in [1, 2]:
                    grow(i, crop)

            with ThreadPoolExecutor(1) as pool:
                pool.submit(concurrent_grow)

                crop.sow_combos((('a', (1, 2)),
                                 ('b', (3, 4))))
                crop.reap(wait=True)

            assert r.last_ds.identical(fn3_fba_ds)

    @pytest.mark.parametrize("cases", [(1, 2, 3),
                                       ([1], [2], [3])])
    def test_runner_cases_parse(self, cases):
        r = Runner(fn1_x, var_names='sq', fn_args='x')
        r.run_cases(cases)
        expected_ds = xr.Dataset(coords={'x': [1, 2, 3]},
                                 data_vars={'sq': ('x', [1, 4, 9])})
        assert r.last_ds.identical(expected_ds)

    @pytest.mark.parametrize("dict_cases", [False, True])
    def test_runner_cases(self, fn3_fba_runner, dict_cases):

        if dict_cases:
            cases = [{'a': 2, 'b': 3}, {'a': 1, 'b': 4}]
        else:
            cases = [(2, 3), (1, 4)]

        fn3_fba_runner.run_cases(cases)
        expected_ds = xr.Dataset(
            coords={'a': [1, 2], 'b': [3, 4], 'time': np.linspace(0, 1.0, 3)},
            data_vars={'sum': (('a', 'b'), [[np.nan, 105],
                                            [105, np.nan]]),
                       'even': (('a', 'b'), [[None, False],
                                             [False, None]]),
                       'array': (('a', 'b', 'time'),
                                 [[[np.nan, np.nan, np.nan],
                                   [100, 102, 104]],
                                  [[200, 203, 206],
                                   [np.nan, np.nan, np.nan]]])},
            attrs={'c': 100, 'fruit': 'apples'})
        assert fn3_fba_runner.last_ds.identical(expected_ds)


# -------------------------- Harvester Tests -------------------------------- #

class TestHarvester:

    def test_save_and_load_ds(self, fn3_fba_runner, fn3_fba_ds):
        with tempfile.TemporaryDirectory() as tmpdir:
            fl_pth = os.path.join(tmpdir, 'test.h5')
            h = Harvester(fn3_fba_runner, fl_pth, full_ds=fn3_fba_ds)
            h.save_full_ds()
            h.load_full_ds()
            assert h.full_ds.equals(fn3_fba_ds)

    def test_harvest_combos_new(self, fn3_fba_runner, fn3_fba_ds):
        with tempfile.TemporaryDirectory() as tmpdir:
            fl_pth = os.path.join(tmpdir, 'test.h5')
            h = Harvester(fn3_fba_runner, fl_pth)
            h.harvest_combos((('a', (1, 2)), ('b', (3, 4))))
            hds = load_ds(fl_pth)
        assert h.last_ds.identical(fn3_fba_ds)
        assert h.full_ds.identical(fn3_fba_ds)
        assert hds.identical(fn3_fba_ds)

    def test_harvest_combos_new_sow_reap_separate(self, fn3_fba_runner,
                                                  fn3_fba_ds):
        with tempfile.TemporaryDirectory() as tmpdir:
            fl_pth = os.path.join(tmpdir, 'test.h5')
            h = Harvester(fn3_fba_runner, fl_pth)
            crop = h.Crop(parent_dir=tmpdir, num_batches=2)

            crop.sow_combos((('a', (1, 2)), ('b', (3, 4))))

            for i in [1, 2]:
                grow(i, crop)
            crop.reap()

            hds = load_ds(fl_pth)

            assert h.last_ds.identical(fn3_fba_ds)
            assert h.full_ds.identical(fn3_fba_ds)
            assert hds.identical(fn3_fba_ds)

    def test_harvest_combos_sow_and_reap_thread_wait(self, fn3_fba_runner,
                                                     fn3_fba_ds):
        with tempfile.TemporaryDirectory() as tmpdir:
            fl_pth = os.path.join(tmpdir, 'test.h5')
            h = Harvester(fn3_fba_runner, fl_pth)
            crop = h.Crop(parent_dir=tmpdir, num_batches=2)

            def concurrent_grow():
                # wait for cases to be sown
                time.sleep(0.5)
                for i in [1, 2]:
                    grow(i, crop)

            with ThreadPoolExecutor(1) as pool:
                pool.submit(concurrent_grow)
                crop.sow_combos((('a', (1, 2)), ('b', (3, 4))))
                crop.reap(wait=True)

            hds = load_ds(fl_pth)

        assert h.last_ds.identical(fn3_fba_ds)
        assert h.full_ds.identical(fn3_fba_ds)
        assert hds.identical(fn3_fba_ds)

    def test_harvest_combos_new_sow_reap_incomplete(self, fn3_fba_runner,
                                                    fn3_fba_ds):
        with tempfile.TemporaryDirectory() as tmpdir:
            fl_pth = os.path.join(tmpdir, 'test.h5')
            h = Harvester(fn3_fba_runner, fl_pth)
            crop = h.Crop(parent_dir=tmpdir, num_batches=3)
            crop.sow_combos((('a', (1, 2)), ('b', (3, 4))))

            grow(1, crop)
            crop.reap(allow_incomplete=True)
            assert not h.full_ds.identical(fn3_fba_ds)
            hds = load_ds(fl_pth)
            assert not hds.identical(fn3_fba_ds)

            crop.grow_missing()
            crop.reap(allow_incomplete=True)
            h.load_full_ds()
            assert h.full_ds.equals(fn3_fba_ds)
            hds = load_ds(fl_pth)
            assert hds.identical(fn3_fba_ds)

    def test_harvest_combos_merge(self, fn3_fba_runner, fn3_fba_ds):
        with tempfile.TemporaryDirectory() as tmpdir:
            fl_pth = os.path.join(tmpdir, 'test.h5')
            h = Harvester(fn3_fba_runner, fl_pth)
            h.harvest_combos((('a', (1,)), ('b', (3, 4))))
            h.harvest_combos((('a', (2,)), ('b', (3, 4))))
            hds = load_ds(fl_pth)
        assert not h.last_ds.identical(fn3_fba_ds)
        assert h.full_ds.identical(fn3_fba_ds)
        assert hds.identical(fn3_fba_ds)

    def test_harvest_combos_overwrite(self, fn3_fba_runner, fn3_fba_ds):
        with tempfile.TemporaryDirectory() as tmpdir:
            fl_pth = os.path.join(tmpdir, 'test.h5')
            mod_ds = fn3_fba_ds.copy(deep=True)
            mod_ds['array'].loc[{'a': 1, 'b': 3}] = 999
            h = Harvester(fn3_fba_runner, fl_pth, full_ds=mod_ds)
            h.save_full_ds()
            assert not h.full_ds.equals(fn3_fba_ds)
            h.harvest_combos((('a', (1,)), ('b', (3,))), overwrite=True)
            assert h.full_ds.equals(fn3_fba_ds)

    @pytest.mark.parametrize('dict_cases', [False, True])
    def test_harvest_cases_new(self, fn3_fba_runner, fn3_fba_ds, dict_cases):

        if dict_cases:
            cases = [{'a': 1, 'b': 3}, {'a': 1, 'b': 4},
                     {'a': 2, 'b': 3}, {'a': 2, 'b': 4}]
        else:
            cases = [(1, 3), (1, 4), (2, 3), (2, 4)]

        with tempfile.TemporaryDirectory() as tmpdir:
            fl_pth = os.path.join(tmpdir, 'test.h5')
            h = Harvester(fn3_fba_runner, fl_pth)
            h.harvest_cases(cases)
            hds = load_ds(fl_pth)
        assert h.last_ds.identical(fn3_fba_ds)
        assert h.full_ds.identical(fn3_fba_ds)
        assert hds.identical(fn3_fba_ds)

    def test_harvest_cases_merge(self, fn3_fba_runner, fn3_fba_ds):
        with tempfile.TemporaryDirectory() as tmpdir:
            fl_pth = os.path.join(tmpdir, 'test.h5')
            h = Harvester(fn3_fba_runner, fl_pth)
            h.harvest_cases([(1, 3), (2, 4)])
            h.harvest_cases([(1, 4), (2, 3)])
            hds = load_ds(fl_pth)
        assert not h.last_ds.identical(fn3_fba_ds)
        assert h.full_ds.identical(fn3_fba_ds)
        assert hds.identical(fn3_fba_ds)

    def test_harvest_cases_overwrite(self, fn3_fba_runner, fn3_fba_ds):
        with tempfile.TemporaryDirectory() as tmpdir:
            fl_pth = os.path.join(tmpdir, 'test.h5')
            mod_ds = fn3_fba_ds.copy(deep=True)
            mod_ds['array'].loc[{'a': 1, 'b': 3}] = 999
            h = Harvester(fn3_fba_runner, fl_pth, full_ds=mod_ds)
            h.save_full_ds()
            assert not h.full_ds.equals(fn3_fba_ds)
            h.harvest_cases([(1, 3)], overwrite=True)
            assert h.full_ds.equals(fn3_fba_ds)

    @pytest.mark.parametrize('engine', ['h5netcdf', 'netcdf4', 'zarr'])
    def test_harvest_cases_merge_dask(self, fn3_fba_runner,
                                      fn3_fba_ds, engine):
        import dask
        dask.config.set(scheduler='threads')

        with tempfile.TemporaryDirectory() as tmpdir:
            fl_pth = os.path.join(tmpdir, 'test.h5')
            h = Harvester(fn3_fba_runner, fl_pth, engine=engine)
            h.harvest_cases([(1, 3), (2, 4)], chunks=1)
            h.harvest_cases([(1, 4), (2, 3)], chunks=1)
            assert not h.last_ds.identical(fn3_fba_ds)
            assert h.full_ds.identical(fn3_fba_ds)
            assert h.full_ds['sum'].chunks is not None
            h.full_ds.close()

    def test_harvest_cases_merge_dask_default(self, fn3_fba_runner,
                                              fn3_fba_ds):

        import dask
        dask.config.set(scheduler='threads')

        with tempfile.TemporaryDirectory() as tmpdir:
            fl_pth = os.path.join(tmpdir, 'test.h5')
            h = Harvester(fn3_fba_runner, fl_pth, engine='netcdf4',
                          chunks=1)
            h.harvest_cases([(1, 3), (2, 4)])
            h.harvest_cases([(1, 4), (2, 3)])
            assert not h.last_ds.identical(fn3_fba_ds)
            assert h.full_ds.identical(fn3_fba_ds)
            assert h.full_ds['sum'].chunks is not None
            h.full_ds.close()


class TestSampler:

    @pytest.mark.parametrize("fname,engine", [
        ('test.pkl', 'pickle'),
        ('test.csv', 'csv'),
    ])
    def test_sample_combos_new(self, fname, engine):

        @label(var_names=['sum', 'diff', 'divisor', 'const'],
               constants={'c': 42})
        def sum_diff(a, b, c):
            return a + b, a - b, a % b == 0, c

        with tempfile.TemporaryDirectory() as tmpdir:
            fl_pth = os.path.join(tmpdir, fname)
            s = Sampler(sum_diff, fl_pth, engine=engine)
            print(s)
            combos = (
                ('a', (1, 2, 3, 4, 5)),
                ('b', lambda: np.random.randint(10, 20))
            )
            s.sample_combos(10, combos)
            hdf = load_df(fl_pth, engine=engine)

            assert len(hdf) == 10
            assert s.last_df.equals(hdf)
            assert s.full_df.equals(hdf)

            s.sample_combos(20, combos)
            hdf = load_df(fl_pth, engine=engine)
            assert len(hdf) == 30
            assert not s.last_df.equals(hdf)
            assert s.full_df.equals(hdf)

            assert set(s.full_df['const']) == set(s.full_df['c']) == {42}

            for col in ['sum', 'diff', 'divisor', 'const', 'a', 'b', 'c']:
                assert col in s.full_df

    @pytest.mark.parametrize('batchsize', [1, 3])
    def test_sow_reap_samples(self, batchsize):

        @label(var_names=['sum', 'diff', 'divisor', 'const'],
               constants={'c': 42})
        def sum_diff(a, b, c):
            return a + b, a - b, a % b == 0, c

        default_combos = (
            ('a', (1, 2, 3, 4, 5)),
            ('b', lambda: np.random.randint(10, 20))
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            fl_pth = os.path.join(tmpdir, 'test.pkl')
            s = Sampler(sum_diff, fl_pth, default_combos=default_combos)
            c = s.Crop('test-crop-def', batchsize=batchsize, parent_dir=tmpdir)
            c.sow_samples(10)
            c.grow_missing()
            c.reap()
            hdf = load_df(fl_pth)
            assert len(hdf) == 10
            assert s.last_df.equals(hdf)
            assert s.full_df.equals(hdf)
