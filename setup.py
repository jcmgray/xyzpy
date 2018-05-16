from setuptools import setup, find_packages
import versioneer


def readme():
    with open('README.rst') as f:
        import re
        long_desc = f.read()
        # strip out the raw html images
        long_desc = re.sub('\.\. raw::[\S\s]*?>\n\n', "", long_desc)
        return long_desc


setup(
    name='xyzpy',
    description='Easily generate large parameter space data',
    long_description=readme(),
    url='http://xyzpy.readthedocs.io',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author='Johnnie Gray',
    author_email="john.gray.14@ucl.ac.uk",
    license='MIT',
    packages=find_packages(exclude=['docs', 'test*']),
    install_requires=[
        'numpy>=1.10.0',
        'dask>=0.11.1',
        'xarray>=0.9.0',
        'h5py>=2.6.0',
        'h5netcdf>=0.2.2',
        'joblib>=0.10',
        'tqdm>=4.7.6',
        'matplotlib>=2.2.0',
        'bokeh>=0.12.3',
        'cytoolz>=0.8',
    ],
    extras_require={
        'tests': [
            'coverage',
            'pytest',
            'pytest-cov',
        ],
        'docs': [
            'sphinx',
            'sphinx_bootstrap_theme',
            'nbsphinx',
            'ipython',
        ]
    },
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
