from setuptools import setup, find_packages

setup(
    name='xyzpy',
    version='0.1.1',
    author='Johnnie Gray',
    license='MIT',
    packages=find_packages(exclude=['contrib', 'docs', 'test*']),
    install_requires=[
        'numpy>=1.10.0',
        'dask>=0.11.1',
        'distributed>=1.13.2',
        'xarray>=0.8.0',
        'h5py>=2.6.0',
        'h5netcdf>=0.2.2',
        'tqdm>=4.7.6',
        'matplotlib>=1.5.0',
        'bokeh>=0.12.3',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
    ],
)
