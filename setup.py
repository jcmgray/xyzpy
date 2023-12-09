from setuptools import setup, find_packages


def readme():
    with open("README.md") as f:
        long_desc = f.read()
        # strip out the raw html images?
        return long_desc


setup(
    name="xyzpy",
    description="Easily generate large parameter space data",
    long_description=readme(),
    url="http://xyzpy.readthedocs.io",
    project_urls={  # Optional
        "Bug Reports": "https://github.com/jcmgray/xyzpy/issues",
        "Source": "https://github.com/jcmgray/xyzpy/",
    },
    author="Johnnie Gray",
    author_email="johnniemcgray@gmail.com",
    license="MIT",
    packages=find_packages(exclude=["docs", "test*"]),
    entry_points={
        "console_scripts": ["xyzpy-grow = xyzpy.gen.xyzpy_grow_cli:main"]
    },
    install_requires=[
        "numpy>=1.10.0",
        "dask>=0.11.1",
        "xarray>=0.9.0",
        "pandas>=0.20",
        "h5py>=2.6.0",
        "h5netcdf>=0.2.2",
        "joblib>=0.12",
        "tqdm>=4.7.6",
    ],
    extras_require={
        "tests": [
            "coverage",
            "pytest",
            "pytest-cov",
        ],
        "plotting": [
            "matplotlib",
            "bokeh",
        ],
        "docs": [
            "matplotlib",
            "bokeh",
            "furo",
            "ipython!=8.7.0",
            "myst-nb",
            "setuptools_scm",
            "sphinx-autoapi",
            "astroid<3.0.0",
            "sphinx-copybutton",
            "sphinx-design",
            "sphinx>=2.0",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
