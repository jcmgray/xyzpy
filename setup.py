from setuptools import setup, find_packages

setup(name='ximu',
      version='0.1.1',
      author='Johnnie Gray',
      license='MIT',
      packages=find_packages(exclude=['contrib', 'docs', 'test*']),
      install_requires=[
          'numpy>=1.10',
          'xarray>=7.0',
          'plotly>=1.9',
          'matplotlib>=1.5'],
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.5'])
