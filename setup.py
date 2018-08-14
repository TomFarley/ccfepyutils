from setuptools import setup
import os
import ccfepyutils

setup(name='ccfepyutils',
      version=ccfepyutils.__version__,
      description='General purpose python tools',
      url='git@git.ccfe.ac.uk:tfarley/ccfepyutils.git',
      author=ccfepyutils.__author__,
      author_email=ccfepyutils.__email__,
      license=ccfepyutils.__license__,
      packages=['ccfepyutils', 'ccfepyutils.tests'],
      install_requires=["numpy >= 1.12.0", "scipy", "matplotlib", "pandas", "xarray", "cv2", "natsort", "PyQt5"],
      python_requires='>=3',
      setup_requires=['pytest-runner'],
      test_suite='tests.testsuite',
      tests_require=['pytest-cov'],
      zip_safe=False,
      long_description=open('README.md').read())