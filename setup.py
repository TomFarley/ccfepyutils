from setuptools import setup
import os

setup(name='ccfepyutils',
      version='0.1',
      description='General purpose python tools',
      url='git@git.ccfe.ac.uk:tfarley/ccfepyutils.git',
      author='tfarley',
      author_email='tfarley@ukaea.uk',
      # license=ccfepyutils.__license__,
      packages=['ccfepyutils'],
      install_requires=["numpy >= 1.12.0", "scipy", "matplotlib", "pandas", "xarray", "natsort", "PyQt5", "future"],  # "cv2",
      python_requires='>=3',
      setup_requires=['pytest-runner'],
      test_suite='tests.testsuite',
      tests_require=['pytest-cov'],
      zip_safe=False,
      long_description=open('README.md').read())