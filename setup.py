from setuptools import setup, find_packages
import os

setup(name='ccfepyutils',
      version='0.1',
      description='General purpose python tools',
      url='git@git.ccfe.ac.uk:tfarley/ccfepyutils.git',
      author='tfarley',
      author_email='tfarley@ukaea.uk',
      # license=ccfepyutils.__license__,
      packages=['ccfepyutils', 'ccfepyutils.classes', 'ccfepyutils.demos', 'ccfepyutils.guis', 'ccfepyutils.scripts'],
      # packages=find_packages(exclude=['docs', 'external', 'misc', 'tests', 'third_party']),
      package_data={
            'ccfepyutils': ['template_settings/values/*/*'],
            # 'ccfepyutils/template_settings': ['*']
      },
      include_package_data=True,
      install_requires=["numpy >= 1.12.0", "scipy", "matplotlib", "pandas", "xarray", "natsort", "PyQt5", "future"],  # "cv2",
      python_requires='>=3',
      setup_requires=['pytest-runner'],
      test_suite='tests.testsuite',
      tests_require=['pytest-cov'],
      zip_safe=False,
      long_description=open('README.md').read())