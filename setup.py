#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
  name = 'phonlab',
  version = '0.15.1',
  packages = ['phonlab', 'phonlab.third_party',
              'phonlab.acoustic','phonlab.artic',
              'phonlab.auditory','phonlab.utils'],
  install_requires=[
    'importlib_resources; python_version < "3.9"',
    'numpy', 'pandas', 'librosa',
    'nitime','matplotlib',
    'praat-parselmouth', 'scipy',
    'soundfile',
  ],
  scripts = [
  ],
  package_data = {'phonlab': ['*.txt', 'data/noise/*.wav']},
  classifiers = [
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering',
    'Topic :: Multimedia :: Sound/Audio :: Speech'
  ]
)
