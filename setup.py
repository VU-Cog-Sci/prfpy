# -*- coding: utf-8 -*-

"""
Setup file for the *CMasher* package.
"""


# %% IMPORTS
# Built-in imports
from codecs import open
import re

# Package imports
from setuptools import find_packages, setup


# %% SETUP DEFINITION
# Get the long description from the README file
with open('README.rst', 'r') as f:
    long_description = f.read()

# Get the requirements list
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()


# Setup function declaration
setup(name="prfpy",
      version='0.0.1',
      author="Marco Aqil",
      author_email='marco.aqil@gmail.com',
      description=("prfpy: a package to fit and simulate population receptive field models"),
      long_description=long_description,
      url="https://github.com/VU-Cog-Sci/prfpy",
      project_urls={
          'Documentation': "https://github.com/VU-Cog-Sci/prfpy",
          'Source Code': "https://github.com/VU-Cog-Sci/prfpy",
          },
      license='GPL3',
      platforms=['Windows', 'Mac OS-X', 'Linux', 'Unix'],
      classifiers=["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: GPL-v3 License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"],
      python_requires='>=3.6, <4',
      packages=find_packages(),
      install_requires=requirements,
      zip_safe=False,
      )
