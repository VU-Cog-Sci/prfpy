from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: GPL-v3 License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "prfpy: a package to fit and simulate population receptive fields"
# Long description will go up on the pypi page
long_description = """

pRFpy
========
pRFpy is a package that allows you to simulate 
and fit population receptive field (pRF) parameters from time series data.


To get started using these components in your own software, please go to the
repository README_.

.. _README: https://github.com/spinoza-centre/prfpy/blob/master/README.md

License
=======
``prfpy`` is licensed under the terms of the GPL-v3 license. See the file
"LICENSE" for information on the history of this software, terms & conditions
for usage, and a DISCLAIMER OF ALL WARRANTIES.

All trademarks referenced herein are property of their respective holders.

Copyright (c) 2019--, Tomas Knapen, 
Spinoza Centre for Neuroimaging, Amsterdam.
"""

NAME = "prfpy"
MAINTAINER = "Tomas Knapen"
MAINTAINER_EMAIL = "tknapen@gmail.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "https://github.com/spinoza-centre/prfpy"
DOWNLOAD_URL = ""
LICENSE = "GPL3"
AUTHOR = "Tomas Knapen"
AUTHOR_EMAIL = "tknapen@gmail.com"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {'prfpy': [pjoin('test', 'data', '*')]}
REQUIRES = ["numpy","scipy","statsmodels"]
