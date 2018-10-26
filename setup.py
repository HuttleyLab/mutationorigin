#!/usr/bin/env python
from setuptools import setup
import sys

__author__ = "Gavin Huttley"
__copyright__ = "Copyright 2014, Gavin Huttley"
__credits__ = ["Yicheng Zhu", "Cheng Soon Ong", "Gavin Huttley"]
__license__ = "BSD"
__version__ = "0.2"
__maintainer__ = "Gavin Huttley"
__email__ = "Gavin.Huttley@anu.edu.au"
__status__ = "Development"

# Check Python version, no point installing if unsupported version inplace
if sys.version_info < (3, 5):
    py_version = ".".join([str(n) for n in sys.version_info])
    raise RuntimeError(
        "Python-3.5 or greater is required, Python-%s used." % py_version)

short_description = "mutation_origin"

# This ends up displayed by the installer
long_description = """mutation_origin
classifiers for mutation origin
Version %s.
""" % __version__

setup(
    name="mutation_origin",
    version=__version__,
    author="Yicheng Zhu",
    author_email="gavin.huttley@anu.edu.au",
    description=short_description,
    long_description=long_description,
    platforms=["any"],
    license=["GPL"],
    keywords=["biology", "genomics", "genetics", "statistics", "evolution",
              "bioinformatics"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Operating System :: OS Independent",
    ],
    packages=['mutation_origin'],
    install_requires=[
        'numpy',
        'cogent3',
        'pandas',
        'click',
        'sklearn',
        'scitrack',
        'scipy'
    ],
    # note: http://stackoverflow.com/questions/3472430/how-can-i-make-setuptools-install-a-package-thats-not-on-pypi
    # and http://stackoverflow.com/questions/17366784/setuptools-unable-to-use-link-from-dependency-links/17442663#17442663
    # changing it to http://github.com/mtai/python-gearman/tarball/master#egg=gearman-2.0.0beta instead
    entry_points={
        'console_scripts': ['mutori=mutation_origin.cli:main',
                            ],
    }
)
