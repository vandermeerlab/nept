[![Build Status](https://travis-ci.org/vandermeerlab/nept.svg?branch=master)](https://travis-ci.org/vandermeerlab/nept)
[![Coverage Status](https://img.shields.io/codecov/c/github/vandermeerlab/nept/master.svg)](https://codecov.io/gh/vandermeerlab/nept/branch/master)
[![Documentation Status](https://readthedocs.org/projects/nept/badge/?version=latest)](https://readthedocs.org/projects/nept/?badge=latest)

nept - Neuroelectrophysiology tools
===================================

Formerly `vdmlab', renamed to emphasize general abilities of this module.

Getting started for beginners
=============================

* Download Miniconda from
  [Continuum Analytics](http://conda.pydata.org/miniconda.html).
  We support Python 3.
* Open a *new* terminal, create and activate a new conda environment.

  ```
  conda create -n yourenv python=3.5
  activate yourenv [Windows] or source activate yourenv [Linux]
  ```

* Install package dependencies (it's possible to
  install multiple packages at once or individually).
  If conda doesn't have a package of interest (eg. shapely),
  in the terminal try: `pip install shapely`.
  In Windows, download the most recent `*.whl` file
  [here](http://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely)
  and install using `pip install yourshapelyinstall.whl`
  (must be in the directory where this .whl is located).

  ```
  conda install matplotlib jupyter scipy numpy pandas pytest coverage
  ```

* Clone the analysis code from Github and use a developer installation.

  ```
  git clone https://github.com/vandermeerlab/nept.git
  cd nept
  python setup.py develop
  ```

* **All set!**

Documentation
=============

Users
-----

The latest version of the nept documentation is available
[here](http://nept.readthedocs.io/en/latest/index.html).

Developers
----------

```
conda install sphinx numpydoc mock
```

Build latest version of the documentation using 
`python setup.py build_sphinx` in the nept directory.

Testing
=======

Run tests with [pytest](http://docs.pytest.org/en/latest/usage.html).

Check coverage with [codecov](https://codecov.io/gh/vandermeerlab/nept).
Or 'py.test' and `coverage report' in the command line.


License
=======

The nept codebase is made available under made available 
under the [MIT license](LICENSE.md) 
that allows using, copying, and sharing.

The file `nept/nlx_loaders.py` contains code from 
[nlxio](https://github.com/bwillers/nlxio) by Bernard Willers, used with permission. 

Projects using nept
===================

* [emi_shortcut](https://github.com/vandermeerlab/emi_shortcut)
* [emi_biconditional](https://github.com/vandermeerlab/emi_biconditional)
