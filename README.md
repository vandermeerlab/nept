Tools used for the analysis of neural recording data
====================================================

Getting started
===============

* Download Miniconda from
  [Continuum Analytics](http://conda.pydata.org/miniconda.html).
  We recommend the Python 3 version.
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
  conda install numpy scipy shapely matplotlib
  ```

* Clone the analysis code from Github and developer installation.

  ```
  git clone https://github.com/vandermeerlab/python-vdmlab.git
  cd python-vdmlab
  python setup.py develop
  ```

* **All set!**

Documentation
=============

Users
-----

The latest version of the vdmlab documentation is available
[here](http://python-vdmlab.readthedocs.io/en/latest/index.html).

Developers
----------

```
conda install sphinx numpydoc mock
```

Build latest version of the documentation using 
`python setup.py build_sphinx` in the vdmlab directory.

Testing
=======

Run tests with [pytest](http://docs.pytest.org/en/latest/usage.html).

```
conda install pytest
```

License
=======

The vdmlab codebase is made available under made available 
under the [MIT license](LICENSE.md) 
that allows using, copying, and sharing.

The file `vdmlab/nlx_loaders.py` contains code from 
[nlxio](https://github.com/bwillers/nlxio) by Bernard Willers, used with permission. 

Projects using vdmlab
=====================

* [emi_shortcut](https://github.com/vandermeerlab/emi_shortcut)
