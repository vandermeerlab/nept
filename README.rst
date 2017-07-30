.. image:: https://img.shields.io/travis/vandermeerlab/nept/master.svg
  :target: https://travis-ci.org/vandermeerlab/nept
  :alt: Travis-CI build status

.. image:: https://img.shields.io/codecov/c/github/vandermeerlab/nept/master.svg
  :target: https://codecov.io/gh/vandermeerlab/nept/branch/master
  :alt: Test coverage

.. image:: https://readthedocs.org/projects/nept/badge/?version=latest
  :target: http://nept.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status

**********************************
nept: Neuroelectrophysiology tools
**********************************

Formerly ``vdmlab``, renamed to emphasize general abilities of this library.

Getting started
===============

If you don't already have python 3, we recommend you download it using Miniconda 
from `Continuum Analytics <http://conda.pydata.org/miniconda.html>`_.

We recommend using a separate python environment.

Open a **new** terminal, create and activate a new conda environment::

  conda create -n yourenv python=3.5
  activate yourenv [Windows] or source activate yourenv [Linux]

Install package dependencies::

  conda install matplotlib jupyter scipy numpy pandas seaborn pytest coverage

For Shapely, try::

  pip install shapely

If that fails, in Windows, download the most recent wheel file 
`here <http://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely>`_.
Once downloaded, install with wheel.

.. code:: bash

  pip install yourshapelyinstall.whl

Installation
============

Clone nept from Github and use a developer installation::

  git clone https://github.com/vandermeerlab/nept.git
  cd nept
  python setup.py develop

Documentation
=============

Users
-----

Check `GitHub Pages <https://vandermeerlab.github.io/nept/>`_
for the latest version of the nept documentation.

Developers
----------

Ensure you have sphinx, numpydic, and mock::

  conda install ghp-import sphinx numpydoc sphinx_rtd_theme

Install nbsphinx so notebooks in the documentations can be executed::
  
  pip install nbsphinx --user

Build the latest version of the documentation using 
in the nept directory prior to pushing it to Github::

  sphinx-build docs docs/_build

And push it to Github::

  docs/update.sh

Testing
=======

Run tests with `pytest <http://docs.pytest.org/en/latest/usage.html>`_.

Check coverage with `codecov <https://codecov.io/gh/vandermeerlab/nept>`_.

License
=======

The nept codebase is made available under made available 
under the `MIT license <LICENSE.rst>`_
that allows using, copying and sharing.

The file ``nept/neuralynx_loaders.py`` contains code from 
`nlxio <https://github.com/bwillers/nlxio>`_ by Bernard Willers, 
used with permission. 

Projects using nept
===================

`emi_shortcut <https://github.com/vandermeerlab/emi_shortcut>`_

`emi_biconditional <https://github.com/vandermeerlab/emi_biconditional>`_
