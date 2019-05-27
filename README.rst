.. image:: https://img.shields.io/travis/vandermeerlab/nept/master.svg
  :target: https://travis-ci.org/vandermeerlab/nept
  :alt: Travis-CI build status

.. image:: https://img.shields.io/codecov/c/github/vandermeerlab/nept/master.svg
  :target: https://codecov.io/gh/vandermeerlab/nept/branch/master
  :alt: Test coverage

.. image:: https://img.shields.io/badge/docs-latest-blue.svg
  :target: https://vandermeerlab.github.io/nept/
  :alt: Documentation Status

**********************************
nept: Neuroelectrophysiology tools
**********************************

Getting started
===============

If you don't already have python 3, we recommend you download it using Miniconda 
from `Continuum Analytics <http://conda.pydata.org/miniconda.html>`_.

We recommend using a separate python environment.

Open a **new** terminal, create and activate a new conda environment::

  conda create -n yourenv python=3.6
  activate yourenv [Windows] or source activate yourenv [Linux/Mac]

Install package dependencies::

  conda install matplotlib jupyter scipy numpy pandas seaborn pytest coverage

For Shapely, try::

  pip install shapely

If that fails (usually in Windows) download the most recent wheel file 
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

Check `GitHub Pages <https://vandermeerlab.github.io/nept/>`_
for the latest version of the nept documentation.

License
=======

The nept codebase is made available under made available 
under the `MIT license <LICENSE.rst>`_
that allows using, copying and sharing.

The file ``nept/neuralynx_loaders.py`` contains code from 
`nlxio <https://github.com/bwillers/nlxio>`_ by Bernard Willers, 
used with permission. 

Example projects using nept in the vandermeer lab
=================================================

`emi_biconditional <https://github.com/vandermeerlab/emi_biconditional>`_

`emi_experience <https://github.com/vandermeerlab/emi_experience>`_

`emi_shortcut <https://github.com/vandermeerlab/emi_shortcut>`_