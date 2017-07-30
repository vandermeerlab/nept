Getting started
===============

If you don't already have python 3, we recommend you download it using Miniconda 
from `Continuum Analytics <http://conda.pydata.org/miniconda.html>`_.

We recommend using a separate python environment.

Open a **new** terminal, create and activate a new conda environment::

  conda create -n yourenv python=3.5
  activate yourenv [Windows] or source activate yourenv [Linux]

Install package dependencies::

  conda install matplotlib jupyter scipy numpy pandas pytest coverage

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

Set up a developer installation::

  cd nept
  python setup.py develop

All set! You're ready to start using the nept module.

  .. code-block:: python

    import nept
