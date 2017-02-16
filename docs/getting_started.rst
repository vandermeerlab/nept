Getting Started
===============

Download Python
---------------
Download Miniconda from
`Continuum Analytics <http://conda.pydata.org/miniconda.html>`_.
We recommend the Python 3 version.

Open a *new* terminal, create and activate a new conda environment.

  .. code-block:: bash

    conda create -n yourenv python=3.5
    activate yourenv [Windows] or source activate yourenv [Linux]

Install package dependencies (it's possible to
install multiple packages at once or individually).

  .. code-block:: bash

    conda install numpy scipy shapely matplotlib

If conda doesn't have a package of interest (eg. shapely),
in the terminal try: ``pip install shapely``.
In Windows, you may need to download the most recent ``*.whl`` file
`here <http://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely>`_
and install using ``pip install yourshapelyinstall.whl``
(remember, you must be in the directory where this .whl is located).

Clone nept from Github
----------------------

Clone the analysis code from Github.

  .. code-block:: bash

    git clone https://github.com/vandermeerlab/nept.git

Set up a developer installation.

  .. code-block:: bash

    cd nept
    python setup.py develop

All set! You're ready to start using the nept module.

  .. code-block:: python

    import nept
