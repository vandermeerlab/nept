**********************************
nept: Neuroelectrophysiology tools
**********************************

Developer Guide
===============

Documentation
-------------

Ensure you have sphinx, numpydic, and mock::

  conda install ghp-import sphinx numpydoc sphinx_rtd_theme

Install nbsphinx so notebooks in the documentations can be executed::
  
  pip install nbsphinx --user

Build the latest version of the documentation using 
in the nept directory prior to pushing it to Github::

  sphinx-build docs docs/_build

And push it to Github::

  docs/update.sh

Style Guide
-----------

See the nept `style guide <https://github.com/vandermeerlab/nept/blob/master/style_guide.rst>`_.

Testing
-------

Run tests with `pytest <http://docs.pytest.org/en/latest/usage.html>`_.

Check coverage with `codecov <https://codecov.io/gh/vandermeerlab/nept>`_.