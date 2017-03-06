***********
Style guide
***********

Python
------
We adhere to `PEP8 <https://www.python.org/dev/peps/pep-0008/>`_.

Docstrings
----------
We use `numpydoc` and `NumPy's guidelines for docstrings <https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt>`_, since they're pretty readable in plain text and when rendered with Sphinx.

Git
---
We follow `Nengo's use of git <https://nengo/github.io/git.html>`_, with some modifications. 
In short, development happens on `Github <https://github.com/vandermeerlab/nept>`_, with
all commits in the master branch passing all the tests. Commit messages follow certain
guidelines outlined below. Only maintainers can merge anything into master and all development 
code must in a separate branch and pass through a pull request.

Commit messages
~~~~~~~~~~~~~~~
Commit messages should be in the following format.

.. code::

  Capitalized, short (50 chars or less) summary

  More detailed body text, if necessary. Wrap it to around 72 characters.
  The blank line separating the summary from the body is critical.

  Paragraphs must be separated by a blank line.

  - Bullet points are okay, too.
  - Typically a hyphen or asterisk is used for the bullet, followed by a
    single space, with blank lines before and after the list.
  - Use a hanging indent if the bullet point is longer than a 
    single point (like in this point).