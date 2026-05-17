=======
phonlab
=======

A collection of python functions for doing phonetics.

==============================
Installation and Documentation
==============================

* To install phonlab into an existing environment use pip:

.. code-block:: 

  pip install phonlab

* The phonlab documentation is here:  https://phonlab.readthedocs.io

* The github page for this package (https://github.com/phonetics-projects/phonlab) includes a yaml file that can be used to create a conda environment with all of the phonlab dependencies installed.  Use this command:

.. code-block:: 

  conda env create --file phonlab_env.yaml


========
Examples
========

See the `Phonlab Workshop slide deck <https://docs.google.com/presentation/d/1gfwlxLWZaZY7Zth8zP1LvmHAi21qQs6uBAm1Ep7DpYw/edit?usp=sharing>`_ for some background on this package.  There are example jupyter notebooks in the google drive linked in those slides.

See also the `examples` folder in the github repository!

=======
Testing
=======

To run the unit tests:

.. code-block::

  pip install pytest
  python -m pytest test/ -v

============
Contributing
============

We solicit contributions/corrections from the research community.  Here are some steps you can follow to contribute to this project.

#. Open a new issue in the https://github.com/phonetics-projects/phonlab repository.
#. Use git to Clone the repository.
#. Make your changes on your local clone of the repository

    * if you are adding a function, add it to the repository in a separate file, and write a good doc string for it
    * if you are correcting or extending an existing function, modify the existing file

#. If you are adding a function, also do these things:

    * edit the __init__.pyi file to include your file in the package, following the examples in that file.
    * edit the appropriate .rst file in the docs/source folder to include your function in the package documentation.

#. If your change requires a new python package, update the pyproject.toml and phonlab_env.yaml files to include the dependency.
#. Add/Commit your changes (preferably one commit per change rather than a set of unrelated changes in one big commit).
#. Push to your clone of the repository on github
#. Issue a pull request to add your change to the phonetics-projects/phonlab repository

    * the changes will be reviewed before they become part of the published repository
