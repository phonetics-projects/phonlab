# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Phonlab'
copyright = '2026, Regents of the University of California'
author = 'Keith Johnson and Ronald Sprouse'

# Read the version from the source file rather than importing phonlab, so that
# the docs build does not depend on the package's dependencies being importable.
import re
from pathlib import Path
_init = Path(__file__).resolve().parents[2] / "phonlab" / "__init__.py"
release = re.search(r'^__version__\s*=\s*"([^"]+)"', _init.read_text(), re.M).group(1)
version = ".".join(release.split(".")[:2])   # short X.Y version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys
sys.path.insert(0, os.path.abspath('../../'))

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.viewcode', 
              'sphinx.ext.napoleon',
              'sphinx_rtd_theme']

autodoc_typehints = 'description'

templates_path = ['_templates']
exclude_patterns = ['.ipynb_checkpoints']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme = 'sphinx_rtd_theme'
html_logo = 'logo3.png'
#html_theme = 'cloud'
html_static_path = ['_static']
html_css_files = ['custom.css']
