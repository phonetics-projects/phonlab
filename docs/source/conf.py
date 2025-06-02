# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Phonlab'
copyright = '2025, Regents of the University of California'
author = 'Keith Johnson and Ronald Sprouse'
release = "Beta"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))


extensions = ['sphinx.ext.autodoc','sphinx.ext.viewcode', 'sphinx.ext.napoleon']
#autoapi_options = ['members']

autodoc_typehints = 'description'

templates_path = ['_templates']
exclude_patterns = ['.ipynb_checkpoints']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_logo = 'logo3.png'
#html_theme = 'cloud'
html_static_path = ['_static']
