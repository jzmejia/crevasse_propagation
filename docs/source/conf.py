# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../'))


# -- Project information -----------------------------------------------------

project = 'CrevProp'
copyright = '2022, Jessica Mejia'
author = 'Jessica Mejia'
version = '0.1'

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.duration',
    'sphinx.ext.napoleon',
    'numpydoc',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'furo'
html_theme = 'sphinx_material'
html_theme_options = {
    'nav_title' : 'crevProp',
    'repo_url' : 'https://github.com/jzmejia/crevasse_propagation',
    'repo_type' : 'github',
    'repo_name' : 'crevasse_propagation',
    'base_url' : 'https://crevasse-propagation.readthedocs.io/en/latest/',
    'css_minify': True,
    'html_minify': True
}

html_sidebars = {
    "**": ["globaltoc.html"]
}

html_logo = './_static/crack.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']


# -- Options for EPUB output ------------------------------------------------
epub_show_urls = 'footnote'


# add_module_names = False


# autosummary settings
autosummary_generate = True


# Napoleon settings
napoleon_google_docstring = False
napoleon_use_admonition_for_notes = True

