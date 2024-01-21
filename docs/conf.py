# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------

project = 'gqe'
copyright = '2024, NVIDIA Corporation'
author = 'NVIDIA Corporation'
release = '0.0.1'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx_copybutton',
    'sphinx.ext.autodoc'
]

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'furo'
