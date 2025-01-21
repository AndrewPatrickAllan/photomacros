# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = 'photomacros'
copyright = '2024, AndrewPatrickAllan_Osmar234'
author = 'AndrewPatrickAllan_Osmar234'
release = '1'

# -- General configuration ---------------------------------------------------
# extensions = [
#     'sphinx_rtd_theme',
#     'sphinx.ext.autodoc',      # For generating documentation from docstrings
#     'sphinx.ext.napoleon',     # For Google and NumPy style docstrings
#     'sphinx.ext.viewcode',     # To include links to source code
#     'numpydoc',                # to make Sphinx recognize the NumPy style
# ]


extensions = [
    'sphinx.ext.autodoc',
    'numpydoc',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinx.ext.doctest',
    'sphinx.ext.autosummary',
    'sphinx.ext.coverage',
    'sphinx.ext.viewcode',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
]


# extensions = [
#     'sphinx.ext.autodoc',
#     'numpydoc',
#     'sphinx_copybutton',
#     'sphinx_design',
#     'sphinx.ext.doctest',
#     'sphinx.ext.autosummary',
#     'sphinx.ext.viewcode',
#     'IPython.sphinxext.ipython_console_highlighting',
#     'IPython.sphinxext.ipython_directive',
# ]


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for autodoc -----------------------------------------------------
autodoc_default_options = {
    'members': True,           # Include class members
    'undoc-members': True,     # Include undocumented members
    'show-inheritance': True,  # Show class inheritance
}
autodoc_typehints = "description"  # Show type hints in descriptions

# -- Options for HTML output -------------------------------------------------
# html_theme = 'sphinx_rtd_theme'
html_theme = 'pydata_sphinx_theme' # also used by numpy and pandas!
html_static_path = ['_static']

# Configuration of sphinx.ext.coverage
coverage_show_missing_items = True

# Add your project directory to sys.path to allow imports
import os
import sys
# sys.path.insert(0, os.path.abspath('../src'))  # Update if your Python files are in a different directory
# sys.path.insert(0, os.path.abspath('../modeling/'))  # Update if your Python files are in a different directory
# sys.path.insert(0, os.path.abspath('../'))
# sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../photomacros/'))

