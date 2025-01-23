# Configuration file for the Sphinx documentation builder.
import sys
import os
# -- Project information -----------------------------------------------------
project = 'photomacros'
copyright = '2024, AndrewPatrickAllan_Osmar234'
author = 'AndrewPatrickAllan_Osmar234'
release = '1'


# -- General configuration ---------------------------------------------------
extensions = [
	'numpydoc',
	'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx_copybutton',
    'sphinx_design',
    'sphinx.ext.doctest',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'IPython.sphinxext.ipython_console_highlighting',
    'IPython.sphinxext.ipython_directive',
]
templates_path = ['_templates']
exclude_patterns = ['modeling/sklearn-env/**','_build', 'Thumbs.db', '.DS_Store']

# -- Options for autodoc -----------------------------------------------------
autodoc_default_options = {
    'members': True,           # Include class members
    'undoc-members': True,     # Include undocumented members
    'show-inheritance': True,  # Show class inheritance
}
autodoc_typehints = "description"  # Show type hints in descriptions
numpydoc_show_class_members=False
# -- Options for HTML output -------------------------------------------------
#html_theme = 'alabaster'

# Set the theme
html_theme = 'sphinx_rtd_theme'
#html_static_path = ['_static']
coverage_show_missing_items = True
# Optional: Customize theme options
html_theme_options = {
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}
# Add your project directory to sys.path to allow imports


