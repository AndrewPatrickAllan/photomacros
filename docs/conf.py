# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------
project = 'photomacros'
copyright = '2024, AndrewPatrickAllan_Osmar234'
author = 'AndrewPatrickAllan_Osmar234'
release = '1'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',      # For generating documentation from docstrings
    'sphinx.ext.napoleon',     # For Google and NumPy style docstrings
    'sphinx.ext.viewcode',     # To include links to source code
]

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
html_theme = 'alabaster'
html_static_path = ['_static']

# Add your project directory to sys.path to allow imports
import os
import sys
# sys.path.insert(0, os.path.abspath('../src'))  # Update if your Python files are in a different directory
# sys.path.insert(0, os.path.abspath('../modeling/'))  # Update if your Python files are in a different directory
sys.path.insert(0, os.path.abspath('../'))

