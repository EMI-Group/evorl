# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "EvoRL"
copyright = "2025, Bowen Zheng"
author = "Bowen Zheng"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",
    # "sphinx.ext.coverage",
    # "sphinx_copybutton",
    "autodoc2",
]

autodoc2_packages = [
    "../evorl",
]

autodoc2_render_plugin = "myst"

autodoc2_docstring_parser_regexes = [
    (r".*", "autodoc2_docstrings_parser"),
]

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    # "tasklist",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
