# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "EvoRL"
copyright = "2025, Bowen Zheng"
author = "Bowen Zheng"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx_copybutton",
    "autodoc2",
]

# autodoc2_packages = [
#     {
#         "path": "../evorl",
#         "auto_mode": True,
#     }
# ]
autodoc2_packages = [
    "../evorl",
]
autodoc2_render_plugin = "myst"
autodoc2_docstring_parser_regexes = [
    (r".*", "autodoc2_docstrings_parser"),
]

# avoid recursive apidoc build in these paths
# use __all__
autodoc2_module_all_regexes = [
    r"evorl\.distributed",
    r"evorl\.ec\.[a-zA-Z_]+",
    r"evorl\.envs\.wrappers",
    # r"evorl\.envs",
    r"evorl\.evaluators",
    r"evorl\.networks",
    r"evorl\.recorders",
    r"evorl\.replay_buffers",
    r"evorl\.workflows",
    # file level modules with __all__
    # r"evorl\.distribution",
    # r"evorl\.types",


]
autodoc2_class_docstring = "both"
autodoc2_hidden_objects = [
    # "undoc",
    "inherited",
    "private",
    "dunder",
]
autodoc2_hidden_regexes = [
    r"evorl\..*logger$",
]
# package with unstable API:
autodoc2_skip_module_regexes = [
    r"evorl\.rollout_ma",
    r"evorl\.multi_agent_rollout",
    r"evorl\.envs\.jaxmarl_envs",
    r"evorl\.envs\.multi_agent_env",
]
autodoc2_sort_names = True

viewcode_line_numbers = True

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "fieldlist",
    # "linkify",
]

# fix duplicate object description issue for class attributes (e.g.: dataclass)
napoleon_use_ivar = True
napoleon_include_special_with_doc = False

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
# html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

html_theme_options = {
    "light_logo": "evox_logo_light.png",
    "dark_logo": "evox_logo_dark.png",
}

html_favicon = "_static/favicon-32x32.ico"
