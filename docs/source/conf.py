# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "generative_diffusion"
copyright = "2025, Manuel Muñoz"
author = "Manuel Muñoz"
release = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "nbsphinx",
]

nbsphinx_execute = "never"

templates_path = ["_templates"]
exclude_patterns = []

language = "es"

import shutil
import os

# Copiar notebooks cada vez que compiles
NOTEBOOKS_SRC = os.path.abspath(os.path.join(__file__, "../../../demo_notebooks"))
NOTEBOOKS_DST = os.path.abspath(os.path.join(__file__, "../notebooks"))

if os.path.exists(NOTEBOOKS_DST):
    shutil.rmtree(NOTEBOOKS_DST)
shutil.copytree(NOTEBOOKS_SRC, NOTEBOOKS_DST)

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
