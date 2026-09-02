# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath('..'))

# Write a minimal .cdsapirc only when running in CI (env vars are set)
_cds_url = os.environ.get('CDSAPI_URL')
_cds_key = os.environ.get('CDSAPI_KEY')
if _cds_url and _cds_key:
    cdsapirc_path = Path.home() / ".cdsapirc"
    cdsapirc_path.write_text(f"url: {_cds_url}\nkey: {_cds_key}\n")

os.environ['SKIP_DATA_DOWNLOAD'] = '1'

project = 'oceanicospy'
copyright = '2025, OCEANICOS research group'
author = 'OCEANICOS research group'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

suppress_warnings = ["myst.header"]  # (si usas MyST)
autodoc_mock_imports = ["utide","geopandas","shapely"]

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # Optional: for Google or NumPy-style docstrings
    'sphinx.ext.viewcode',  # Optional, for source code links
    'nbsphinx',  # Optional, for Jupyter Notebook integration
    "myst_nb"
]

nbsphinx_execute = 'never'
nb_execution_mode = "off"  # never re-execute
nbsphinx_input_prompt = ""
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
autoclass_content = "both"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
#html_static_path = ['_static']
