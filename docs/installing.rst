Installation: user guide
========================

.. note::
   **oceanicospy** requires Python 3.7 or later.

.. contents:: On this page
   :local:
   :depth: 3

Installation options
--------------------

The library can be installed in several ways:

Locally into a Python environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Library can be installed into a local Python environment (conda, venv, etc.):

.. code-block:: bash

   pip install oceanicospy

Google Colab
~~~~~~~~~~~~

Install the library in a Colab notebook cell usually at the beginning of the notebook. 

.. code-block:: bash

   !pip install oceanicospy

.. warning::
   You may need to restart the runtime after installation to resolve package conflicts with pre-installed libraries in Colab.

Verifying the installation
--------------------------
To verify that the library is installed correctly, you can run the following command in your terminal or Python environment:

.. code-block:: bash

   pip show oceanicospy

or 

.. code-block:: bash

   !pip show oceanicospy

if you are using Colab.

That command should display the package information, including the version number and installation path. Similar to:

.. code-block:: bash

   Name: oceanicospy
   Version: 0.1.0b6
   Summary: A Python library for oceanographic data analysis, numerical model preprocessing, and data retrieval
   Home-page:
   Author: 
   Author-email: OCEANICOS developer team <oceanicos_med@unal.edu.co>
   License: GNU GENERAL PUBLIC LICENSE

Importing the library
---------------------

Import the full package or individual subpackages:

.. code-block:: python

   import oceanicospy                                           # full package
   from oceanicospy.observations.pressure_sensors import RBR   # specific class
   from oceanicospy.analysis import WaveSpectralAnalyzer        # specific class

.. note::
   Avoid wildcard imports (``from oceanicospy.analysis import *``) in
   production scripts — they can shadow names from other libraries.
