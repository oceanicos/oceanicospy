.. _dev-setup:

Developer guide
===============

Follow this guide to set up a local development environment for contributing to **oceanicospy**.
The workflow is based on the standard GitHub fork-and-pull-request model.

.. contents:: On this page
   :local:
   :depth: 2

1. Create a GitHub account
--------------------------

Sign up at https://github.com/signup. You can create an educational account linked to your
university email.

2. Fork the repository
----------------------

Go to https://github.com/oceanicos-dev-org/oceanicospy and click **Fork** (top-right corner).
This creates your own copy under your GitHub account:

.. code-block:: text

   https://github.com/YOUR-USERNAME/oceanicospy

.. hint::

   Forking creates an independent copy of a repository, allowing you to freely experiment
   with changes without affecting the original project.

3. Set up SSH authentication
-----------------------------

SSH is the recommended protocol for pushing and pulling commits. Generate a key and link it
to your GitHub account if you have not done so:

https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account

4. Clone your fork
------------------

Navigate to the local folder where you want to work and run:

.. code-block:: bash

   git clone git@github.com:YOUR-USERNAME/oceanicospy.git
   cd oceanicospy

Verify that your local clone points to your fork:

.. code-block:: bash

   git remote -v

You should see your fork listed as ``origin``. Add the upstream remote so you can pull in
changes from the original repository:

.. code-block:: bash

   git remote add upstream git@github.com:oceanicos-dev-org/oceanicospy.git

5. Install in editable mode
----------------------------

.. hint::

   A helper script is included to automate the setup process for developers. From the repository root, run:

   .. code-block:: bash

      bash setup_dev.sh

   This creates a ``.venv`` virtual environment, installs the package with all ``dev``
   dependencies in editable mode, and registers a Jupyter kernel named ``oceanicospy-dev``.
   Activate the environment afterwards with:

   .. code-block:: bash

      source .venv/bin/activate

   To use a specific Python version, pass it via the ``PYTHON`` variable:

   .. code-block:: bash

      PYTHON=python3.11 bash setup_dev.sh

   The manual steps below are equivalent if you prefer a different setup.

Install the package in *editable* mode so that local changes are reflected immediately
without reinstalling:

.. code-block:: bash

   pip install -e .

To also install dependencies for building the documentation and running tests, use the
``docs`` and ``dev`` optional groups defined in ``pyproject.toml``:

.. code-block:: bash

   pip install -e ".[docs]"
   pip install -e ".[dev]"

This installs:

- ``sphinx`` — documentation generator.
- ``sphinx_rtd_theme`` — Read the Docs HTML theme.
- ``sphinx-book-theme`` — alternative book-style theme.
- ``nbsphinx`` — embeds Jupyter notebooks in Sphinx docs.
- ``myst-nb`` — MyST Markdown and notebook support for Sphinx.
- ``pytest`` — testing framework.
- ``build`` — PEP 517 build frontend.
- ``twine`` — utility for publishing packages to PyPI.

Next steps
----------

Once your environment is ready, see :doc:`contributing` for the development workflow
(branching, committing, and opening pull requests).
