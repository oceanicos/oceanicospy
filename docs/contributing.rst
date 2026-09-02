Contributing
============

.. contents:: On this page
   :local:
   :depth: 2

Types of contributions
----------------------

Report bugs
~~~~~~~~~~~

Report bugs by opening an issue at https://github.com/oceanicos-dev-org/oceanicospy/issues.

A good bug report includes:

- Your operating system name and version.
- Python version and relevant package versions (``pip show oceanicospy``).
- The minimal code snippet that reproduces the problem.
- The full traceback or error message.

Fix bugs
~~~~~~~~

Browse the open issues on GitHub. Issues tagged **bug** are
good starting points for first-time contributors.

Implement features
~~~~~~~~~~~~~~~~~~

Look for open issues tagged **enhancement** and **feature**.
If you have a new idea, open an issue first to discuss it before writing code.

Improve documentation
~~~~~~~~~~~~~~~~~~~~~

The documentation lives under ``docs/`` and is built with `Sphinx <https://www.sphinx-doc.org>`_.
Improvements to docstrings, tutorials, examples, or these pages are always welcome.

Development workflow
--------------------

After installing the library in editable mode (see :doc:`dev_setup`), you can work on your changes locally and test them interactively.
In the forked repository, follow these steps to contribute your changes back to the main project:

1. Create a feature branch
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Never commit directly to ``main`` or ``integration``. Create a descriptive branch instead:

.. code-block:: bash

   git branch YOUR-USERNAME/my-feature
   git checkout YOUR-USERNAME/my-feature

Branch names should start with your username while the team is getting familiar
with the workflow (e.g. ``jdoe/add-feature1``).

2. Make changes and commit
~~~~~~~~~~~~~~~~~~~~~~~~~~

Stage and commit your changes with a short, descriptive message:

.. code-block:: bash

   git add path/to/changed_file.py
   git commit -m "add some particular feature to certain module"

Push your branch to your fork:

.. code-block:: bash

   git push origin YOUR-USERNAME/my-feature

3. Open a pull request
~~~~~~~~~~~~~~~~~~~~~~~

On GitHub, navigate to your fork and click **Contribute → Open a pull request**.
GitHub will show a diff between your branch and ``oceanicospy:main``.

Provide a clear title and description explaining *what* changed and *why*.
A maintainer will review and merge your contribution.


Pull request checklist
----------------------

Before submitting a pull request, please verify that:

- Tests pass locally (``pytest``).
- New functionality includes corresponding tests.
- Docstrings are updated for any changed public API.
- The entry in ``CHANGELOG.md`` is updated.
- The PR description explains *what* changed and *why*.

Code of conduct
---------------

This project follows a standard open-source code of conduct: be respectful, inclusive,
and constructive in all interactions.
