============
Installation
============

We recommend using `uv`_, a modern Python package manager that is **blazingly fast**, **reproducible**, and easy to use.
It serves as a drop-in replacement for ``pip``, ``pip-tools``, and ``virtualenv``.
Follow the `installation guide`_ to install ``uv``.

.. _uv: https://github.com/astral-sh/uv

.. _installation guide: https://docs.astral.sh/uv/getting-started/installation/

----------------------
Installing the Package
----------------------

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Option 1: From PyPI (not available yet)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Create a virtual environment using uv:

.. code-block:: console

    $ uv venv .venv
    $ source .venv/bin/activate

2. Install the package from PyPI:

.. code-block:: console

    $ uv pip install gwsim

^^^^^^^^^^^^^^^^^^^^^
Option 2: From Source
^^^^^^^^^^^^^^^^^^^^^

If you want the latest development version or plan to contribute:

1. Clone the repository:

.. code-block:: console

   $ git clone https://gitlab.et-gw.eu/eluminat/software/gwsim.git
   $ cd gwsim

2. Create and activate a virtual environment:

.. code-block:: console

    $ uv venv .venv
    $ source .venv/bin/activate

3. Install the package in editable mode with dependencies:

.. code-block:: console

    $ uv pip install -e .

--------------------------
Verifying the Installation
--------------------------

To confirm everything is working, try:

.. code-block:: console

    $ python -m gwsim
