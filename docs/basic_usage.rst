===========
Basic Usage
===========

The GWSim command-line tool can be used to generate synthetic data for gravitational-wave simulations.
This guide explains how to use the tool to generate data, including metadata, with support for checkpointing and file overwriting.

To generate data using a configuration file:

.. code-block:: console

    $ gwsim simulate config.yaml

This command reads the configuration from ``config.yaml`` and generates the specified data along with metadata files.

--------------------------
Overwriting Existing Files
--------------------------

By default, GWSim does not overwrite existing output files.
If a file already exists, the tool will raise an error and halt execution.

To force overwriting of existing files, use the ``--overwrite`` flag:

.. code-block:: console

    $ gwsim simulate config.yaml --overwrite

--------------------------
Checkpointing and Resuming
--------------------------

GWSim includes a built-in checkpointing mechanism that keeps track of the generation progress.
If the process is interrupted (e.g., due to system shutdown or error), it can resume from the last checkpointed state.

The checkpoint file is named ``checkpoint.json`` and is saved in the working directory specified by the ``working-directory`` field in the configuration.

To resume a previously interrupted generation process, simply rerun the same command:

.. code-block:: console

    $ gwsim simulate config.yaml

If a valid ``checkpoint.json`` exists, the tool will continue from where it left off.

-----------------------------------
Reproducing a subset of data segments
-----------------------------------

GWSim is designed for reproducible, resumable, and metadata-rich synthetic data generation workflows.

It is possible to reproduce a subset of data segments using the metadata files as

.. code-block:: console

    $ gwsim simulate some_metadata.yaml –output_dir output –metadata_dir metadata
