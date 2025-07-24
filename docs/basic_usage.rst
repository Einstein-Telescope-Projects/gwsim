===========
Basic Usage
===========

The ``gwsim`` command-line tool can be used to generate synthetic data for gravitational-wave simulations.
This guide explains how to use the tool to generate data, including metadata,
with support for checkpointing and file overwriting.

To generate data using a configuration file:

.. code-block:: console

    $ gwsim generate config.yaml --metadata

This command reads the configuration from ``config.yaml`` and generates the specified data along with metadata files.
The ``--metadata`` flag ensures that metadata describing the generated output is also created.

--------------------------
Overwriting Existing Files
--------------------------

By default, ``gwsim`` does not overwrite existing output files.
If a file already exists, the tool will raise an error and halt execution.

To force overwriting of existing files, use the ``--overwrite`` flag:

.. code-block:: console

    $ gwsim generate config.yaml --metadata --overwrite

--------------------------
Checkpointing and Resuming
--------------------------

``gwsim`` includes a built-in checkpointing mechanism that keeps track of the generation progress.
If the process is interrupted (e.g., due to system shutdown or error), it can resume from the last checkpointed state.

The checkpoint file is named ``checkpoint.json`` and is saved in the working directory specified by the ``working-directory`` field in the configuration.

To resume a previously interrupted generation process, simply rerun the same command:

.. code-block:: console

    $ gwsim generate config.yaml --metadata

If a valid ``checkpoint.json`` exists, the tool will continue from where it left off.

--------------------------
Example Configuration File
--------------------------

Below is an example ``config.yaml`` file used for generating white noise data:

.. code-block:: yaml

    generator:
      class: gwsim.noise.white_noise.WhiteNoise
      arguments:
        batch_size: 1
        duration: 4
        loc: 0.0
        max_samples: 10
        sampling_frequency: 16
        scale: 1.0
        seed: 0
        start_time: 123
    output:
      file_name: E1-MDC-{{ start_time }}-{{ duration }}.gwf
      arguments:
        channel: STRAIN
    working-directory: .

-----
Notes
-----

- The template syntax (e.g., ``{{ start_time }}``) in the output filename allows dynamic naming based on generation parameters.
- Ensure the working directory is writable and that sufficient disk space is available for output files and checkpointing.

``gwsim`` is designed for reproducible, resumable, and metadata-rich synthetic data generation workflows.
