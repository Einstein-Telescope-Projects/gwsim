==================
Generating ET data
==================

This guide shows how to use GWSim to create one day of mock data for the Einstein Telescope (ET).
It includes realistic gravitational wave signals and detector noise.

We will use the ET triangular configuration located in the Meuse-Rhine Euregion as an example.
For other detector configurations, see `SECTION X`.

The data will be saved in :code:`.gwf` (frame file) format at a sampling rate of 4096 Hz.
These files can be read and manipulated with the `gwpy <https://gwpy.github.io/>`_ package, as explained in `SECTION Y`.

-------------------------
Generating detector noise
-------------------------

Detector noise can be generated using configuration files located in :code:`gwsim/examples/noise`.
An example configuration for producing one day of ET noise data is provided in :code:`ET_Triangle_EMR_noise_config.yaml`:

.. code-block:: yaml

    globals:
      working-directory: .
      simulator-arguments:
        sampling-frequency: 4096
        max-samples: 22
        duration: 4096
        start_time: 1577491218
      output-arguments: {}
      output-directory: ET_Triangle_EMR_noise
      metadata-directory: ET_Triangle_EMR_noise/metadata
    simulators:
      noise:
        class: ColoredNoise
        arguments:
          psd_file: ET_10_full_cryo_psd.txt
          detectors:
            - E1_Triangle_EMR
            - E2_Triangle_EMR
            - E3_Triangle_EMR
          seed: 0
        output:
          file_name: "E-{{ detector }}_STRAIN_NOISE-{{ start_time }}-{{ duration }}.gwf"
          arguments:
            channel: STRAIN

This configuration file creates 22 frame files per detector (E1, E2, E3 in the ET triangular setup).
Each file covers 4096 seconds, for a total of slightly more than 24 hours starting from January 1, 2030.

Frame files are saved in the :code:`./ET_Triangle_EMR_noise` folder, and metadata in :code:`./ET_Triangle_EMR_noise/metadata`.
Noise is simulated from the `ET_10_full_cryo_psd` sensitivity curve used in the `CoBA science study <https://iopscience.iop.org/article/10.1088/1475-7516/2023/07/068>`_ and available at `LINK_TO_NOISE_CURVES_FOLDER`.
A lower cutoff frequency of 2 Hz is used.

.. note::
    Before generating the dataset, ensure that sufficient disk space is available.
    Each :code:`.gwf` file is about 125 MB, so you will need around 8.5 GB total for one day of noise files and metadata.

    The noise generation will take around `ADD_RUNTIME_ESTIMATE` on a CPU `ADD_STATISTICS`.
    To generate the dataset in batches, or to resume a generation that has been interrupted (e.g., due to system shutdown), check `SECTION W`.

To generate the ET noise data, run this command in your working directory:

.. code-block:: console

    $ gwsim simulate ET_Triangle_EMR_noise_config.yaml

----------------------
Generating CBC signals
----------------------

Signals from compact binary coalescences (CBCs) can be generated using configuration files located in :code:`gwsim/examples/CBC_signals`.
An example configuration for producing one day of ET data containing binary black hole (BBH) signals from a realistic population is provided in :code:`ET_Triangle_EMR_BBH_config.yaml`:

.. code-block:: yaml

    globals:
      working-directory: .
      simulator-arguments:
        sampling-frequency: 4096
        max-samples: 22
        duration: 4096
        start_time: 1577491218
      output-arguments: {}
      output-directory: ET_Triangle_EMR_BBH
      metadata-directory: ET_Triangle_EMR_BBH/metadata
    simulators:
      noise:
        class: CBCSimulator
        arguments:
          population_file: 18321_1yrCatalogBBH.h5
          waveform_model: IMRPhenomXPHM,
          waveform_arguments: {'earth_rotation': True, 'time_dependent_timedelay': True}
          minimum_frequency: 2,
          detectors:
            - E1_Triangle_EMR
            - E2_Triangle_EMR
            - E3_Triangle_EMR
        output:
          file_name: "E-{{ detector }}_STRAIN_BBH-{{ start_time }}-{{ duration }}.gwf"
          arguments:
            channel: STRAIN

Similar to the noise example, this creates 22 frame files per detector.
Each file lasts 4096 seconds, for a total of slightly more than 24 hours starting from January 1, 2030.

Frame files are saved in the :code:`./ET_Triangle_EMR_BBH` folder, and metadata in :code:`./ET_Triangle_EMR_BBH/metadata`.
BBHs are injected in zero noise from the `18321_1yrCatalogBBH` population file used in the `CoBA science study <https://iopscience.iop.org/article/10.1088/1475-7516/2023/07/068>`_ and `publicly available <https://apps.et-gw.eu/tds/?content=3&r=18321>`_ .
The `IMRPhenomXPHM <https://journals.aps.org/prd/abstract/10.1103/PhysRevD.103.104056>`_ waveform model is used, with a lower cutoff frequency of 2 Hz and including Earth rotation effects.

The configuration file to generate one day of ET data containing binary neutron stars (BNS) signals can be found at :code:`ET_Triangle_EMR_BBH_config.yaml`, with the following arguments:

- population_file: `18321_1yrCatalogBNS`, used in the `CoBA science study <https://iopscience.iop.org/article/10.1088/1475-7516/2023/07/068>`_ and `publicly available <https://apps.et-gw.eu/tds/?content=3&r=18321>`_ .
- waveform_model: `IMRPhenomPv2_NRTidalv2 <https://journals.aps.org/prd/abstract/10.1103/PhysRevD.100.044003>`_ , with a lower cutoff frequency of 2 Hz and including Earth rotation effects.

Files are saved in :code:`./ET_Triangle_EMR_BNS`, metadata in :code:`./ET_Triangle_EMR_BNS/metadata`.

.. note::
    Before generating the dataset, ensure that sufficient disk space is available.
    Each :code:`.gwf` file is about `ADD_ESTIMATE`, so you will need around `ADD_ESTIMATE` total for one day of signal files and metadata, for each kind of source.

    The signals generation will take around `ADD_RUNTIME_ESTIMATE` on a CPU `ADD_STATISTICS` for each kind of source.
    To generate the dataset in batches, or to resume a generation that has been interrupted (e.g., due to system shutdown), check `SECTION W`.

To generate the ET signals data from CBCs, run this command in your working directory:

.. tabs::

   .. tab:: BBH

      .. code-block:: console

          $ gwsim simulate ET_Triangle_EMR_BBH_config.yaml

   .. tab:: BNS

      .. code-block:: console

          $ gwsim simulate ET_Triangle_EMR_BNS_config.yaml


----------------------------
Generating detector glitches
----------------------------


---------------------------------------
Using different detector configurations
---------------------------------------

Several detector configurations are available at :code:`gwsim/detector/detectors` for ET as in :code:`.interferometer` files:

- Triangular configuration (Meuse–Rhine Euregion)

  - :code:`E1_Triangle_EMR`
  - :code:`E2_Triangle_EMR`
  - :code:`E3_Triangle_EMR`

- Triangular configuration (Sardinia)

  - :code:`E1_Triangle_Sardinia`
  - :code:`E2_Triangle_Sardinia`
  - :code:`E3_Triangle_Sardinia`

- 2L aligned configuration (Sardinia + Meuse–Rhine Euregion)

  - :code:`E1_2L_Aligned_Sardinia`
  - :code:`E2_2L_Aligned_EMR`

- 2L misaligned configuration (Sardinia + Meuse–Rhine Euregion)

  - :code:`E1_2L_Misaligned_Sardinia`
  - :code:`E2_2L_Misaligned_EMR`

The coordinates of the Meuse-Rhine Euregion and Sardinia locations follows from the `CoBA science study <https://iopscience.iop.org/article/10.1088/1475-7516/2023/07/068>`_ .

To use a specific detector configuration, update the detectors list in your :code:`.yaml` configuration file.
For example, to generate data for 2L aligned configuration of ET:

.. code-block:: yaml

      detectors:
        - E1_2L_Aligned_Sardinia
        - E2_2L_Aligned_EMR

Bear in mind that, although not compulsory for the data generation, it is a good practice to keep different dataset in separate folders.
For this reason, it always recommendable to change the :code:`output-directory` and :code:`metadata-directory` arguments in your configuration file accordingly.

.. note::
    It is not necessary to specify all the detectors forming a configuration.
    For just E1 data:

    .. code-block:: yaml

          detectors:
            - E1_2L_Aligned_Sardinia

----------------------------------
Using different sensitivity curves
----------------------------------

Several ET sensitivity curves are available at :code:`gwsim/detector/noise_curves`, following the `CoBA science study <https://iopscience.iop.org/article/10.1088/1475-7516/2023/07/068>`_ .

Likewise the detector configuration, to use a specific detector sensitivity curve it is sufficient to update your :code:`.yaml` configuration file.
For example, to generate data for a 15 km arms interferometer without the xylophone configuration (only with the high-frequency instrument):

.. code-block:: yaml

      psd_file: ET_15_HF_psd.txt

Note that the detector geometries implemented in GWsim assume 10 km arms interferometers for the ET triangular configuration, and 15 km arms interferometers for the ET 2L configuration.
For consistency, choose the sensitivity curves accordingly.

----------------------------------
Using different signal populations
----------------------------------


-----------------------
Checking metadata files
-----------------------
