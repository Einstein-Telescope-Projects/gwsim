from __future__ import annotations

import os
import tempfile

import bilby
import numpy as np
import pandas as pd
import pytest
from gwsim.signal.cbc_signal import CBCSignal


@pytest.fixture
def temp_population_file_single_event():
    # Parameters for a single BBH event
    mass1 = 30.0
    mass2 = 32.0
    spin1x = 0.0
    spin1y = 0.0
    spin1z = 0.0
    spin2x = 0.0
    spin2y = 0.0
    spin2z = 0.0
    ra = 1.97
    dec = -1.21
    pol = 1.6
    iota = 2.68
    phase = 0.0
    dl = 1400.0
    redshift = 0.3
    geocent_time = 4300.0  # Placed to span across two frames
    flow = 2.0
    duration = bilby.gw.utils.calculate_time_to_merger(flow, mass1 * (1 + redshift), mass2 * (1 + redshift), safety=1.2)

    data = {
        "mass_1": [mass1],
        "mass_2": [mass2],
        "geocent_time": [geocent_time],
        "luminosity_distance": [dl],
        "spin_1x": [spin1x],
        "spin_1y": [spin1y],
        "spin_1z": [spin1z],
        "spin_2x": [spin2x],
        "spin_2y": [spin2y],
        "spin_2z": [spin2z],
        "right_ascension": [ra],
        "declination": [dec],
        "polarization_angle": [pol],
        "iota": [iota],
        "phase": [phase],
        "redshift": [redshift],
        "duration": [duration],
    }
    df = pd.DataFrame(data)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
        df.to_csv(temp_file.name, index=False)
    yield temp_file.name
    os.remove(temp_file.name)


@pytest.fixture
def temp_population_file_single_event_long():
    # Parameters for a single BBH event
    mass1 = 10
    mass2 = 10.0
    spin1x = 0.0
    spin1y = 0.0
    spin1z = 0.0
    spin2x = 0.0
    spin2y = 0.0
    spin2z = 0.0
    ra = 1.97
    dec = -1.21
    pol = 1.6
    iota = 2.68
    phase = 0.0
    dl = 1400.0
    redshift = 0.3
    geocent_time = 3900.0  # Placed to span across two frames
    flow = 2.0
    duration = bilby.gw.utils.calculate_time_to_merger(flow, mass1 * (1 + redshift), mass2 * (1 + redshift), safety=1.2)

    data = {
        "mass_1": [mass1],
        "mass_2": [mass2],
        "geocent_time": [geocent_time],
        "luminosity_distance": [dl],
        "spin_1x": [spin1x],
        "spin_1y": [spin1y],
        "spin_1z": [spin1z],
        "spin_2x": [spin2x],
        "spin_2y": [spin2y],
        "spin_2z": [spin2z],
        "right_ascension": [ra],
        "declination": [dec],
        "polarization_angle": [pol],
        "iota": [iota],
        "phase": [phase],
        "redshift": [redshift],
        "duration": [duration],
    }
    df = pd.DataFrame(data)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
        df.to_csv(temp_file.name, index=False)
    yield temp_file.name
    os.remove(temp_file.name)


def test_signal_continuity_across_segments(temp_population_file_single_event):
    """Test that a CBC signal that spans multiple data segments maintain continuity"""
    approximant = "IMRPhenomD"
    flow = 2.0
    sampling_frequency = 4096.0
    frame_duration = 4096.0
    detector_names = ["E1"]
    earth_rotation = True
    time_dependent_timedelay = True
    start_time = 0.0

    gen = CBCSignal(
        detector_names=detector_names,
        population_file=temp_population_file_single_event,
        approximant=approximant,
        flow=flow,
        sampling_frequency=sampling_frequency,
        duration=frame_duration,
        earth_rotation=earth_rotation,
        time_dependent_timedelay=time_dependent_timedelay,
        start_time=start_time,
    )

    # Generate two adjacent frames
    frame1 = gen.next()
    gen.update_state()
    frame2 = gen.next()

    # Concatenate the frames
    concat_data = np.concatenate((frame1[0], frame2[0]))

    # Generate reference long frame
    ref_duration = 2 * frame_duration
    ref_start_time = 0.0
    ref_end_time = ref_duration
    ref_data = np.zeros(int(ref_duration * sampling_frequency))

    # Load population DF
    df = pd.read_csv(temp_population_file_single_event)
    parameters = df.iloc[0].to_dict()

    # Get hp, hc
    hp, hc = gen.get_polarization_at_time(parameters, gen.waveform_arguments)

    # Inject into reference
    det = gen.detectors[0]
    ref_data = gen.inject_signal_in_frame(
        hp, hc, parameters, det, ref_data, sampling_frequency, ref_start_time, ref_end_time
    )

    # Check if they match
    np.testing.assert_allclose(
        concat_data, ref_data, atol=1e-26, rtol=1e-3, err_msg="Signal is not continuous across adjacent frames."
    )


def test_earth_rotation_option(temp_population_file_single_event_long):
    approximant = "IMRPhenomD"
    flow = 2.0
    sampling_frequency = 4096.0
    frame_duration = 4096.0
    detector_names = ["E1"]
    start_time = 0.0

    # Generate with earth_rotation=True
    gen_with_rotation = CBCSignal(
        detector_names=detector_names,
        population_file=temp_population_file_single_event_long,
        approximant=approximant,
        flow=flow,
        sampling_frequency=sampling_frequency,
        duration=frame_duration,
        earth_rotation=True,
        earth_rotation_timestep=100.0,
        time_dependent_timedelay=True,
        start_time=start_time,
    )
    frame_with_rotation = gen_with_rotation.next()[0]

    # Generate with earth_rotation=False
    gen_without_rotation = CBCSignal(
        detector_names=detector_names,
        population_file=temp_population_file_single_event_long,
        approximant=approximant,
        flow=flow,
        sampling_frequency=sampling_frequency,
        duration=frame_duration,
        earth_rotation=False,
        start_time=start_time,
    )
    frame_without_rotation = gen_without_rotation.next()[0]

    # Check that they are different (Earth rotation should affect the signal)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(frame_with_rotation, frame_without_rotation, atol=1e-26, rtol=1e-5)
